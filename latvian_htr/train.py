"""
Neironu tīkla apmācības skripts (ускоренная версия).

Kā skriet:
    # 1) один раз сгенерировать набор (все ядра процессора):
    python make_synth_dataset.py --out data/synth_30k --n 30000
    # 2) обучать на нём с нуля (500 примеров автоматически откладываются для проверки):
    python train.py --synth-data data/synth_30k --steps 20000
    python train.py --synth-data data/synth_30k --real-data data/real_dataset   # + реальные сканы
    python train.py --synth-data data/synth_30k --x-stride 4                    # ещё ~2x быстрее (см. model.py)
    # без --synth-data синтетика генерируется на лету (каждая строка новая, но нужен мощный CPU)

Kas ir "apmācība"? Mēs atkārtojam vienu un to pašu ciklu daudzas reizes:
1) tīklam rādām attēlu ar tekstu;
2) tīkls mēģina to nolasīt un pieļauj kļūdu;
3) aprēķina, cik liela ir kļūda (šo skaitli sauc par zudumiem);
4) nedaudz pielāgojam tīkla iekšējos skaitļus (tā "svarus"), lai nākamreiz kļūda būtu mazāka.

Что ускорено по сравнению с прошлой версией:
  * данные: готовый набор читается с диска в нескольких воркерах, воркеры НЕ перезапускаются каждую эпоху;
  * bucketing: в батче строки похожей ширины -> почти нет паддинга (раньше ~45% счёта уходило в пустоту);
  * mixed precision (fp16 на тензорных ядрах RTX 3080) + channels_last + cudnn.benchmark + fused Adam;
  * никаких синхронизаций CPU<->GPU на каждом шаге ради печати loss;
  * в логе видно, сколько времени шаг ждёт данные ("data wait") - сразу понятно, где узкое место;
  * при каждом сохранении чекпоинта - проверка (loss и CER) на примерах, на которых сеть не учится.
"""

import argparse   # komandrindas iestatījumu nolasīšana (--steps 2000 u. tml.)
import os         # darbs ar mapēm
import random     # nejauši skaitļi
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from alphabet import NUM_CLASSES, decode_greedy
from dataset import BucketBatchSampler, LineFolderDataset, SyntheticBucketStream, collate_batch
from evaluate import cer
from model import CRNN, load_checkpoint, save_checkpoint


class PlateauScheduler:
    """Уменьшает lr, если средний loss перестал улучшаться.
    loss копится прямо на GPU (без .item() на каждом шаге), в число переводится раз в `every` шагов."""

    def __init__(self, optimizer, every=200, factor=0.5, patience=3, min_lr=1e-6):
        self.optimizer = optimizer
        self.every = every  # как часто (в шагах) проверять loss
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
        )
        self.total, self.count = 0.0, 0

    def update(self, step, loss):
        self.total = self.total + loss.detach()
        self.count += 1
        if step % self.every == 0:
            avg = float(self.total) / self.count
            self.scheduler.step(avg)
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"[scheduler] step {step}: avg loss={avg:.4f}, lr={lr:.2e}")
            self.total, self.count = 0.0, 0


def default_workers() -> int:
    # Чтения PNG с диска хватает 4-6 воркеров; генерации на лету нужно больше (до 12).
    n = os.cpu_count() or 2
    return max(1, min(8, n - 2))


def loader_kwargs(args, device) -> dict:
    # persistent_workers: воркеры живут всё обучение (на Windows их перезапуск стоит секунды).
    # prefetch_factor: каждый воркер заранее готовит несколько батчей, пока GPU занята.
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, timeout=600)
    return kw


def build_folder_loader(args, device):
    """Режим «готовый набор»: обучаемся по кругу на картинках из --synth-data (+ --real-data)."""
    synth = LineFolderDataset(args.synth_data, as_uint8=True)
    t0 = time.perf_counter()
    synth_w = synth.widths()
    print(f"Synthetic set: {len(synth)} lines from {args.synth_data} (widths read in {time.perf_counter() - t0:.1f}s)")

    # Проверочные примеры: отдельная папка --val-data или --val-size примеров, отложенных из набора.
    # На них сеть НЕ учится - по ним видно, учится ли она читать или просто запоминает картинки.
    order = list(range(len(synth)))
    random.Random(12345).shuffle(order)
    if args.val_data:
        val_set = LineFolderDataset(args.val_data, as_uint8=True)
        train_idx = order
    else:
        n_val = min(args.val_size, len(synth) // 10)
        val_set = Subset(synth, order[:n_val])
        train_idx = order[n_val:]
    train_idx.sort()

    parts, widths = [Subset(synth, train_idx)], [synth_w[i] for i in train_idx]
    if args.real_data:
        # Реальные строки просто добавляются к набору (как раньше ConcatDataset).
        real = LineFolderDataset(args.real_data, as_uint8=True)
        parts.append(real)
        widths += real.widths()
        print(f"Real data: {len(real)} lines added to training set")
    train_set = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    print(f"Training on {len(train_set)} lines, validation on {len(val_set)} lines")

    sampler = BucketBatchSampler(widths, args.batch_size, seed=args.seed)
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate_batch,
                        **loader_kwargs(args, device))

    # Проверочные батчи готовим один раз и держим в памяти (их немного).
    val_items = sorted((val_set[i] for i in range(len(val_set))), key=lambda s: s[0].shape[-1])
    val_batches = [collate_batch(val_items[i:i + args.batch_size])
                   for i in range(0, len(val_items), args.batch_size)]
    return loader, val_batches, len(train_set)


def build_stream_loader(args, device):
    """Режим «бесконечная генерация»: синтетика рисуется на лету, каждая строка новая."""
    real = LineFolderDataset(args.real_data) if args.real_data else None
    frac = 0.0
    if real is not None:
        frac = args.real_fraction
        if frac is None:
            frac = len(real) / (args.synth_per_epoch + len(real))
        print(f"Real data: {len(real)} lines, share in batches = {frac:.3f}")
    stream = SyntheticBucketStream(args.batch_size, real=real, real_fraction=frac)
    # batch_size=None: батчи уже собраны в воркерах.
    return DataLoader(stream, batch_size=None, **loader_kwargs(args, device)), None


@torch.no_grad()
def validate(model, batches, device, use_amp, ctc_loss):
    """Средний loss и CER (доля ошибочных символов, жадное декодирование) на проверочных батчах."""
    model.eval()
    total_loss, n = 0.0, 0
    refs, hyps = [], []
    for images, targets, target_lengths, widths, texts in batches:
        images = images.to(device).float().div_(255.0)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            log_probs = model(images)
        input_lengths = model.output_lengths(widths)
        loss = ctc_loss(log_probs, targets.to(device), input_lengths, target_lengths)
        total_loss += loss.item() * len(texts)
        n += len(texts)
        best = log_probs.argmax(dim=2).t().cpu()          # [B, T]
        for row, length, text in zip(best, input_lengths.tolist(), texts):
            hyps.append(decode_greedy(row[:length].tolist()))
            refs.append(text)
    model.train()
    return total_loss / max(n, 1), cer(refs, hyps)


def train(args):
    """Galvenā apmācības funkcija. args ir lietotāja nodotie iestatījumi."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Calculating on cuda: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True          # подобрать самые быстрые свёртки для каждой ширины
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 на Ampere (для частей вне autocast)
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("!" * 70)
        print("ВНИМАНИЕ: CUDA недоступна, обучение идёт на ПРОЦЕССОРЕ - это в десятки раз медленнее.")
        print("Проверьте: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"")
        print("Если в версии нет '+cu...', установлен torch без CUDA - переустановите с pytorch.org.")
        print("!" * 70)
        torch.set_num_threads(os.cpu_count())
    use_amp = device.type == "cuda" and not args.no_amp

    # --- данные ---
    val_batches = None
    if args.synth_data:
        loader, val_batches, epoch_size = build_folder_loader(args, device)
    else:
        loader, epoch_size = build_stream_loader(args, device)
    print(f"Data workers: {args.num_workers}")

    # --- модель ---
    model = CRNN(num_classes=NUM_CLASSES, x_stride=args.x_stride)
    if args.resume:
        state, ckpt_stride = load_checkpoint(args.resume)
        model.load_state_dict(state)
        print(f"Загружен чекпоинт: {args.resume} (x_stride в чекпоинте={ckpt_stride}, сейчас={args.x_stride})")
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)   # быстрее для свёрток в fp16

    # Optimizators - «labotājs». fused=True - весь шаг Adam одним CUDA-ядром.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=device.type == "cuda")
    plateau = PlateauScheduler(optimizer)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    # GradScaler: в fp16 маленькие градиенты могут "обнулиться"; scaler умножает loss перед backward
    # и делит градиенты обратно. Если градиенты стали inf/nan - шаг пропускается автоматически.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (fp16): {'on' if use_amp else 'off'}, x_stride={args.x_stride}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    model.train()
    batches = iter(loader)
    t_last, data_time = time.perf_counter(), 0.0
    bad_total = 0
    while step < args.steps:
        t0 = time.perf_counter()
        images, targets, target_lengths, widths, texts = next(batches)
        data_time += time.perf_counter() - t0

        # uint8 -> float [0..1] уже на GPU (передаём по шине в 4 раза меньше байт).
        images = images.to(device, non_blocking=True).float().div_(255.0)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            log_probs = model(images)  # [T, B, C] float32 (см. model.forward)
        T = log_probs.size(0)

        # Attēli tika papildināti (padding); katram piemēram reālais soļu skaits pēc CNN.
        input_lengths = model.output_lengths(widths)
        if input_lengths.max().item() > T:
            raise RuntimeError(f"input_lengths bigger than T: max={input_lengths.max().item()}, T={T}")
        bad = (input_lengths < target_lengths).sum().item()   # считается на CPU, GPU не ждём
        if bad:
            bad_total += bad
            print(f"WARNING: {bad}/{len(texts)} примеров с T < длины текста (loss обнулён)")

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                       # вернуть настоящие градиенты перед обрезкой
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        plateau.update(step, loss)
        if step % args.log_every == 0:
            now = time.perf_counter()
            dt = now - t_last
            epoch_info = f"epoch {step * args.batch_size / epoch_size:.1f}  " if epoch_size else ""
            print(f"{step}/{args.steps}  {epoch_info}failure={loss.item():.4f}  "
                  f"{dt / args.log_every * 1000:.0f} ms/step  "
                  f"(data wait {data_time / dt * 100:.0f}%)  "
                  f"ETA {(args.steps - step) * dt / args.log_every / 60:.0f} min")
            t_last, data_time = now, 0.0
        if step % args.checkpoint_every == 0 or step == args.steps:
            ckpt_path = os.path.join(args.checkpoint_dir, f"crnn_step{step}.pt")
            save_checkpoint(model, ckpt_path, step)
            print(f"saved checkpoint -> {ckpt_path}")
            if val_batches:
                v_loss, v_cer = validate(model, val_batches, device, use_amp, ctc_loss)
                print(f"[validation] step {step}: loss={v_loss:.4f}  CER={v_cer:.3f}")
                t_last = time.perf_counter()     # время проверки не считаем в ms/step

    print("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)         # learning steps
    parser.add_argument("--batch-size", type=int, default=64)      # examples in one step
    parser.add_argument("--lr", type=float, default=1e-3)          # the speed of learning (1e-3 = 0.001)
    parser.add_argument("--synth-data", type=str, default=None,
                        help="папка с готовой синтетикой (labels.csv + images/), см. make_synth_dataset.py. "
                             "Если не задана - синтетика генерируется на лету")
    parser.add_argument("--val-data", type=str, default=None, help="отдельная папка для проверки (необязательно)")
    parser.add_argument("--val-size", type=int, default=500,
                        help="сколько примеров отложить из --synth-data для проверки, если --val-data не задана")
    parser.add_argument("--synth-per-epoch", type=int, default=5000,
                        help="только для режима генерации на лету: нужен, чтобы посчитать долю реальных данных")
    parser.add_argument("--real-data", type=str, default=None, help="path to the folder with real scans (when they appear)")
    parser.add_argument("--real-fraction", type=float, default=None,
                        help="только для режима генерации на лету: доля реальных строк в батчах (0..1)")
    parser.add_argument("--num-workers", type=int, default=default_workers())  # parallel data helpers
    parser.add_argument("--x-stride", type=int, default=2, choices=[2, 4],
                        help="сжатие ширины в CNN: 4 = примерно в 2 раза быстрее (см. model.py)")
    parser.add_argument("--no-amp", action="store_true", help="выключить mixed precision (fp16)")
    parser.add_argument("--log-every", type=int, default=500)       # how often to print an error
    parser.add_argument("--checkpoint-every", type=int, default=1000)  # how often save weights
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")  # where to save
    parser.add_argument("--resume", type=str, default=None, help="чекпоинт для дообучения")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Fiksējam «nejaušību», lai palaišanu varētu atkārtot ar tādu pašu rezultātu.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)

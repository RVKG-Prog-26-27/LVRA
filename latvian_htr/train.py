"""
Training script for the neural network.

How to run:
    # 1) generate a synthetic set once (uses all CPU cores):
    python make_synth_dataset.py --out data/synth_30k --n 30000
    # 1b) optional: pack it for reading from RAM (after EVERY new generation):
    python packed_lines.py data/synth_30k
    # 2) train on it from scratch (500 examples are held out automatically for validation):
    python train.py --synth-data data/synth_30k --steps 20000
    python train.py --synth-data data/synth_30k --real-data data/real_dataset   # + real scans
    python train.py --synth-data data/synth_30k --x-stride 4                    # ~2x faster (see model.py)
    # without --synth-data the synthetic lines are generated on the fly (every line is new, but a strong CPU is needed)

What is "training"? The same cycle is repeated many times:
1) the network is shown an image with text;
2) it tries to read it and makes a mistake;
3) the size of the mistake is calculated (this number is called the loss);
4) the inner numbers of the network (its "weights") are adjusted a little so that the next mistake is smaller.

Real data and validation: --real-data lines are split by split_real() (dataset.py): --real-val-fraction of them
(10% by default) is NEVER used for training and serves as the honest validation set. The best checkpoint
(crnn_best.pt) is chosen by the CER on these real lines; if there are no real lines, by the synthetic validation.
evaluate.py uses the same split.

Speed:
  * data: a ready-made set is read from disk by several workers that are NOT restarted every epoch;
  * bucketing: a batch holds lines of similar width -> almost no padding;
  * mixed precision (fp16 on tensor cores) + channels_last + cudnn.benchmark + fused Adam;
  * no CPU<->GPU synchronisation at every step just to print the loss;
  * the log shows how long a step waits for data ("data wait") - the bottleneck is visible at once;
  * at every checkpoint the loss and CER are measured on examples the network does not learn from.
"""

import argparse   # reads command line settings (--steps 2000 etc.)
import os
import random
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from alphabet import NUM_CLASSES, decode_greedy
from dataset import (DEFAULT_REAL_VAL_FRACTION, BucketBatchSampler, LineFolderDataset, SyntheticBucketStream,
                     collate_batch, split_real)
from metrics import cer, wer
from model import CRNN, check_num_classes, load_checkpoint, save_checkpoint
from packed_lines import open_line_dataset   # data/<set>/packed/ -> reading from RAM (see packed_lines.py)
from synthetic_data import load_words, make_sample


class PlateauScheduler:
    """Lowers the learning rate when the average loss stops improving.
    The loss is accumulated on the GPU (no .item() at every step) and converted to a number once per `every` steps."""

    def __init__(self, optimizer, every=200, factor=0.5, patience=3, min_lr=1e-6):
        self.optimizer = optimizer
        self.every = every  # how often (in steps) the loss is checked
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
    # Reading PNG files from disk needs 4-6 workers; generation on the fly needs more (up to 12).
    n = os.cpu_count() or 2
    return max(1, min(8, n - 2))


def loader_kwargs(args, device) -> dict:
    # persistent_workers: workers live during the whole training (restarting them costs seconds on Windows).
    # prefetch_factor: every worker prepares several batches ahead while the GPU is busy.
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, timeout=600)
    return kw


def make_val_batches(items, batch_size: int) -> list:
    """Validation batches are prepared once and kept in memory (there are few of them).
    items: anything indexable that returns (uint8 image, text)."""
    items = sorted((items[i] for i in range(len(items))), key=lambda s: s[0].shape[-1])
    return [collate_batch(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]


def make_synth_val_batches(n: int, batch_size: int, seed: int) -> list:
    """A fixed synthetic validation set for the on-the-fly mode. The global random state is restored
    afterwards, so the training stream does not depend on the validation size."""
    from dataset import image_to_uint8
    state = random.getstate()
    random.seed(seed + 777)
    try:
        words = load_words()
        items = []
        for _ in range(n):
            img, text = make_sample(words)
            items.append((image_to_uint8(img), text))
    finally:
        random.setstate(state)
    return make_val_batches(items, batch_size)


def load_real(args):
    """Real lines and their (train, validation) indices; (None, [], []) if --real-data is not given."""
    if not args.real_data:
        return None, [], []
    real = LineFolderDataset(args.real_data, as_uint8=True)
    r_train, r_val = split_real(real, args.real_val_fraction)
    if args.real_val_fraction > 0 and not r_val:
        print(f"WARNING: the real data set is too small for a validation part "
              f"({len(real)} lines); all of it is used for training")
    print(f"Real data: {len(real)} lines from {args.real_data}: {len(r_train)} for training, "
          f"{len(r_val)} held out for validation (fraction {args.real_val_fraction})")
    return real, r_train, r_val


def build_folder_loader(args, device):
    """"Ready-made set" mode: train in circles on the images of --synth-data (+ --real-data)."""
    synth = open_line_dataset(args.synth_data, as_uint8=True)
    t0 = time.perf_counter()
    synth_w = synth.widths()
    print(f"Synthetic set: {len(synth)} lines from {args.synth_data} (widths read in {time.perf_counter() - t0:.1f}s)")

    # Validation examples: a separate --val-data folder or --val-size examples held out from the set.
    # The network does NOT learn from them - they show whether it learns to read or just memorises the images.
    order = list(range(len(synth)))
    random.Random(12345).shuffle(order)
    if args.val_data:
        synth_val = open_line_dataset(args.val_data, as_uint8=True)
        train_idx = order
    else:
        n_val = min(args.val_size, len(synth) // 10)
        synth_val = Subset(synth, order[:n_val])
        train_idx = order[n_val:]
    train_idx.sort()

    parts, widths = [Subset(synth, train_idx)], [synth_w[i] for i in train_idx]
    val_sets = {"synth": make_val_batches(synth_val, args.batch_size)}
    n_synth_val = len(synth_val)

    real, r_train, r_val = load_real(args)
    if real is not None:
        real_w = real.widths()
        parts.append(Subset(real, r_train))
        widths += [real_w[i] for i in r_train]
        if r_val:
            val_sets["real"] = make_val_batches(Subset(real, r_val), args.batch_size)
    train_set = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    print(f"Training on {len(train_set)} lines; validation: {n_synth_val} synthetic"
          + (f" + {len(r_val)} real lines" if r_val else " lines"))

    sampler = BucketBatchSampler(widths, args.batch_size, seed=args.seed)
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate_batch, **loader_kwargs(args, device))
    return loader, val_sets, len(train_set)


def build_stream_loader(args, device):
    """"Endless generation" mode: synthetic lines are drawn on the fly, every line is new."""
    real, r_train, r_val = load_real(args)
    frac = 0.0
    if real is not None and r_train:
        frac = args.real_fraction
        if frac is None:
            frac = len(r_train) / (args.synth_per_epoch + len(r_train))
        print(f"Share of real lines in batches = {frac:.3f}")
    stream = SyntheticBucketStream(args.batch_size, real=real, real_fraction=frac, real_indices=r_train)

    val_sets = {"synth": make_synth_val_batches(args.val_size, args.batch_size, args.seed)}
    if r_val:
        val_sets["real"] = make_val_batches(Subset(real, r_val), args.batch_size)
    # batch_size=None: the batches are already assembled in the workers.
    return DataLoader(stream, batch_size=None, **loader_kwargs(args, device)), val_sets, None


@torch.no_grad()
def validate(model, batches, device, use_amp, ctc_loss):
    """Average loss, CER and WER (greedy decoding) on validation batches."""
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
    return total_loss / max(n, 1), cer(refs, hyps), wer(refs, hyps)


def resolve_x_stride(args):
    """Returns (x_stride, checkpoint state or None). When resuming, the architecture must be the one the
    checkpoint was trained with: x_stride is taken from the checkpoint; a different explicit --x-stride is an error."""
    state, ckpt_stride = None, None
    if args.resume:
        state, ckpt_stride = load_checkpoint(args.resume)
        check_num_classes(state, NUM_CLASSES, args.resume)
    if args.x_stride is None:
        return (ckpt_stride or 2), state
    if ckpt_stride is not None and args.x_stride != ckpt_stride:
        raise SystemExit(f"--x-stride {args.x_stride} does not match the checkpoint ({ckpt_stride}). "
                         "Omit --x-stride to continue with the checkpoint's value.")
    return args.x_stride, state


def train(args):
    """The main training function. args are the settings passed by the user."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Calculating on cuda: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True          # pick the fastest convolutions for every width
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 on Ampere (for the parts outside autocast)
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("!" * 70)
        print("WARNING: CUDA is not available, training runs on the CPU - dozens of times slower.")
        print("Check: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"")
        print("If the version has no '+cu...', torch without CUDA is installed - reinstall it from pytorch.org.")
        print("!" * 70)
        torch.set_num_threads(os.cpu_count())
    use_amp = device.type == "cuda" and not args.no_amp

    # --- model (first: the checkpoint decides x_stride) ---
    x_stride, state = resolve_x_stride(args)
    model = CRNN(num_classes=NUM_CLASSES, x_stride=x_stride)
    if state is not None:
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.resume} (x_stride={x_stride})")

    # --- data ---
    if args.synth_data:
        loader, val_sets, epoch_size = build_folder_loader(args, device)
    else:
        loader, val_sets, epoch_size = build_stream_loader(args, device)
    select = "real" if "real" in val_sets else "synth"   # the checkpoint is chosen by this validation set
    print(f"Data workers: {args.num_workers}; best checkpoint is chosen by the '{select}' validation set")

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)   # faster for fp16 convolutions

    # Optimizer - the "adjuster". fused=True - the whole Adam step in one CUDA kernel.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=device.type == "cuda")
    plateau = PlateauScheduler(optimizer)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    # GradScaler: in fp16 small gradients can "vanish"; the scaler multiplies the loss before backward and divides
    # the gradients back. If gradients become inf/nan, the step is skipped automatically.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (fp16): {'on' if use_amp else 'off'}, x_stride={x_stride}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    model.train()
    batches = iter(loader)
    t_last, data_time = time.perf_counter(), 0.0
    bad_total = 0
    best_key, no_improve = None, 0   # best validation result (CER, loss) and how many checks without improvement
    vram_total = torch.cuda.get_device_properties(0).total_memory / 2**30 if device.type == "cuda" else 0
    while step < args.steps:
        t0 = time.perf_counter()
        images, targets, target_lengths, widths, texts = next(batches)
        data_time += time.perf_counter() - t0

        # uint8 -> float [0..1] on the GPU (4x fewer bytes travel over the bus).
        images = images.to(device, non_blocking=True).float().div_(255.0)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            log_probs = model(images)  # [T, B, C] float32 (see model.forward)
        T = log_probs.size(0)

        # The images were padded; for every example this is its real number of steps after the CNN.
        input_lengths = model.output_lengths(widths)
        if input_lengths.max().item() > T:
            raise RuntimeError(f"input_lengths bigger than T: max={input_lengths.max().item()}, T={T}")
        bad = (input_lengths < target_lengths).sum().item()   # computed on the CPU, the GPU is not awaited
        if bad:
            bad_total += bad
            print(f"WARNING: {bad}/{len(texts)} examples have T < text length (their loss is zeroed)")

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                       # restore the real gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        plateau.update(step, loss)
        if step % args.log_every == 0:
            now = time.perf_counter()
            dt = now - t_last
            epoch_info = f"epoch {step * args.batch_size / epoch_size:.1f}  " if epoch_size else ""
            # VRAM: how much video memory is used now / at the peak. If the peak is close to the card's size,
            # Windows may quietly put data into ordinary RAM ("shared GPU memory") and a step becomes 10-20x slower.
            mem_info = (f"  VRAM {torch.cuda.memory_reserved() / 2**30:.1f}/{vram_total:.0f} GB "
                        f"(peak {torch.cuda.max_memory_reserved() / 2**30:.1f})") if device.type == "cuda" else ""
            print(f"{step}/{args.steps}  {epoch_info}loss={loss.item():.4f}  "
                  f"{dt / args.log_every * 1000:.0f} ms/step  "
                  f"(data wait {data_time / dt * 100:.0f}%)  "
                  f"ETA {(args.steps - step) * dt / args.log_every / 60:.0f} min{mem_info}")
            t_last, data_time = now, 0.0
        if step % args.checkpoint_every == 0 or step == args.steps:
            ckpt_path = os.path.join(args.checkpoint_dir, f"crnn_step{step}.pt")
            save_checkpoint(model, ckpt_path, step)
            print(f"saved checkpoint -> {ckpt_path}")
            results = {}
            for name, val_batches in val_sets.items():
                results[name] = validate(model, val_batches, device, use_amp, ctc_loss)
                v_loss, v_cer, v_wer = results[name]
                print(f"[validation:{name}] step {step}: loss={v_loss:.4f}  CER={v_cer:.3f}  WER={v_wer:.3f}")
            v_loss, v_cer, _ = results[select]
            # The best checkpoint by validation (CER first, loss on a tie) -> crnn_best.pt
            if best_key is None or (v_cer, v_loss) < best_key:
                best_key, no_improve = (v_cer, v_loss), 0
                best_path = os.path.join(args.checkpoint_dir, "crnn_best.pt")
                save_checkpoint(model, best_path, step)
                print(f"[validation] new best -> {best_path}")
            else:
                no_improve += 1
                print(f"[validation] no improvement for {no_improve} check(s) "
                      f"(best CER={best_key[0]:.3f}, loss={best_key[1]:.4f})")
                if args.early_stop and no_improve >= args.early_stop:
                    print(f"Early stop: validation did not improve {no_improve} times in a row. "
                          f"Best weights: {os.path.join(args.checkpoint_dir, 'crnn_best.pt')}")
                    break
            t_last = time.perf_counter()     # the validation time is not counted in ms/step

    print("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)         # learning steps
    parser.add_argument("--batch-size", type=int, default=64)      # examples in one step
    parser.add_argument("--lr", type=float, default=1e-3)          # the speed of learning (1e-3 = 0.001)
    parser.add_argument("--synth-data", type=str, default=None,
                        help="folder with a ready-made synthetic set (labels.csv + images/), see "
                             "make_synth_dataset.py. If omitted, synthetic lines are generated on the fly")
    parser.add_argument("--val-data", type=str, default=None,
                        help="a separate synthetic validation folder (optional)")
    parser.add_argument("--val-size", type=int, default=500,
                        help="number of synthetic validation examples (held out from --synth-data when --val-data "
                             "is not given; generated with a fixed seed in the on-the-fly mode)")
    parser.add_argument("--synth-per-epoch", type=int, default=5000,
                        help="on-the-fly mode only: used to compute the share of real data")
    parser.add_argument("--real-data", type=str, default=None,
                        help="folder with real lines (labels.csv + images/), e.g. data/real_dataset")
    parser.add_argument("--real-val-fraction", type=float, default=DEFAULT_REAL_VAL_FRACTION,
                        help="share of real lines held out for validation and never trained on (0 = use all of "
                             "them for training, then the real CER is not measured). Same default in evaluate.py")
    parser.add_argument("--real-fraction", type=float, default=None,
                        help="on-the-fly mode only: share of real lines in the batches (0..1)")
    parser.add_argument("--num-workers", type=int, default=default_workers())  # parallel data helpers
    parser.add_argument("--x-stride", type=int, default=None, choices=[2, 4],
                        help="width squeeze in the CNN: 4 is ~2x faster (see model.py). Default: 2 for a new model, "
                             "the checkpoint's value with --resume")
    parser.add_argument("--no-amp", action="store_true", help="turn mixed precision (fp16) off")
    parser.add_argument("--log-every", type=int, default=500)       # how often to print the loss
    parser.add_argument("--checkpoint-every", type=int, default=1000)  # how often to save the weights
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")  # where to save
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to continue training from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-stop", type=int, default=0,
                        help="stop if validation did not improve N checks in a row (0 = off)")
    args = parser.parse_args()

    # Fix the "randomness" so that a run can be repeated with the same result.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)

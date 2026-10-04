"""
Скрипт обучения нейросети.

Как запускать:
    python train.py --steps 2000                     # только на искусственных данных
    python train.py --real-data path/to/real_dataset  # добавить настоящие сканы, когда они появятся

Что такое «обучение»? Мы много раз повторяем один и тот же цикл:
  1) показываем сети картинку с текстом;
  2) сеть пытается прочитать и ошибается;
  3) считаем, насколько она ошиблась (это число называется loss, «потеря»);
  4) чуть-чуть подправляем внутренние числа сети (её «веса»), чтобы в
     следующий раз ошибка была меньше.
После тысяч таких повторов сеть начинает читать правильно.
"""

import argparse   # чтение настроек из командной строки (--steps 2000 и т.п.)
import os         # работа с папками
import random     # случайные числа

import torch
from torch.utils.data import ConcatDataset, DataLoader  # склейка складов данных и загрузчик пачек

from alphabet import NUM_CLASSES, encode
from dataset import RealHTRDataset, SyntheticHTRDataset, collate_batch
from model import CRNN


def build_dataset(real_data_path: str | None, synth_length: int):
    """Собирает «склад» данных для обучения: искусственные примеры + (если есть) настоящие."""
    # Искусcственные данные есть всегда.
    datasets = [SyntheticHTRDataset(length=synth_length)]
    if real_data_path:
        # Если указали папку с настоящими сканами - добавляем и их.
        datasets.append(RealHTRDataset(real_data_path))
    # Если склад один - возвращаем его, если два - склеиваем в один (ConcatDataset).
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def train(args):
    """Главная функция обучения. args - настройки, которые передал пользователь."""
    # Выбираем, на чём считать: видеокарта (cuda) в десятки раз быстрее процессора (cpu).
    # Если видеокарта есть - берём её, иначе - процессор.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Считаем на устройстве: {device}")

    dataset = build_dataset(args.real_data, args.synth_per_epoch)
    # DataLoader - «раздатчик»: берёт примеры из склада и выдаёт их пачками.
    # batch_size - сколько примеров в одной пачке; shuffle=True - перемешивать
    # порядок (чтобы сеть не заучила последовательность); collate_fn - как
    # склеивать примеры в пачку (выравнивать по ширине); num_workers - сколько
    # параллельных помощников готовят данные.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=args.num_workers,
    )

    # Создаём саму сеть и переносим её на выбранное устройство (.to(device)).
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    # Оптимизатор - «исправлятель»: знает, как именно менять веса сети, чтобы
    # ошибка уменьшалась. Adam - популярный и надёжный вариант.
    # lr (learning rate) - «скорость обучения»: насколько большими шагами менять веса.
    # Слишком большие шаги - сеть «скачет» и не сходится, слишком маленькие - учится вечно.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # CTCLoss - функция, которая считает ошибку для текста переменной длины
    # (blank=0 - номер «пустого» символа). zero_infinity=True не даёт обучению
    # сломаться, если для какого-то примера ошибка вдруг получилась бесконечной.
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # Папка для сохранений. exist_ok=True: если папка уже есть - всё в порядке.
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0          # счётчик шагов обучения (один шаг = одна пачка примеров)
    model.train()     # включаем «режим обучения» (некоторые слои ведут себя в нём иначе, чем при проверке)
    while step < args.steps:
        # Один проход по складу данных. Каждая итерация цикла - одна пачка:
        # картинки, их тексты и ширины картинок.
        for images, texts, widths in loader:
            images = images.to(device)  # переносим картинки туда же, где сеть

            # Готовим «правильные ответы» для CTC. Каждый текст переводим в
            # номера символов и склеиваем всё в один длинный список, а в
            # target_lengths запоминаем, сколько символов в каждом тексте.
            targets, target_lengths = [], []
            for t in texts:
                enc = encode(t)
                targets.extend(enc)
                target_lengths.append(len(enc))
            targets = torch.tensor(targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)

            # Прямой проход: сеть смотрит на картинки и выдаёт предсказания.
            log_probs = model(images)  # [T, B, C] - T шагов, B картинок, C символов
            T = log_probs.size(0)      # максимальное число шагов в батче (по самой широкой картинке)

            # ВАЖНО: картинки были дополнены справа до общей ширины max_w.
            # widths содержит настоящую ширину каждой картинки ДО padding.
            # Поэтому для каждого примера отдельно считаем, сколько шагов
            # действительно получилось после CNN. Хвост после этого шага
            # относится только к padding и не должен участвовать в CTC.
            input_lengths = torch.tensor(
                [model.output_length(int(w)) for w in widths],
                dtype=torch.long,
            )

            # Защита от ошибки в случае рассинхронизации формулы output_length()
            # и фактического размера выхода модели.
            if input_lengths.max().item() > T:
                raise RuntimeError(
                    f"input_lengths содержит значение больше T: "
                    f"max(input_lengths)={input_lengths.max().item()}, T={T}"
                )

            # Считаем ошибку: насколько предсказание сети отличается от правильного текста.
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Три обязательных шага «обучающего ритуала»:
            optimizer.zero_grad()   # 1) обнуляем накопленные с прошлого раза поправки
            loss.backward()         # 2) считаем, как нужно изменить каждый вес (обратное распространение ошибки)
            # Защита от «взрыва»: если поправки получились огромными, ограничиваем их размер значением 5.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()        # 3) применяем поправки - сеть стала чуть умнее

            step += 1
            # Раз в log_every шагов печатаем текущую ошибку. Она должна постепенно уменьшаться.
            if step % args.log_every == 0:
                print(f"шаг {step}/{args.steps}  ошибка={loss.item():.4f}")
            # Раз в checkpoint_every шагов (и в самом конце) сохраняем «чекпоинт» -
            # файл со всеми весами сети. Это как «сохранение в игре»: можно
            # загрузить позже и не учить заново.
            if step % args.checkpoint_every == 0 or step == args.steps:
                ckpt_path = os.path.join(args.checkpoint_dir, f"crnn_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"сохранён чекпоинт -> {ckpt_path}")
            # Достигли нужного числа шагов - выходим из цикла по пачкам.
            if step >= args.steps:
                break

    print("Обучение завершено.")


# Этот блок срабатывает только при прямом запуске файла («python train.py»).
if __name__ == "__main__":
    # ArgumentParser описывает, какие настройки можно передать при запуске.
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)         # сколько всего шагов обучения
    parser.add_argument("--batch-size", type=int, default=32)      # сколько примеров в пачке
    parser.add_argument("--lr", type=float, default=1e-3)          # скорость обучения (1e-3 = 0.001)
    parser.add_argument("--synth-per-epoch", type=int, default=5000, help="сколько искусственных примеров делать за одну «эпоху» (один круг обучения)")
    parser.add_argument("--real-data", type=str, default=None, help="путь к папке с настоящими сканами (когда они появятся)")
    parser.add_argument("--num-workers", type=int, default=2)      # параллельные помощники для данных
    parser.add_argument("--log-every", type=int, default=20)       # как часто печатать ошибку
    parser.add_argument("--checkpoint-every", type=int, default=500)  # как часто сохранять веса
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")  # куда сохранять
    args = parser.parse_args()  # читаем то, что ввёл пользователь

    # Фиксируем «случайность» числом 0, чтобы запуск можно было повторить с тем же результатом.
    random.seed(0)
    torch.manual_seed(0)
    train(args)


"""
Классы-«поставщики данных» для обучения.

В PyTorch (библиотеке для нейросетей) есть понятие Dataset: это такой
«склад», из которого можно по номеру достать один пример (картинку + текст
на ней). Здесь два склада:

- SyntheticHTRDataset: бесконечно делает искусственные примеры «на лету»
  (наша временная замена настоящим данным).
- RealHTRDataset: читает настоящие сканы почерка с диска, когда они появятся.
  Ждёт папку с картинками и файлом labels.csv, где записано «имя файла,
  что на нём написано». Подготовка такой папки - это отдельный шаг,
  его делают заранее, вручную.

Оба склада отдают пару (картинка-тензор [1, высота, ширина], строка текста).
Тензор - это просто многомерная таблица чисел, с которой работает нейросеть.
Функция collate_batch в конце выравнивает картинки по ширине, потому что
строки разной длины дают картинки разной ширины, а в одну «пачку» (batch)
нужно складывать одинаковые по размеру картинки.
"""

import csv   # чтение таблиц в формате CSV
import os    # работа с путями

import torch                          # главная библиотека для нейросетей
from PIL import Image                 # открытие картинок
from torch.utils.data import Dataset  # базовый класс «склада данных»

from synthetic_data import TARGET_HEIGHT, generate_batch, load_corpus, render_text_line

import numpy as np  # массивы чисел


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Превращает картинку Pillow в тензор чисел от 0 до 1 для нейросети."""
    # Переводим в оттенки серого ("L") и в массив чисел. Делим на 255, чтобы
    # яркость 0..255 превратилась в 0.0..1.0 (нейросетям так проще учиться).
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    # Переворачиваем яркость: было «белый фон = 1, чёрные чернила = 0»,
    # стало «фон = 0, чернила ~ 1». Так нейросети легче: «есть чернила» = большое число.
    arr = 1.0 - arr
    # unsqueeze(0) добавляет ещё одно измерение спереди - «канал» (у нас он один,
    # т.к. картинка серая). Получается форма [1, высота, ширина].
    return torch.from_numpy(arr).unsqueeze(0)


class SyntheticHTRDataset(Dataset):
    """Склад искусственных примеров: каждый раз рисует новые, а не хранит старые."""

    def __init__(self, length: int = 2000, corpus: list[str] | None = None):
        # length - сколько примеров считается «одним кругом» (эпохой) обучения.
        self.length = length
        # Набор текстов, из которых будем выбирать (если не передали - загрузим свой).
        self.corpus = corpus or load_corpus()

    def __len__(self):
        # PyTorch спрашивает: «сколько у тебя примеров?» - отвечаем.
        return self.length

    def __getitem__(self, idx):
        # PyTorch просит: «дай пример номер idx». Номер нам не важен -
        # мы каждый раз делаем случайный пример.
        import random  # импортируем здесь же, где используем

        text = random.choice(self.corpus)   # случайный текст
        img = render_text_line(text)        # рисуем его с искажениями
        return image_to_tensor(img), text   # отдаём пару (тензор, текст)


class RealHTRDataset(Dataset):
    """
    Загружает настоящие сканы почерка.

    Ожидаемая структура папки:
        root/
          labels.csv        # первая строка-заголовок: filename,text
          images/
            0001.png
            0002.png
            ...
    """

    def __init__(self, root: str):
        self.root = root
        # Список пар (имя_файла, текст). Подсказка типа list[tuple[str, str]] -
        # просто напоминание для людей и редактора кода, на работу не влияет.
        self.samples: list[tuple[str, str]] = []
        labels_path = os.path.join(root, "labels.csv")
        if not os.path.exists(labels_path):
            # Нет файла с подписями - дальше работать нельзя, останавливаемся с ошибкой.
            raise FileNotFoundError(
                f"Не найден файл {labels_path} с колонками 'filename,text'. "
                "Это шаг «Rokrakstu bāzes sagatavošana» (подготовка базы рукописей): подготовь файл заранее, вручную."
            )
        with open(labels_path, encoding="utf-8") as f:
            # DictReader читает каждую строку CSV как словарь:
            # {"filename": "0001.png", "text": "Labdien!"}
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], row["text"]))

    def __len__(self):
        # Сколько всего размеченных картинок.
        return len(self.samples)

    def __getitem__(self, idx):
        # Берём имя файла и правильный текст по номеру.
        filename, text = self.samples[idx]
        # Открываем картинку из папки images и делаем её серой.
        img = Image.open(os.path.join(self.root, "images", filename)).convert("L")
        w, h = img.size
        # Приводим к высоте 32 пикселя, сохраняя пропорции (так же, как в синтетике).
        new_w = max(1, int(w * (TARGET_HEIGHT / h)))
        img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
        return image_to_tensor(img), text


def collate_batch(batch):
    """Склеивает несколько примеров в одну «пачку» (batch).

    Картинки разной ширины, а в одну таблицу можно сложить только одинаковые.
    Поэтому находим самую широкую картинку и добавляем остальным справа
    пустые поля (нули = «пустой фон»). Возвращаем: картинки, тексты и
    настоящие ширины до выравнивания.
    """
    # batch - это список пар (картинка, текст). zip(*batch) «разворачивает» его
    # в два списка: все картинки отдельно, все тексты отдельно.
    imgs, texts = zip(*batch)
    # Самая большая ширина среди картинок (shape[-1] - последнее измерение, то есть ширина).
    max_w = max(img.shape[-1] for img in imgs)
    # Создаём «пустую» пачку из нулей: [сколько картинок, 1 канал, 32 высота, max_w ширина].
    padded = torch.zeros(len(imgs), 1, TARGET_HEIGHT, max_w)
    # Здесь запоминаем настоящую ширину каждой картинки.
    widths = torch.zeros(len(imgs), dtype=torch.long)
    for i, img in enumerate(imgs):
        w = img.shape[-1]
        padded[i, :, :, :w] = img  # кладём картинку в левую часть, справа остаются нули
        widths[i] = w
    return padded, list(texts), widths

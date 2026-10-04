"""
Генератор синтетических (искусственных) данных.

Проблема: чтобы научить нейросеть читать рукописный текст, нужны тысячи
фотографий настоящего почерка с подписями. У нас их пока нет.
Решение: берём текст, «рисуем» его шрифтом на картинке, а потом слегка
портим картинку (наклоняем, поворачиваем, добавляем шум и размытие), чтобы
она была похожа на неидеальный человеческий почерк.

Это временная заглушка, а не замена настоящим данным. Как только появятся
настоящие сканы, их подключают через класс RealHTRDataset (файл dataset.py).

Чем лучше шрифт, тем реалистичнее картинки: положи в папку fonts/ любой
шрифт .ttf в стиле рукописного, который поддерживает латышские буквы
(ā č ē ģ ī ķ ļ ņ š ū ž), и генератор подхватит его сам.
"""

import glob       # поиск файлов по шаблону (например, все *.ttf в папке)
import os         # работа с путями к файлам и папкам
import random     # случайные числа

import numpy as np                                       # быстрые операции с массивами чисел
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # Pillow: рисование и обработка картинок

# Папка со шрифтами лежит рядом с этим файлом, в подпапке fonts.
# __file__ - путь к текущему файлу, dirname берёт из него папку.
FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

# Запасной шрифт на случай, если в папке fonts ничего нет (путь для Linux).
FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Высота всех картинок после обработки - ровно 32 пикселя. Нейросеть ждёт
# картинки одной высоты, а вот ширина у каждой своя (длинная строка - широкая картинка).
TARGET_HEIGHT = 32


def _available_fonts() -> list[str]:
    """Возвращает список путей к шрифтам, которые можно использовать.

    Нижнее подчёркивание в начале имени - привычка программистов: так
    помечают функции «для внутреннего пользования», их вызывают только
    внутри этого файла.
    """
    # Ищем все файлы .ttf и .otf в папке fonts и склеиваем два списка в один.
    fonts = glob.glob(os.path.join(FONT_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_DIR, "*.otf"))
    # Если нашли хоть один шрифт - возвращаем их, иначе - запасной.
    return fonts if fonts else [FALLBACK_FONT]


def _default_corpus() -> list[str]:
    """Маленький встроенный список латышских слов и фраз.

    «Корпус» - это набор текстов для обучения. Используется, если нет
    файла data/corpus.txt с большим набором своих текстов.
    """
    return [
        "Labdien", "paldies", "lūdzu", "sveiki", "atā", "jā", "nē",
        "Rīga", "Latvija", "valoda", "grāmata", "skola", "pilsēta",
        "saule", "lietus", "sniegs", "vējš", "koks", "upe", "jūra",
        "draugs", "ģimene", "māja", "ceļš", "darbs", "laiks",
        "Es mācos latviešu valodu.", "Šodien ir skaista diena.",
        "Viņš dzīvo Rīgā.", "Mums ir daudz darba.", "Kur ir tuvākā aptieka?",
        "Cik tas maksā?", "Es gribu kafiju.", "Rīt būs saulains laiks.",
        "Bērni spēlējas parkā.", "Šī grāmata ir ļoti interesanta.",
    ]


def load_corpus() -> list[str]:
    """Загружает тексты для обучения: из файла data/corpus.txt, а если его нет - встроенный список."""
    # Собираем путь к файлу: <папка этого файла>/data/corpus.txt
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
    if os.path.exists(corpus_path):
        # Открываем файл в кодировке utf-8, чтобы латышские буквы читались правильно.
        with open(corpus_path, encoding="utf-8") as f:
            # Берём каждую строку, убираем пробелы по краям и выбрасываем пустые.
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            return lines
    # Файла нет или он пустой - используем маленький встроенный список.
    return _default_corpus()


def render_text_line(text: str, font_path: str | None = None, augment: bool = True) -> Image.Image:
    """Рисует текст `text` одной строкой на серой (чёрно-белой) картинке фиксированной высоты.

    augment=True значит «испортить» картинку случайными искажениями.
    """
    # Если шрифт не указали - выбираем случайный из доступных (для разнообразия).
    font_path = font_path or random.choice(_available_fonts())
    # Размер шрифта тоже случайный: от 26 до 40, чтобы буквы получались разного размера.
    font_size = random.randint(26, 40)
    font = ImageFont.truetype(font_path, font_size)

    # Нам нужно знать, какого размера получится текст, чтобы сделать подходящий холст.
    # Для этого создаём «пробную» крошечную картинку и измеряем на ней текст.
    # Режим "L" - оттенки серого (один канал вместо трёх RGB), 255 - белый цвет.
    dummy = Image.new("L", (10, 10), color=255)
    # textbbox возвращает рамку вокруг текста: (левый, верхний, правый, нижний край).
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    # Ширина и высота холста = размер рамки + по 20 пикселей запаса (отступы).
    # max(1, ...) защищает от нулевого размера, который сломал бы картинку.
    w = max(1, bbox[2] - bbox[0]) + 20
    h = max(1, bbox[3] - bbox[1]) + 20

    # Создаём настоящий белый холст и берём «карандаш» (draw), чтобы на нём писать.
    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)
    # Пишем текст чёрным (fill=0). Координаты сдвигаем на отступ 10 и на
    # поправку bbox, чтобы текст оказался строго по центру холста.
    draw.text((10 - bbox[0], 10 - bbox[1]), text, font=font, fill=0)

    if augment:
        # Портим картинку случайными искажениями (функция ниже).
        img = _augment(img)

    # Приводим к одной высоте 32 пикселя, сохраняя пропорции.
    img = _resize_keep_ratio(img, TARGET_HEIGHT)
    return img


def _augment(img: Image.Image) -> Image.Image:
    """Случайно искажает картинку, чтобы она напоминала живой почерк."""
    # 1) Лёгкий случайный поворот на угол от -3 до +3 градусов.
    #    expand=True - увеличить холст, чтобы углы не обрезались,
    #    fillcolor=255 - новые пустые уголки закрасить белым.
    angle = random.uniform(-3, 3)
    img = img.rotate(angle, expand=True, fillcolor=255)

    # 2) Наклон (shear) - как будто человек пишет курсивом с наклоном вправо или влево.
    #    Делается через «аффинное преобразование» - формулу, которая сдвигает
    #    каждую строку пикселей на своё расстояние (чем выше, тем сильнее сдвиг).
    shear = random.uniform(-0.25, 0.25)
    w, h = img.size
    img = img.transform(
        (w + int(abs(shear) * h), h),  # новый размер: чуть шире, чтобы наклонённый текст влез
        Image.AFFINE,                  # тип преобразования - аффинное
        (1, shear, -shear * h if shear < 0 else 0, 0, 1, 0),  # шесть чисел-коэффициентов формулы
        fillcolor=255,                 # пустые места - белые
    )

    # 3) В половине случаев (random.random() < 0.5) слегка размываем картинку,
    #    как будто фото чуть не в фокусе. radius - сила размытия.
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    # 4) Ещё в половине случаев добавляем «шум» - случайные зёрнышки, как на плохой фотографии.
    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32)  # картинка -> таблица чисел (яркость каждого пикселя)
        # Для каждого пикселя берём случайное число (нормальное распределение
        # вокруг 0) и прибавляем к яркости.
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
        # Яркость должна оставаться от 0 до 255, поэтому «обрезаем» всё лишнее (clip),
        # а потом возвращаем целые числа (uint8) и собираем картинку обратно.
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def _resize_keep_ratio(img: Image.Image, target_height: int) -> Image.Image:
    """Меняет высоту картинки до target_height, а ширину подгоняет так, чтобы не было растяжения."""
    w, h = img.size
    # Во сколько раз меняется высота, во столько же меняем и ширину.
    # Это и значит «сохранить пропорции». max(1, ...) - чтобы ширина не стала нулём.
    new_w = max(1, int(w * (target_height / h)))
    # BILINEAR - способ плавного пересчёта пикселей при изменении размера.
    return img.resize((new_w, target_height), Image.BILINEAR)


def generate_batch(n: int, corpus: list[str] | None = None) -> list[tuple[Image.Image, str]]:
    """Делает n обучающих примеров: пары «картинка + правильный текст»."""
    # Если набор текстов не передали - загружаем свой.
    corpus = corpus or load_corpus()
    samples = []
    for _ in range(n):  # «_» - имя для переменной, значение которой нам не нужно
        text = random.choice(corpus)                    # берём случайный текст
        samples.append((render_text_line(text), text))  # рисуем его и запоминаем пару
    return samples


# Этот блок срабатывает ТОЛЬКО если файл запустили напрямую («python synthetic_data.py»),
# а не импортировали из другого файла. Удобно для быстрой проверки.
if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)  # создаём папку samples (если уже есть - не ругаемся)
    # Генерируем 500 картинок и сохраняем каждую в файл.
    for i, (img, text) in enumerate(generate_batch(500)):
        img.save(f"samples/sample_{i}.png")
        print(f"sample_{i}.png -> {text!r}  size={img.size}")

"""
Распознавание (инференс): берём готовую обученную сеть и читаем текст с картинки.

Как запускать:
    python infer.py --checkpoint checkpoints/crnn_step2000.pt --image samples/sample_0.png --beam-width 5

Это финал всей цепочки: «картинка -> сеть выдаёт числа -> числа превращаются
в буквы (CTC-декодирование) -> печатаем готовый текст».
Слово «инференс» просто значит «использование уже обученной модели».
"""

import argparse

import torch
from PIL import Image

from alphabet import NUM_CLASSES, decode_beam_search
from dataset import image_to_tensor
from model import CRNN
from synthetic_data import TARGET_HEIGHT


def load_image(path: str) -> torch.Tensor:
    """Открывает файл с картинкой и готовит его для сети (так же, как при обучении)."""
    # Открываем и переводим в оттенки серого.
    img = Image.open(path).convert("L")
    w, h = img.size
    # Приводим высоту к 32 пикселям, ширину меняем пропорционально.
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
    # Превращаем в тензор [1, 32, W] и добавляем ещё одно измерение спереди -
    # «пачку из одной картинки». Сеть всегда ждёт пачку, даже если картинка одна.
    return image_to_tensor(img).unsqueeze(0)  # [1,1,H,W]


def predict(model: CRNN, image_tensor: torch.Tensor, device: torch.device, beam_width: int = 5) -> str:
    """Прогоняет картинку через сеть и возвращает распознанный текст."""
    model.eval()  # режим «проверки» (не обучения): слои ведут себя стабильно
    # no_grad говорит: «ничего не запоминай для обучения». Экономит память и время,
    # ведь мы сейчас только читаем, а не учимся.
    with torch.no_grad():
        log_probs = model(image_tensor.to(device))  # [T,1,C] - вероятности символов на каждом шаге
        # Для одной картинки убираем измерение batch: [T, 1, C] -> [T, C].
        # Вместо argmax/greedy держим несколько лучших CTC-префиксов.
        sequence_log_probs = log_probs.squeeze(1)  # [T, C]
    return decode_beam_search(sequence_log_probs, beam_width=beam_width)


# Запускается только при прямом вызове файла («python infer.py ...»).
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required=True - без этих двух параметров программа не запустится.
    parser.add_argument("--checkpoint", type=str, required=True)  # путь к файлу с весами сети
    parser.add_argument("--image", type=str, required=True)       # путь к картинке для чтения
    parser.add_argument("--beam-width", type=int, default=5,
                        help="размер beam для CTC beam search (по умолчанию 5)")
    args = parser.parse_args()

    # Выбираем видеокарту, если она есть, иначе процессор.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Создаём пустую сеть нужного устройства...
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    # ...и загружаем в неё сохранённые веса. map_location говорит, куда их положить
    # (это нужно, если учили на видеокарте, а читаем на процессоре).
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    image_tensor = load_image(args.image)
    text = predict(model, image_tensor, device, beam_width=args.beam_width)
    # !r печатает строку «как в коде» - в кавычках, так видно пробелы по краям.
    print(f"Распознанный текст: {text!r}")

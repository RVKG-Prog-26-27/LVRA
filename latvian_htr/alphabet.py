"""
Алфавит (словарь символов) для распознавания латышского рукописного текста.

Зачем он нужен? Нейросеть не умеет работать с буквами напрямую, она понимает
только числа. Поэтому каждому символу мы даём свой номер: «a» -> 1, «ā» -> 2
и так далее. Этот файл как раз хранит такую таблицу «символ <-> номер» и
умеет переводить туда и обратно.

Номер 0 зарезервирован под специальный символ «blank» (по-русски «пустой»,
«пропуск»). Он нужен для метода CTC (объясню ниже в encode / decode_greedy).
Не удаляй его, иначе всё сломается.
"""
import math
import torch

# Все буквы латышского алфавита (строчные). Обрати внимание на буквы с
# «чёрточками» и «запятыми»: ā, č, ē, ģ, ī, ķ, ļ, ņ, š, ū, ž - их в английском нет.
LATVIAN_LETTERS = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž"

# В латышском алфавите нет букв q, w, x, y, но они встречаются в заимствованных
# словах и иностранных именах. Добавляем их про запас, чтобы модель не терялась.
BASE_LATIN = "qwxy"

# Цифры от 0 до 9.
DIGITS = "0123456789"

# Знаки препинания и пробел (пробел тоже символ! Он стоит в самом начале строки).
# Символ \" внутри строки записан с обратным слэшем, чтобы Python не решил,
# что строка закончилась.
PUNCT = (
    " .,!?-:;()\"'/"                        # базовая пунктуация
    + "\u201e\u201c\u201d\u2018\u2019"       # „ " " ' '  (латышские/изогнутые кавычки)
    + "\u2013\u2014"                        # – — (короткое и длинное тире)
    + "[]{}"                                 # скобки
    + "«»"                                   # ёлочки
    + "*_+=<>@#%&^~|\\"                      # символы
    + "\u2026"                               # … многоточие
    + "\u2116"                               # № (номерной знак)
    + "€$"                                   # валюта
)


# Собираем всё вместе в один список:
#  - LATVIAN_LETTERS.upper() даёт заглавные версии букв (A, Ā, B, ...);
#  - set(...) убирает повторы (если какая-то буква попала дважды);
#  - sorted(...) выстраивает символы по порядку, чтобы номера всегда
#    получались одинаковыми при каждом запуске программы.
CHARS = sorted(set(LATVIAN_LETTERS + LATVIAN_LETTERS.upper() + BASE_LATIN + BASE_LATIN.upper() + DIGITS + PUNCT))

# Название специального «пустого» символа. Это просто метка, настоящей буквы нет.
BLANK = "<blank>"

# Итоговый алфавит: сначала blank (он получит номер 0), потом все остальные символы.
ALPHABET = [BLANK] + CHARS

# Два словаря-переводчика:
#  CHAR_TO_IDX: символ -> номер, например {"<blank>": 0, " ": 1, ...}
#  IDX_TO_CHAR: номер -> символ (обратная таблица).
# enumerate(ALPHABET) выдаёт пары (номер, символ): (0, "<blank>"), (1, " "), ...
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

# Сколько всего «классов» (разных вариантов ответа) у нейросети: каждый символ + blank.
# Нейросеть на каждом шаге будет выбирать один из NUM_CLASSES вариантов.
NUM_CLASSES = len(ALPHABET)


def encode(text: str) -> list[int]:
    """Превращает текст в список номеров символов.

    Пример: «ja» -> [номер "j", номер "a"].
    Blank-символы мы сюда НЕ вставляем: функция потерь CTCLoss сама этого
    не ждёт, ей нужен чистый текст в виде номеров.
    """
    # Ищем «неизвестные» символы - те, которых нет в нашем алфавите
    # (например, русская буква или эмодзи).
    unknown = set(ch for ch in text if ch not in CHAR_TO_IDX)
    if unknown:
        # Если нашли, останавливаем программу с понятной ошибкой.
        raise ValueError(f"В тексте есть символы, которых нет в алфавите: {unknown!r}. Добавь их в CHARS в файле alphabet.py.")
    # Для каждого символа берём его номер из таблицы.
    return [CHAR_TO_IDX[ch] for ch in text]


def decode_greedy(indices: list[int]) -> str:
    """Превращает ответ нейросети (список номеров) обратно в текст.

    Это «жадное» декодирование CTC. Нейросеть выдаёт ответ для каждого
    маленького кусочка картинки по очереди, поэтому одна и та же буква
    может повториться много раз подряд. Например, для слова «sveiki»
    сеть может выдать: s s <blank> v e e i <blank> k k i.
    Правила, чтобы получить текст:
      1) подряд идущие одинаковые номера схлопываем в один (e e -> e);
      2) blank выбрасываем (он значит «тут ничего нет»).
    Зачем тогда blank? Чтобы можно было написать двойную букву: если между
    двумя «l» стоит blank, то это настоящее «ll», а не одна «l».
    """
    out = []      # сюда складываем найденные буквы
    prev = None   # какой номер был на прошлом шаге (в начале - никакого)
    for idx in indices:
        if idx != prev:       # номер изменился -> это новый символ (правило 1)
            if idx != 0:      # 0 - это blank, его пропускаем (правило 2)
                out.append(IDX_TO_CHAR[idx])
        prev = idx            # запоминаем текущий номер для следующего шага
    # Склеиваем список букв в одну строку.
    return "".join(out)


def decode_beam_search(log_probs, beam_width: int = 5) -> str:
    """CTC prefix beam search по матрице log-probabilities [T, C].

    В отличие от greedy decoding, здесь одновременно сохраняются несколько
    лучших вариантов текста. Для каждого префикса отдельно храним вероятность
    состояния, заканчивающегося на CTC blank, и вероятность состояния,
    заканчивающегося на обычный символ.

    Args:
        log_probs: torch.Tensor формы [T, C], обычно результат model(...)
            после log_softmax.
        beam_width: сколько лучших префиксов сохранять на каждом шаге.

    Returns:
        Самый вероятный CTC-префикс как строка.
    """
    if beam_width < 1:
        raise ValueError("beam_width должен быть >= 1")


    if log_probs.ndim != 2:
        raise ValueError(
            f"decode_beam_search ожидает [T, C], получено {tuple(log_probs.shape)}"
        )
    if log_probs.size(1) > NUM_CLASSES:
        raise ValueError(
            f"В модели {log_probs.size(1)} классов, но алфавит содержит только {NUM_CLASSES}"
        )

    def log_add(a: float, b: float) -> float:
        """log(exp(a) + exp(b)) без потери устойчивости на -inf."""
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        if a < b:
            a, b = b, a
        return a + math.log1p(math.exp(b - a))

    # prefix -> (p_blank, p_nonblank), всё в log-space.
    # Пустой префикс начинается с вероятности 1 через blank => log(1) = 0.
    neg_inf = -math.inf
    beams = {(): (0.0, neg_inf)}

    probs = log_probs.detach().float().cpu()

    for t in range(probs.size(0)):
        step = probs[t].tolist()
        next_beams = {}

        def add(prefix, p_blank=neg_inf, p_nonblank=neg_inf):
            old_blank, old_nonblank = next_beams.get(prefix, (neg_inf, neg_inf))
            next_beams[prefix] = (
                log_add(old_blank, p_blank),
                log_add(old_nonblank, p_nonblank),
            )

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = log_add(p_blank, p_nonblank)

            # CTC blank: текстовый префикс не меняется.
            add(prefix, p_blank=total + step[0])

            last = prefix[-1] if prefix else None

            for c in range(1, log_probs.size(1)):
                p = step[c]

                if c == last:
                    # Тот же символ без blank остаётся тем же префиксом:
                    # ... c c -> ... c.
                    add(prefix, p_nonblank=p_nonblank + p)

                    # А если перед вторым c был blank, это уже новая буква:
                    # ... c blank c -> ... cc.
                    add(prefix + (c,), p_blank=p_blank + p)
                else:
                    # Новый символ может прийти как из blank-, так и из
                    # nonblank-состояния.
                    add(prefix + (c,), p_nonblank=total + p)

        # Сортируем по полной вероятности префикса и оставляем beam_width.
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: log_add(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix = max(
        beams,
        key=lambda prefix: log_add(beams[prefix][0], beams[prefix][1]),
    )
    return "".join(IDX_TO_CHAR[i] for i in best_prefix)
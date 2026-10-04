"""
Alfabēts (simbolu vārdnīca).

Mēs katram simbolam piešķiram unikālu numuru: "a" -> 1, "ā" -> 2
un tā tālāk. Šajā failā tiek glabāta tieši šī "simbols <-> numurs" tabula,
un tas nodrošina konvertēšanu abos virzienos.

0 = "blank" (blank)
Tas ir nepieciešams CTC metodei.
"""
import math
import torch

LATVIAN_LETTERS = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž"
BASE_LATIN = "qwxy"
DIGITS = "0123456789"

# Pieturzīmes un atstarpe (atstarpe arī ir simbols! Tas atrodas pašā rindas sākumā).
# Rakstzīme \" virknē tiek rakstīta ar atpakaļvērstās slīpsvītru, lai Python nevarētu pieņemt lēmumu
# ka līnija ir beigusies.
PUNCT = (
    " .,!?-:;()\"'/"                        # pamata pieturzīmes
    + "\u201e\u201c\u201d\u2018\u2019"       # „ " " ' ' (latviešu/izliektas pēdiņas)
    + "\u2013\u2014"                         # – – (defise en un em)
    + "[]{}"                                 # iekavas 
    + "«»"                                   # eglītes
    + "*_+=<>@#%&^~|\\"                      # simboli
    + "\u2026"                               # … 
    + "\u2116"                               # № 
    + "€$"                                   # valūta 
)

# Salieciet visu vienā sarakstā:
# - LATVIEŠU_BURTI.upper() dod lielo burtu versijas (A, Ā, B, ...);
# - set(...) noņem atkārtojumus (ja kāds burts parādās divreiz);
# - sakārtots(...) sakārto rakstzīmes tā, lai cipari vienmēr būtu
# izrādījās vienādi katru reizi, kad programma tika palaista.
CHARS = sorted(set(LATVIAN_LETTERS + LATVIAN_LETTERS.upper() + BASE_LATIN + BASE_LATIN.upper() + DIGITS + PUNCT))

BLANK = "<blank>"

ALPHABET = [BLANK] + CHARS

# Divas vārdnīcas tulkošanai:

# enumerate(ALPHABET) izvada pārus (0, "<blank>")
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

# Kopējais «klašu» skaits (dažādie atbilžu varianti) neironu tīklam: katrs simbols + blank.
# Neironu tīkls katrā solī izvēlēsies vienu no NUM_CLASSES variantiem.
NUM_CLASSES = len(ALPHABET)


def encode(text: str) -> list[int]:
    """Pārvērš tekstu rakstzīmju kodu sarakstā. Izslēdz blank. """
    # Meklējam «nezināmos» simbolus
    unknown = set(ch for ch in text if ch not in CHAR_TO_IDX)
    if unknown:
        raise ValueError(f"В тексте есть символы, которых нет в алфавите: {unknown!r}.")
    # Katram simbolam paņemam tā numuru no tabulas.
    return [CHAR_TO_IDX[ch] for ch in text]


def decode_greedy(indices: list[int]) -> str:
    """Pārvērš neironu tīkla izvadi atpakaļ tekstā.
    Piemēram, vārdam "sveiki"
    tīkls varētu izvadīt: s s <blank> v e e i <blank> k k i.
    Noteikumi teksta iegūšanai:
      1) apvienot secīgas vienādas rakstzīmes vienā (e e -> e);
      2) atmest blank (tas nozīmē "šeit nekā nav").
    blank = dubultburts
    """
    out = []      # burti
    prev = None   # kāds numurs bija iepriekšējā solī (sākumā - neviena)
    for idx in indices:
        if idx != prev:       # numurs mainījās -> tas ir jauns simbols (1. noteikums)
            if idx != 0:      
                out.append(IDX_TO_CHAR[idx])
        prev = idx            # saglabājam pašreizējo numuru nākamajam solim
    # Savienojam burtu sarakstu vienā virknē.
    return "".join(out)


def decode_beam_search(log_probs, beam_width: int = 5) -> str:
    """Izmantojot CTC, vienlaikus tiek saglabāti vairāki labākie teksta varianti.
    Katram prefiksam atsevišķi tiek glabāta varbūtība, ka stāvoklis
    beidzas ar CTC blank simbolu, un varbūtība, ka stāvoklis
    beidzas ar parastu simbolu.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    if log_probs.ndim != 2:
        raise ValueError(
            f"decode_beam_search waiting [T, C], received {tuple(log_probs.shape)}"
        )
    if log_probs.size(1) > NUM_CLASSES:
        raise ValueError(
            f"model {log_probs.size(1)}  {NUM_CLASSES}"
        )

    def log_add(a: float, b: float) -> float:
        """log(exp(a) + exp(b))"""
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        if a < b:
            a, b = b, a
        return a + math.log1p(math.exp(b - a))

    # Tukšais prefikss sākas ar varbūtību 1 caur blank => log(1) = 0.
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

            # CTC blank: teksta prefikss nemainās.
            add(prefix, p_blank=total + step[0])

            last = prefix[-1] if prefix else None

            for c in range(1, log_probs.size(1)):
                p = step[c]

                if c == last:
                    # Tas pats simbols bez blank paliek tas pats prefikss:
                    add(prefix, p_nonblank=p_nonblank + p)

                    # Ja pirms otrā c bija blank, tā jau ir jauna burta parādīšanās:
                    # ... c blank c -> ... cc.
                    add(prefix + (c,), p_blank=p_blank + p)
                else:
                    # Jaunais simbols var nākt gan no blank, gan no
                    # nonblank stāvokļa.
                    add(prefix + (c,), p_nonblank=total + p)

        # Kārtojam pēc prefiksa pilnās varbūtības un atstājam beam_width labākos.
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

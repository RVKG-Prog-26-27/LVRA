"""
Atpazīšana (secināšana): mēs ņemam jau apmācītu tīklu un nolasām tekstu no attēla.

Šis ir pēdējais solis visā ķēdē: "attēls -> tīkls izvada skaitļus -> skaitļi tiek pārveidoti burtos (CTC dekodēšana) -> izdrukā gatavo tekstu."
"""

import argparse

import torch
from PIL import Image

from alphabet import NUM_CLASSES, decode_beam_search
from dataset import image_to_tensor
from model import CRNN
from synthetic_data import TARGET_HEIGHT


def load_image(path: str) -> torch.Tensor:
    """Atver attēla failu un sagatavo to tīmekļa pārlūkošanai (tāpat kā apmācības laikā)."""
    # Atveram attēlu un pārveidojam to pelēktoņu attēlā.
    img = Image.open(path).convert("L")
    w, h = img.size
    # Samazinām vai palielinām augstumu līdz 32 pikseļiem, platumu mainām proporcionāli.
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
    # Pārveidojam par tenzoru [1, 32, W] un priekšā pievienojam vēl vienu dimensiju -
    # «pakešu ar vienu attēlu». Tīkls vienmēr sagaida pakešu, pat ja attēls ir tikai viens.
    return image_to_tensor(img).unsqueeze(0) 


def predict(model: CRNN, image_tensor: torch.Tensor, device: torch.device, beam_width: int = 5) -> str:
    """Palaiž attēlu tīklā un atgriež atpazīto tekstu.."""
    model.eval()  # «novērtēšanas» režīms (nevis apmācības): slāņi darbojas stabilā režīmā
    # no_grad nozīmē «neuzkrāj informāciju apmācībai». Tas ietaupa atmiņu un laiku,
    with torch.no_grad():
        log_probs = model(image_tensor.to(device))  # [T,1,C] - simbolu varbūtības katrā solī
        # Vienam attēlam noņemam batch dimensiju: [T, 1, C] -> [T, C].
        # saglabājam vairākus labākos CTC prefiksus.
        sequence_log_probs = log_probs.squeeze(1)  # [T, C]
    return decode_beam_search(sequence_log_probs, beam_width=beam_width)


# Izpildās tikai tad, ja fails tiek palaists tieši («python infer.py ...»).
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required=True - bez šiem diviem parametriem programma nepalaižas.
    parser.add_argument("--checkpoint", type=str, required=True)  # ceļš uz tīkla svaru failu
    parser.add_argument("--image", type=str, required=True)       # ceļš uz nolasāmo attēlu
    parser.add_argument("--beam-width", type=int, default=5,
                        help="size beam for CTC beam search (5)")
    args = parser.parse_args()

    # Izvēlamies videokarti, ja tā ir pieejama, pretējā gadījumā procesoru.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Izveidojam tukšu tīklu izvēlētajai ierīcei...
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    # ...un ielādējam tajā saglabātos svarus. map_location norāda, kur tos ievietot.
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    image_tensor = load_image(args.image)
    text = predict(model, image_tensor, device, beam_width=args.beam_width)
    # !r izdrukā virkni «kā kodā» - pēdiņās, tāpēc ir redzamas arī atstarpes malās.
    print(f"Teksts: {text!r}")

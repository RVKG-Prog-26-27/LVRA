"""
"Datu nodrošinātāja" klases apmācībai:
- SyntheticHTRDataset: nepārtraukti un dinamiski ģenerē sintētiskus piemērus.
- RealHTRDataset: nolasa reālus rokraksta attēlus no diska, tiklīdz tie kļūst pieejami.
Abas klases izvada pāri, ko veido attēla tenzors (izmēri: [1, augstums, platums]) un teksta virkne.
Tenzors ir skaitlisks masīvs, ko apstrādā neironu tīkls.
Funkcija `collate_batch` saskaņo attēlus pēc platuma, jo
dažāda garuma teksta rindas rada dažāda platuma attēlus, savukārt vienai pakešapstrādes grupai
ir nepieciešami vienāda izmēra attēli.
"""
import csv   
import os    

import torch                          
from PIL import Image                 
from torch.utils.data import Dataset  

from synthetic_data import TARGET_HEIGHT, generate_batch, load_corpus, render_text_line

import numpy as np  # skaitļu masīvi


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Neironu tīklam pillow attēlu pārvērš skaitļu tenzorā no 0 līdz 1."""
    # Pārveidojam pelēktoņu attēlā ("L") un skaitļu masīvā. Dalām ar 255, lai
    # spilgtums 0..255 pārvērstos par 0.0..1.0 (neironu tīklam tā ir vieglāk mācīties).
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    # Apgriežam spilgtumu: iepriekš «balts fons = 1, melna tinte = 0»,
    # tagad «fons = 0, tinte ~ 1». Tīklam tā ir vieglāk: «ir tinte» = liels skaitlis.
    arr = 1.0 - arr
    # unsqueeze(0) Rezultātā iegūstam formu [1, augstums, platums].
    return torch.from_numpy(arr).unsqueeze(0)


class SyntheticHTRDataset(Dataset):
    """Mākslīgu piemēru noliktava: katru reizi tā zīmē jaunus, nevis glabā vecos."""
    def __init__(self, length: int = 2000, corpus: list[str] | None = None):
        # length - viens mācību cikls.
        self.length = length
        # Tekstu kopa
        self.corpus = corpus or load_corpus()

    def __len__(self):
        # piemēru skaits
        return self.length

    def __getitem__(self, idx):
        # katru reizi izveidojam nejaušu piemēru.
        import random  # importējam tieši tur, kur to izmantojam

        text = random.choice(self.corpus)   # nejaušs teksts
        img = render_text_line(text)        # izveidojam attēlu ar izmaiņām
        return image_to_tensor(img), text   # atgriežam pāri (tenzors, teksts)


class RealHTRDataset(Dataset):
    """
   Augšupielādē īstus rokraksta skenējumus.
    """

    def __init__(self, root: str):
        self.root = root
        # Pāru saraksts (faila_nosaukums, teksts).
        self.samples: list[tuple[str, str]] = []
        labels_path = os.path.join(root, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"file not found {labels_path} with columns 'filename,text'. "
            )
        with open(labels_path, encoding="utf-8") as f:
            # vārdnīca:
            # {"filename": "0001.png", "text": "Labdien!"}
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], row["text"]))

    def __len__(self):
        # Kopējais marķēto attēlu skaits.
        return len(self.samples)

    def __getitem__(self, idx):
        # faila nosaukums un pareizais teksts pēc numura.
        filename, text = self.samples[idx]
        img = Image.open(os.path.join(self.root, "images", filename)).convert("L")
        w, h = img.size
        new_w = max(1, int(w * (TARGET_HEIGHT / h)))
        img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
        return image_to_tensor(img), text

def collate_batch(batch):
    """
    Apvieno vairākus piemērus vienā "batch".

Attēla platums tiek pielāgots platākajam attēlam.
    """
    # batch ir pāru (attēls, teksts) saraksts. zip(*batch) to «izjauc»
    # divos sarakstos: visus attēlus atsevišķi un visus tekstus atsevišķi.
    imgs, texts = zip(*batch)
    # Lielākais platums starp attēliem.
    max_w = max(img.shape[-1] for img in imgs)
    # Izveidojam «tukšu» pakešu masīvu no nullēm.
    padded = torch.zeros(len(imgs), 1, TARGET_HEIGHT, max_w)
    # Šeit saglabājam katra attēla patieso platumu.
    widths = torch.zeros(len(imgs), dtype=torch.long)
    for i, img in enumerate(imgs):
        w = img.shape[-1]
        padded[i, :, :, :w] = img  # ievietojam attēlu kreisajā daļā, labajā pusē paliek nulles
        widths[i] = w
    return padded, list(texts), widths

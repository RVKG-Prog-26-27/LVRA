# Latviešu rokrakstu atpazīšana — sākotnējā versija_02

Sākotnējā (baseline) implementācija pēc tavas blokshēmas. **AI bloks blokshēmā bija uzzīmēts nepareizi** — CNN un BiLSTM tur izskatās pēc diviem neatkarīgiem zariem, bet praksē tā strādāt nevar. Reālajā arhitektūrā (CRNN, standarts rokraksta/OCR atpazīšanai) tie ir **secīgi** posmi + trūkst CTC slāņa, kas savieno tīkla izvadi ar mainīga garuma tekstu:

```
attēls -> CNN (vizuālās pazīmes, "Līnijas atpazīšana" / "Pareiza novietošana")
       -> BiLSTM x2 (konteksts pa sekvenci, "Vārdu atpazīšana" / "Pieturzīmes" / "Cipari")
       -> Linear + CTC (blokshēmā vispār nebija — bez tā nevar mainīga garuma
          izvadi savienot ar tekstu bez rakstzīmju līmeņa segmentācijas)
```

CTC arī nozīmē, ka "Teksta zonas atpazīšana????" (ar jautājuma zīmēm blokshēmā —
acīmredzot arī tev bija šaubas) lielākoties nav vajadzīga uz ievades puses:
var padot veselas rindas attēlu, nevis atsevišķus burtus.

## Faili

| Fails | Blokshēmas daļa |
|---|---|
| `alphabet.py` | rakstzīmju kopa (ā č ē ģ ī ķ ļ ņ š ū ž u.c.) + CTC kodēšana/dekodēšana |
| `synthetic_data.py` | "Sintētiskie dati" — ģenerē attēlus no teksta, ar rotāciju/šķības/troksni |
| `dataset.py` | "Reāli dati" (`RealHTRDataset`) + sintētiskie (`SyntheticHTRDataset`) |
| `model.py` | laboti "AI" — CRNN (CNN -> BiLSTM -> CTC) |
| `train.py` | apmācības cikls |
| `infer.py` | "OUTPUT" — attēls -> teksts |

## Palaišana

```bash
pip install -r requirements.txt

# ātrs tests — ģenerē pāris paraugattēlus
python synthetic_data.py

# apmācība tikai uz sintētiskajiem datiem (pagaidu risinājums, kamēr nav reālu skenējumu)
python train.py --steps 2000 --batch-size 32

# kad būs reāli dati (skat. zemāk), pievieno tos apmācībai
python train.py --steps 5000 --real-data data/real_dataset

# secinājums uz viena attēla
python infer.py --checkpoint checkpoints/crnn_step2000.pt --image samples/sample_0.png
```

## Kā pievienot reālus datus

Sagatavo mapi (tas ir blokshēmas "Rokrakstu bāzes sagatavošana" solis):

```
data/real_dataset/
  labels.csv          # kolonnas: filename,text
  images/
    0001.png
    0002.png
```

`labels.csv` piemērs:
```csv
filename,text
0001.png,Labdien! Kā jums iet?
0002.png,Rīgā šodien līst.
```

## Kas šobrīd ir vienkāršots / jāuzlabo tālāk

- **Sintētiskie dati** izmanto parastu (ne-rokraksta) fontu (DejaVu Sans) ar
  nelielām deformācijām. Reālam rokrakstam ieteicams iemest `fonts/` mapē
  kursīvu/rokraksta stila `.ttf` fontu ar latviešu diakritiku atbalstu —
  ģenerators to automātiski izmantos.
- Nav vēl datu augmentācijas variāciju pēc reāliem paraugiem (piem., papīra
  tekstūras, tintes izplūduma) — der pievienot, tiklīdz ir reāli skeni, lai
  redzētu, kāda veida troksnis tos raksturo.
- "Cik piemēru?" no blokshēmas — nav fiksēts skaitlis kodā; `--steps` un
  `--synth-per-epoch` kontrolē apjomu, pielāgo pēc vajadzības.
- Šī vide nespēja lokāli instalēt/palaist pilnu PyTorch (CUDA atkarības prasa
  vairāk diska vietas nekā šeit pieejams), tāpēc modeļa forward/apmācības
  kods nav palaists end-to-end šajā sandbox — sintaktiski pārbaudīts un
  arhitektūra ir standarta, pārbaudīta CRNN+CTC shēma, bet ieteicams pirmo
  palaišanu izdarīt ar mazu `--steps` skaitli, lai pārliecinātos, ka viss
  strādā tavā vidē.

# Latviešu rokrakstu atpazīšana — alphaa versija_02

**LVRA (Latviešu Valodas Rokraksta Atpazīšana)** ir eksperimentāls latviešu rokraksta atpazīšanas projekts. Mērķis ir no rokrakstā rakstīta teksta attēla iegūt rediģējamu digitālu tekstu.

Šobrīd projektā ir izveidota funkcionējoša **CRNN + CTC** atpazīšanas sistēma, sintētisko un reālo datu atbalsts, individuāli `input_lengths` CTC apmācībai un **CTC Prefix Beam Search** dekodēšana.

> **Alpha versija:** modelis un datu ģenerēšana jau darbojas, taču atpazīšanas kvalitāte vēl nav uzskatāma par gala produkta līmeni. Lai sasniegtu labu reāla rokraksta atpazīšanu, nepieciešams lielāks un kvalitatīvāks reālo rokraksta datu kopums.

---

## Arhitektūra

Sākotnējā blokshēmā CNN un BiLSTM bija attēloti kā divi neatkarīgi zari. Rokraksta atpazīšanai tie tiek izmantoti **secīgi**, veidojot CRNN arhitektūru:

```text
Attēls
  │
  ▼
CNN
  │
  │ vizuālo pazīmju iegūšana
  ▼
BiLSTM
  │
  │ konteksta analīze abos virzienos
  ▼
BiLSTM
  │
  ▼
Linear
  │
  │ rakstzīmju varbūtības
  ▼
CTC
  │
  ▼
Teksts
```

### CNN

CNN apstrādā attēlu un iegūst vizuālās pazīmes:

* burtu formas;
* līnijas;
* rakstzīmju kontūras;
* telpisko izvietojumu.

Attēla platums tiek pārvērsts par secību, kuru pēc tam var apstrādāt ar RNN.

### BiLSTM

Divi BiLSTM slāņi analizē secību abos virzienos.

Tas ļauj modelim izmantot ne tikai informāciju no iepriekšējiem, bet arī no nākamajiem simboliem.

Piemēram, neskaidru rakstzīmi var palīdzēt atpazīt tās konteksts vārdā.

### CTC

CTC (**Connectionist Temporal Classification**) ļauj apmācīt modeli bez nepieciešamības iepriekš sadalīt attēlu atsevišķos burtos.

Modelim nav jāzina:

```text
kur tieši atrodas pirmais burts
kur tieši atrodas otrais burts
kur tieši atrodas trešais burts
```

Tam pietiek ar:

```text
attēls → "Labdien!"
```

CTC pats optimizē iespējamos izlīdzinājumus starp attēla secību un tekstu.

---

# Kas šobrīd darbojas

## 1. Latviešu rakstzīmju kopa

`alphabet.py` satur rakstzīmju kopu, ko izmanto modelis.

Tajā ir latviešu valodai nepieciešamās rakstzīmes, tostarp diakritiskās:

```text
ā č ē ģ ī ķ ļ ņ š ū ž
```

kā arī:

* lielie un mazie burti;
* cipari;
* atstarpes;
* pieturzīmes;
* CTC `blank` simbols.

Ir pieejama teksta kodēšana uz indeksiem un dekodēšana atpakaļ uz tekstu.

---

## 2. Sintētisko datu ģenerēšana

`synthentic_data.py` ģenerē apmācības attēlus automātiski.

No teksta tiek izveidots attēls, kuru pēc tam var modificēt ar dažādiem vizuāliem traucējumiem.

Tiek izmantotas, piemēram:

* neliela rotācija;
* perspektīvas/šķības deformācijas;
* trokšņi;
* nelielas attēla kvalitātes izmaiņas;
* citas vienkāršas deformācijas.

Tas ļauj sākt modeļa izstrādi arī tad, ja vēl nav liela reāla rokraksta datu kopuma.

### Svarīgs ierobežojums

Pašlaik sintētiskie dati galvenokārt izmanto parastos datorfontus, nevis īstu cilvēka rokrakstu.

Tāpēc sintētiskie dati ir piemēroti arhitektūras un apmācības pārbaudei, bet ar tiem vien nepietiks kvalitatīvai reāla rokraksta atpazīšanai.

---

## 3. Reālo datu atbalsts

`dataset.py` satur `RealHTRDataset`, kas ļauj izmantot reālus rokraksta attēlus.

Datu struktūra:

```text
data/real_dataset/
├── labels.csv
└── images/
    ├── 0001.png
    ├── 0002.png
    └── 0003.png
```

`labels.csv`:

```csv
filename,text
0001.png,Labdien! Kā jums iet?
0002.png,Rīgā šodien līst.
0003.png,Šis ir rokraksta piemērs.
```

Reālos un sintētiskos datus iespējams izmantot vienā apmācībā.

---

# 4. Mainīga attēlu platuma atbalsts

Attēli vienā batch parasti nav vienāda platuma.

Piemēram:

```text
160 px
240 px
320 px
```

Lai tos ievietotu vienā batch, īsākie attēli tiek papildināti ar `padding` līdz maksimālajam platumam:

```text
160 → 320
240 → 320
320 → 320
```

Tomēr padding nav īsts attēla saturs.

Tāpēc modelis saglabā katra attēla sākotnējo platumu un aprēķina tam individuālu:

```text
input_length
```

Piemēram:

```text
attēla platums: 160 → input_length = 39
attēla platums: 240 → input_length = 59
attēla platums: 320 → input_length = 79
```

CTC tādējādi redz tikai reālo attēla daļu un neizmanto padding kā teksta informāciju.

Tas ir svarīgs uzlabojums salīdzinājumā ar sākotnējo variantu, kur visiem batch elementiem tika izmantots vienāds `T`.

---

# 5. CTC Beam Search dekodēšana

Pēc modeļa apmācības teksts jāiegūst no modeļa izvades.

Vienkāršākā metode ir **greedy decoding**:

```text
katram laika solim
        ↓
izvēlies visvarbūtīgāko rakstzīmi
        ↓
izveido tekstu
```

Taču visvarbūtīgākā rakstzīme katrā atsevišķā solī ne vienmēr veido visvarbūtīgāko tekstu kopumā.

Tāpēc alpha versijā ir ieviests **CTC Prefix Beam Search**.

Beam Search vienlaikus saglabā vairākus labākos iespējamos teksta variantus.

Piemēram:

```text
beam_width = 3

        ┌── "hat"
        │
        ├── "had"
attēls ─┤
        └── "her"
```

Pēc katra nākamā soļa mazāk ticamie varianti tiek atmesti.

### `beam-width`

Izmēru var mainīt ar:

```bash
--beam-width
```

Piemēram:

```bash
python infer.py \
  --checkpoint checkpoints/crnn_step2000.pt \
  --image samples/sample_0.png \
  --beam-width 5
```

Vai:

```bash
python infer.py \
  --checkpoint checkpoints/crnn_step2000.pt \
  --image samples/sample_0.png \
  --beam-width 10
```

Ieteicamais sākuma diapazons:

```text
5–20
```

Lielāks `beam_width` nozīmē vairāk pārbaudītu variantu un nedaudz lielāku aprēķinu laiku.

**Beam Search neprasa modeļa pārtrenēšanu**, jo tas maina tikai jau iegūto modeļa izvades dekodēšanas veidu.

---

# 6. Attēla sagatavošana

Pirms attēls tiek padots modelim, `infer.py`:

1. atver attēlu;
2. pārveido to pelēktoņu attēlā;
3. saglabā proporcijas;
4. pielāgo attēla augstumu līdz 32 pikseļiem;
5. pārveido attēlu tensorā;
6. padod to CRNN modelim.

Tādējādi inference izmanto tādu pašu attēla formātu kā apmācības laikā.

---

# 7. Apmācība

`train.py` realizē pilnu apmācības ciklu:

```text
dati
 ↓
DataLoader
 ↓
CRNN
 ↓
CTC Loss
 ↓
backpropagation
 ↓
Adam
 ↓
jauni modeļa svari
```

Tiek izmantots:

* `Adam` optimizētājs;
* `CTCLoss`;
* gradientu clipping;
* CUDA, ja pieejama NVIDIA GPU;
* CPU kā rezerves variants;
* batch apstrāde;
* checkpoint saglabāšana.

Piemēram:

```bash
python train.py --steps 2000 --batch-size 32
```

---

# 8. Checkpoint

Pēc noteikta soļu skaita modelis tiek saglabāts:

```text
checkpoints/
├── crnn_step500.pt
├── crnn_step1000.pt
├── crnn_step1500.pt
└── crnn_step2000.pt
```

Checkpoint satur apmācītā modeļa svarus.

Tas ļauj izmantot jau apmācītu modeli inference laikā, neapmācot to vēlreiz no nulles.

---

# 9. Inference

Lai pārbaudītu apmācītu modeli uz vienas bildes:

```bash
python infer.py \
  --checkpoint checkpoints/crnn_step2000.pt \
  --image samples/sample_0.png
```

Modelis:

```text
attēls
  ↓
CNN
  ↓
BiLSTM
  ↓
Linear
  ↓
CTC log-probabilities
  ↓
Beam Search
  ↓
atpazītais teksts
```

---

# 10. Web lietotne

Projektā ir arī `web-app/` mape ar React/Vite lietotni.

Tā jau nodrošina lietotāja saskarni latviešu valodā:

* attēla izvēli;
* fotografēšanu ar telefonu;
* attēla priekšskatījumu;
* attēla pagriešanu;
* atpazīšanas procesa ekrānu;
* atcelšanu;
* rezultāta attēlošanu;
* atpazītā teksta rediģēšanu;
* teksta kopēšanu;
* kļūdu paziņojumus;
* izstrādātāja testēšanas paneli.

Tomēr **web lietotne pašlaik vēl nav savienota ar reālo CRNN modeli**.

Tā izmanto pagaidu `mock` atpazīšanas servisu, kas neatkarīgi no izvēlētā attēla atgriež iepriekš sagatavotu demonstrācijas tekstu.

Tāpēc pašreizējā alpha versijā:

```text
Python CRNN modelis       ✅ darbojas atsevišķi
Python inference          ✅ darbojas
Beam Search               ✅ darbojas
Web lietotne              ✅ darbojas
Web lietotne → CRNN       ❌ vēl nav savienots
```

---

# Palaišana

## Python daļa

Instalē nepieciešamās bibliotēkas:

```bash
pip install -r latvian_htr/requirements.txt
```

Izveido sintētiskos paraugus:

```bash
cd latvian_htr
python synthetic_data.py
```

Apmāci modeli:

```bash
python train.py --steps 2000 --batch-size 32
```

Apmāci modeli ar reāliem datiem:

```bash
python train.py \
  --steps 5000 \
  --real-data ../data/real_dataset
```

Pārbaudi modeli:

```bash
python infer.py \
  --checkpoint checkpoints/crnn_step2000.pt \
  --image ../samples/sample_0.png
```

Izmanto lielāku Beam Search:

```bash
python infer.py \
  --checkpoint checkpoints/crnn_step2000.pt \
  --image ../samples/sample_0.png \
  --beam-width 10
```

---

# Web lietotnes palaišana

```bash
cd web-app
npm install
npm run dev
```

Pēc tam pārlūkprogrammā atver:

```text
http://localhost:5173
```

Telefonam tajā pašā tīklā:

```bash
npm run dev:phone
```

---

# Projekta struktūra

```text
LVRA/
│
├── latvian_htr/
│   ├── alphabet.py
│   ├── dataset.py
│   ├── infer.py
│   ├── model.py
│   ├── requirements.txt
│   ├── synthetic_data.py
│   └── train.py
│
├── samples/
│   └── sample_*.png
│
├── web-app/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

---

# Kas vēl jāuzlabo

Alpha versija vēl nav gala sistēma. Svarīgākie nākamie soļi:

### 1. Lielāka reālo rokraksta datu kopa

Pašlaik sintētiskie dati nevar pilnībā aizvietot īstu cilvēku rokrakstu.

Nepieciešami:

* dažādu cilvēku rokraksti;
* dažādi rakstīšanas stili;
* dažādi pildspalvas veidi;
* dažāds rakstīšanas ātrums;
* dažādi papīra veidi;
* dažādas fotografēšanas un skenēšanas kvalitātes.

### 2. Reālā rokraksta augmentācija

Var pievienot:

* papīra tekstūru;
* tintes izplūšanu;
* ēnas;
* fotografēšanas troksni;
* perspektīvas deformācijas;
* apgaismojuma izmaiņas;
* nelielu izplūšanu.

### 3. Modeļa kvalitātes mērīšana

Jāpievieno objektīvi rādītāji, piemēram:

* CER — Character Error Rate;
* WER — Word Error Rate.

Tas ļaus salīdzināt:

```text
Greedy
vs.
Beam Search
```

un izmērīt, vai Beam Search patiešām uzlabo atpazīšanas kvalitāti.

### 4. Web lietotnes savienošana ar modeli

Nākamais lielais solis ir izveidot backend API:

```text
Web app
   ↓
HTTP API
   ↓
Python / PyTorch
   ↓
CRNN
   ↓
Beam Search
   ↓
JSON ar atpazīto tekstu
   ↓
Web app
```

Tad lietotājs varēs augšupielādēt īstu rokraksta foto un saņemt reālu modeļa rezultātu.

### 5. Labāks checkpoint/resume mehānisms

Pilnvērtīgai ilgstošai apmācībai vēlams saglabāt ne tikai modeļa svarus, bet arī:

* optimizer stāvokli;
* pašreizējo `step`;
* scheduler stāvokli, ja tas tiks pievienots;
* citus apmācības parametrus.

Tas ļaus pilnībā turpināt apmācību no iepriekšējā stāvokļa.

---

# Alpha versijas statuss

| Funkcija                          | Statuss |
| --------------------------------- | ------- |
| Latviešu alfabēts                 | ✅       |
| Sintētisko datu ģenerēšana        | ✅       |
| Reālo datu ielāde                 | ✅       |
| CNN                               | ✅       |
| 2×BiLSTM                          | ✅       |
| Linear + CTC                      | ✅       |
| Mainīga attēlu platuma apstrāde   | ✅       |
| Individuālie `input_lengths`      | ✅       |
| Greedy decoding                   | ✅       |
| CTC Prefix Beam Search            | ✅       |
| `beam-width` konfigurācija        | ✅       |
| GPU/CUDA atbalsts                 | ✅       |
| Checkpoint saglabāšana            | ✅       |
| Viena attēla inference            | ✅       |
| React web lietotne                | ✅       |
| Web lietotnes UI latviešu valodā  | ✅       |
| Web lietotnes savienojums ar CRNN | ❌       |
| Liela reāla rokraksta datu kopa   | ❌       |
| CER/WER automātiska novērtēšana   | ❌       |
| Gala produkta kvalitāte           | ❌       |

**Pašreizējais galvenais mērķis:** izveidot pietiekami lielu un daudzveidīgu reāla latviešu rokraksta datu kopu, lai CRNN+CTC modeli varētu apmācīt uz reāla rokraksta, izmērīt CER/WER un pēc tam savienot modeli ar web lietotni.

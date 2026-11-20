# LVRA, Latviešu valodas rokraksta atpazīšana

**LVRA** ir eksperimentāla sistēma, kas no rokrakstā rakstīta latviešu teksta attēla iegūst rediģējamu digitālu tekstu. Projektu veido divas daļas:

| Daļa | Mape | Tehnoloģijas | Uzdevums |
| --- | --- | --- | --- |
| Atpazīšanas modelis | `latvian_htr/` | Python, PyTorch | Apmācīt un palaist CRNN + CTC modeli, kas nolasa vienu teksta rindu no attēla |
| Web lietotne un serveris | `web-app/` | React 18, TypeScript, Vite; Flask, numpy | Telefonam pielāgota saskarne latviešu valodā: foto → rindas → teksts → labošana → kopēšana. Serveris palaiž modeli bez PyTorch |

> **Statuss: beta.** Web lietotne ir savienota ar reālo modeli: pārlūks sadala foto rindās, serveris (`web-app/server`, PythonAnywhere) katru rindu nolasa ar modeli `crnn_best_migrated.pt`, kas pārrakstīts uz numpy. Atpazīšanas kvalitāte uz reāla rokraksta vēl nav pietiekama gala produktam (skat. [Testēšanas rezultāti](#testēšanas-rezultāti) un [Zināmie ierobežojumi](#zināmie-ierobežojumi)).

---

## Satura rādītājs

1. [Statuss](#statuss)
2. [Ātrais starts](#ātrais-starts)
3. [Modeļa izmantošana](#modeļa-izmantošana)
4. [Kā modelis darbojas](#kā-modelis-darbojas)
5. [Dati](#dati)
6. [Web lietotne](#web-lietotne)
7. [Projekta struktūra](#projekta-struktūra)
8. [Zināmie ierobežojumi](#zināmie-ierobežojumi)
9. [Ceļvedis](#ceļvedis)
10. [Papildu dokumentācija](#papildu-dokumentācija)
11. [Testēšanas rezultāti](#testēšanas-rezultāti)

---

## Statuss

| Komponente | Statuss |
| --- | --- |
| Latviešu rakstzīmju kopa (137 klases, ieskaitot CTC `blank`) | ✅ |
| Čekpointa pārnešana uz jaunu alfabētu (`migrate_checkpoint.py`) | ✅ |
| Sintētisko datu ģenerators (rokraksta imitācija, svītrojumi ar labojumiem, burtnīcas papīrs) | ✅ |
| Sintētiskās datu kopas ģenerēšana uz diska un ielāde no RAM (`make_synth_dataset.py`, `packed_lines.py`) | ✅ |
| Reālo datu ielāde (`LineFolderDataset`) | ✅ |
| CRNN modelis (CNN + 2×BiLSTM + Linear), `x_stride` 2 vai 4 | ✅ |
| Mainīga attēla platuma atbalsts (individuāli `input_lengths`, batch no līdzīga platuma rindām) | ✅ |
| Apmācība ar CTC, Adam, `ReduceLROnPlateau`, gradientu ierobežošana, mixed precision | ✅ |
| Pārbaude apmācības laikā (loss un CER uz atdalītiem piemēriem), labākā modeļa saglabāšana, agrā apstāšanās | ✅ |
| Viena attēla atpazīšana (`infer.py`) ar CTC Prefix Beam Search | ✅ |
| Novērtēšana ar CER (`evaluate.py`, greedy dekodēšana) | ✅ |
| Marķējuma pārbaudes rīks ar apgriešanu, tālummaiņu un Git commit (`verify_labels_git_commit.py`) | ✅ |
| Web lietotnes saskarne latviešu valodā, vairāki foto, rezultāti cilnēs | ✅ |
| Foto sadalīšana rindās pārlūkā (ar slīpuma izlīdzināšanu) | ✅ |
| Web lietotnes savienojums ar modeli (Flask serveris, modelis uz numpy) | ✅ |
| Pietiekami liela reāla rokraksta datu kopa | daļēji (4146 rindas) |
| WER, kā arī Greedy un Beam Search salīdzinājums | ❌ |
| Optimizētāja stāvokļa saglabāšana (pilna apmācības atsākšana) | ❌ |

---

## Ātrais starts

### Prasības

- **Python 3.10 vai jaunāks** (kodā tiek lietota sintakse `list[str] | None`)
- Python bibliotēkas modelim: `torch>=2.2`, `numpy`, `pillow`, `opencv-python-headless` (skat. `latvian_htr/requirements.txt`)
- **Node.js (LTS)**, tikai web lietotnes kompilēšanai
- NVIDIA GPU nav obligāts; ja CUDA ir pieejama, tā tiek izmantota automātiski, citādi apmācība notiek uz CPU

### Python daļa

```bash
cd latvian_htr
pip install -r requirements.txt

# Pārbaudīt apmācītu modeli uz vienas reāla rokraksta rindas
python infer.py \
  --checkpoint checkpoints_v2/crnn_best_migrated.pt \
  --image data/real_dataset/images/0001.png
```

Rezultāts tiek izdrukāts formā `Teksts: '...'`.

### Web lietotne

```bash
cd web-app
npm install
pip install -r requirements.txt
npm run build
python server/app.py
```

Atver `http://127.0.0.1:5000`. Tieši tas pats darbojas PythonAnywhere. Izstrādes režīms ar automātisku pārlādi, palaišana uz telefona un izvietošana PythonAnywhere ir aprakstīta failā [`web-app/README.md`](web-app/README.md).

---

## Modeļa izmantošana

Visas komandas tiek izpildītas mapē `latvian_htr/`.

### 1. Sintētisko paraugu apskate

```bash
python synthetic_data.py
```

Izveido 20 paraugus mapē `samples/` (`sample_0.png` … `sample_19.png`) un izdrukā to tekstus. Noder vizuālai ģeneratora pārbaudei.

### 2. Sintētiskās datu kopas sagatavošana (ieteicams)

```bash
# vienreiz uzģenerēt kopu uz diska (izmanto visus procesora kodolus)
python make_synth_dataset.py --out data/synth_30k --n 30000

# neobligāti: iepakot kopu vienā failā, ko apmācība lasa no RAM (pēc katras jaunas ģenerēšanas)
python packed_lines.py data/synth_30k
```

`make_synth_dataset.py` saglabā attēlus mapē `<out>/images/` un `<out>/labels.csv` ar kolonnām `filename,text,width`. Vienāds `--seed` dod vienādu kopu. `packed_lines.py` saglabā gatavus attēlus mapē `<kopa>/packed/`; ja `labels.csv`, `dataset.py`, alfabēts vai `TARGET_HEIGHT` pēc iepakošanas mainās, `train.py` to pamana, lasa PNG failus un iesaka komandu pārpakošanai.

### 3. Apmācība

```bash
# no gatavas sintētiskās kopas (500 piemēri automātiski tiek atdalīti pārbaudei)
python train.py --synth-data data/synth_30k --steps 20000

# sintētiskie + reālie dati
python train.py --synth-data data/synth_30k --real-data data/real_dataset

# apmēram 2 reizes ātrāk (platums CNN tiek saspiests 4 reizes, skat. model.py)
python train.py --synth-data data/synth_30k --x-stride 4
```

Bez `--synth-data` sintētiskie dati tiek ģenerēti apmācības laikā (katra rinda jauna, bet vajadzīgs jaudīgs procesors).

| Parametrs | Noklusējums | Nozīme |
| --- | --- | --- |
| `--steps` | `20000` | Apmācības soļu skaits (1 solis = 1 batch) |
| `--batch-size` | `64` | Piemēru skaits vienā solī |
| `--lr` | `1e-3` | Sākotnējais mācīšanās ātrums |
| `--synth-data` | nav | Mape ar gatavu sintētisko kopu (`labels.csv` + `images/`) |
| `--val-data` | nav | Atsevišķa pārbaudes kopa (neobligāti) |
| `--val-size` | `500` | Cik piemēru atdalīt no `--synth-data` pārbaudei, ja `--val-data` nav norādīta |
| `--real-data` | nav | Mape ar reālajiem datiem; tiek pievienota sintētiskajiem |
| `--real-fraction` | nav | Tikai ģenerēšanai apmācības laikā: reālo rindu daļa batch (0..1) |
| `--synth-per-epoch` | `5000` | Tikai ģenerēšanai apmācības laikā: vajadzīgs reālo datu daļas aprēķinam |
| `--x-stride` | `2` | Platuma saspiešana CNN: `2` vai `4` (ātrāk) |
| `--no-amp` | izslēgts | Izslēdz mixed precision (fp16) |
| `--num-workers` | atkarīgs no datora | Paralēlie datu sagatavotāji |
| `--log-every` | `500` | Cik bieži izdrukāt zudumus |
| `--checkpoint-every` | `1000` | Cik bieži saglabāt svarus un veikt pārbaudi |
| `--checkpoint-dir` | `checkpoints` | Kur saglabāt čekpointus |
| `--resume` | nav | Ielādē svarus no esoša čekpointa un turpina apmācību |
| `--seed` | `0` | Nejaušības sākumvērtība, lai palaišanu varētu atkārtot |
| `--early-stop` | `0` | Apstāties, ja pārbaude N reizes pēc kārtas neuzlabojas (`0` = izslēgts) |

Tabulā «nav» nozīmē, ka parametram nav noklusējuma vērtības.

Piezīmes par `--resume`: tiek ielādēti tikai **modeļa svari** (nevis optimizētājs vai soļu skaitītājs), tāpēc soļu skaitītājs sākas no 0 un jauni čekpointi var pārrakstīt vecos, ja `--checkpoint-dir` nav mainīts.

### 4. Atpazīšana (inference)

```bash
python infer.py \
  --checkpoint checkpoints_v2/crnn_best_migrated.pt \
  --image data/real_dataset/images/0001.png \
  --beam-width 10
```

`--beam-width` (noklusējums `5`) nosaka, cik labāko teksta variantu tiek saglabāti dekodēšanas laikā. Lielāka vērtība nozīmē vairāk pārbaudītu variantu un nedaudz lēnāku darbību. Parametra maiņa **neprasa modeļa pārtrenēšanu**.

Modelis paredzēts **vienai teksta rindai**, nevis visai lapai. Visas lapas sadalīšanu rindās veic web lietotne.

### 5. Kvalitātes novērtēšana (CER)

```bash
python evaluate.py --checkpoint checkpoints_v2/crnn_best_migrated.pt
python evaluate.py --checkpoint checkpoints_v2/crnn_best_migrated.pt --real-data data/real_dataset --synthetic 500
```

Skripts izdrukā **CER** (Character Error Rate = labojumu skaits / pareizo rakstzīmju skaits; mazāks ir labāk) atsevišķi reālajiem (`--real-data`, noklusējums `data/real_dataset`) un svaigi ģenerētajiem sintētiskajiem paraugiem (`--synthetic`, noklusējums `200`, `0` = nepārbaudīt), kā arī dažus piemērus «pareizi / modelis».

Svarīgi:
- Novērtēšana izmanto **greedy** dekodēšanu, nevis Beam Search.
- Ja reālie dati tika iekļauti apmācībā (`--real-data`), CER uz tiem ir **optimistisks**. Godīgam rezultātam reālos paraugus vēlams atdalīt no apmācības.

### 6. Čekpointa pārnešana uz jaunu alfabētu

Ja alfabētam (`alphabet.py`) pievieno vai noņem rakstzīmes, vecais čekpoints vairs neder (pēdējā slāņa izmērs neatbilst, un rakstzīmju numuri nobīdās). `migrate_checkpoint.py` pārnes svarus, nemācot tīklu no nulles:

```bash
python migrate_checkpoint.py --old-alphabet old_alphabet.py \
  --checkpoint checkpoints_v2/crnn_best.pt --out checkpoints_v2/crnn_best_migrated.pt
python train.py --resume checkpoints_v2/crnn_best_migrated.pt --lr 3e-4 --checkpoint-dir checkpoints_v3
```

Rakstzīmēm, kas bija abos alfabētos, tiek pārkopēti to svari; jaunās rakstzīmes saņem mazas sākuma vērtības un tiek iemācītas tikai pēc tālākas apmācības. `old_alphabet.py` ir alfabēts tāds, kāds tas bija, kad tika apmācīts `crnn_best.pt` (131 klase). Tieši tā ir izveidots `crnn_best_migrated.pt` (137 klases), ko tagad izmanto visur.

---

## Kā modelis darbojas

Izmantota **CRNN** arhitektūra (konvolucionāls tīkls + rekurents tīkls) ar **CTC** zudumu funkciju. CNN un BiLSTM ir savienoti **secīgi**: CNN izvade ir BiLSTM ievade.

```text
Attēls [B, 1, 64, W]
   │
   ▼  CNN: vizuālo pazīmju iegūšana, augstums 64 → 1
[B, 512, 1, W/2]      (ar x_stride 4: W/4)
   │
   ▼  attēla platums kļūst par laika asi T = W / 2 (vai W / 4)
[T, B, 512]
   │
   ▼  2 × BiLSTM (hidden = 256): konteksts abos virzienos
[T, B, 512]
   │
   ▼  Linear → log_softmax
[T, B, 137]   rakstzīmju varbūtības katrā solī
   │
   ▼  CTC dekodēšana (Beam Search)
Teksts
```

| Daļa | Fails | Apraksts |
| --- | --- | --- |
| Alfabēts | `alphabet.py` | Rakstzīmju ↔ indeksu tabula, `encode`, `decode_greedy`, `decode_beam_search`, `strip_struck` |
| CNN | `model.py` → `CNNBackbone` | Konvolūcijas + MaxPool; platumu saspiež vienreiz (×2) vai divreiz (×4, `x_stride=4`), augstumu līdz 1 |
| BiLSTM + Linear | `model.py` → `BiLSTMHead` | 2 slāņi, divvirzienu; izvada rezultātu katrai rakstzīmei |
| Modelis | `model.py` → `CRNN` | Savieno abas daļas; `output_length(W)` = `max(W // 2, 1)` vai `max(W // 4, 1)` |
| Čekpointi | `model.py` → `save_checkpoint`, `load_checkpoint`, `load_model` | Čekpointā tiek saglabāti svari, `x_stride` un soļa numurs; tiek nolasīts arī vecais formāts (tikai svari) |
| Datu ielāde | `dataset.py` | `SyntheticHTRDataset`, `LineFolderDataset`, `BucketBatchSampler`, `SyntheticBucketStream`, `collate_batch`, `image_to_tensor` |

Abiem pašreizējiem čekpointiem (`crnn_best.pt`, `crnn_best_migrated.pt`) ir `x_stride = 4`.

### Attēla sagatavošana

Gan apmācībā, gan atpazīšanā attēls tiek apstrādāts vienādi:

1. pārveidošana pelēktoņos;
2. mērogošana līdz **64 px augstumam** (proporcijas saglabātas);
3. automātiskais kontrasts (`autocontrast`), lai pelēks vai tumšs fons kļūtu gaišs;
4. krāsu apgriešana (fons = 0, tinte ≈ 1);
5. tensors `[1, 64, W]`.

### Mainīga platuma attēli

`BucketBatchSampler` vienā batch saliek līdzīga platuma rindas, tāpēc papildināšanas ar nullēm ir maz. Lai CTC neuztvertu papildinājumu kā saturu, `train.py` katram piemēram aprēķina individuālu `input_length` (`model.output_lengths`).

> **Ja maināt CNN** (pooling vai kernel izmērus), jāpielāgo arī `CRNN.output_length` un `output_lengths`, citādi CTC saņems nepareizus garumus. Serverī tas pats jāatspoguļo failā `web-app/server/htr_numpy.py`.

### CTC un dekodēšana

CTC ļauj apmācīt modeli, nenorādot, kur attēlā atrodas katrs burts. Modelis katrā laika solī izvēlas rakstzīmi vai `blank`.

- **Greedy** (`decode_greedy`): katrā solī ņem visvarbūtīgāko klasi, sakļauj atkārtojumus, noņem `blank`. Izmanto `evaluate.py` un pārbaude apmācības laikā.
- **CTC Prefix Beam Search** (`decode_beam_search`): saglabā `beam_width` labākos teksta prefiksus. Izmanto `infer.py` un serveris.

### Apmācības cikls

`train.py` izmanto `Adam`, `CTCLoss(blank=0, zero_infinity=True)`, gradientu ierobežošanu (norma 5.0) un `PlateauScheduler` (ik pēc 200 soļiem pārbauda vidējo zudumu; ja tas neuzlabojas, `lr` tiek samazināts uz pusi). Uz CUDA tiek izmantota mixed precision (fp16) ar `GradScaler`. Pie katras čekpointa saglabāšanas modelis tiek pārbaudīts uz piemēriem, uz kuriem tas nemācās (loss un CER), un labākais rezultāts tiek saglabāts atsevišķi (`crnn_best.pt` mapē `--checkpoint-dir`, pārējie čekpointi `crnn_step<N>.pt`); ar `--early-stop N` apmācība apstājas, ja pārbaude N reizes pēc kārtas neuzlabojas.

---

## Dati

### Rakstzīmju kopa

`alphabet.py` satur 136 rakstzīmes + `blank` (indekss 0), kopā **137 klases**:

- latviešu burti ar diakritiskajām zīmēm (`ā č ē ģ ī ķ ļ ņ š ū ž`), lielie un mazie;
- `q w x y` (un lielie);
- cipari `0` līdz `9`;
- atstarpe, pieturzīmes, pēdiņas (arī `„ “ ” « »`), domuzīmes, iekavas, simboli, `€ $ № …`;
- `° ≈ ö ü Ç Ć` (pievienoti pēc `crnn_best.pt` apmācības; iepriekšējais alfabēts ar 131 klasi ir `old_alphabet.py`);
- speciālais simbols **`¤`** (`STRIKE_MARK`), kas norāda **pārsvītrotu vārdu**.

Funkcija `strip_struck(text)` noņem `¤` no gala teksta (`"Ekonomika ¤ tirgus"` → `"Ekonomika tirgus"`).

Ja teksts satur rakstzīmi ārpus šīs kopas, `encode` izmet `ValueError`.

### Reālie dati

Atrodas `latvian_htr/data/real_dataset/`:

```text
data/real_dataset/
├── labels.csv          # kolonnas: filename,text
└── images/
    ├── 0001.png
    └── ...
```

Latviešu teksts sintētisko datu vārdiem ir failā `latvian_htr/data/corpus.txt`.

Pašlaik: **4146 marķētas teksta rindas** (`0001.png` līdz `4146.png`, kopā ≈ 170 000 rakstzīmju, no tām 172 atzīmes `¤`), katrs attēls ir viena rinda.

`labels.csv` formāts (UTF-8):

```csv
filename,text
0001.png,Ekonomika izdala (3 vai 4) tirgus dalībniekus
0002.png,"1,3,5,7 - reala ekonomika,"
```

Teksts nedrīkst saturēt rakstzīmes ārpus alfabēta. Pārsvītrotu vārdu marķēšanai izmanto `¤`.

### Marķēšanas noteikumi

Reālo datu rindas tiek marķētas pēc šādiem principiem:

- teksts tiek pārrakstīts **tieši tā, kā tas uzrakstīts attēlā**: bez gramatikas un pareizrakstības labošanas un bez vārdu «pielāgošanas» pareiziem latviešu vārdiem; autora kļūdas paliek etiķetē;
- viens attēla fails = viena teksta rinda; faila nosaukums sastāv no četriem cipariem (`0042.png`), numerācija turpinās bez atkārtojumiem;
- tiek pārrakstīts tikai **audzēkņa rokraksts**: skolotāja sarkanie labojumi, atzīmes un punkti etiķetē netiek iekļauti;
- **pārsvītrotie vārdi** dažās partijās etiķetē tika **izlaisti** (nevis aizvietoti ar `¤`), bet rindas, kurās pārsvītrota lielākā daļa teksta, netika iekļautas vispār. Ja vēlaties, lai modelis mācītos atpazīt svītrojumus, šajās vietās jāliek `¤` (skat. [Rakstzīmju kopa](#rakstzīmju-kopa));
- vārdi, kurus attēlā nevar droši izlasīt, ir pārrakstīti pēc labākā minējuma; tos vēlams pārbaudīt ar zemāk aprakstīto rīku;
- etiķetē drīkst būt tikai alfabēta rakstzīmes. Pārbaudes rīks alfabētu **nepārbauda**, tāpēc pirms apmācības jāpārliecinās, ka `encode` izdodas visām rindām.

### Marķējuma pārbaudes rīks

`verify_labels_git_commit.py` ir neliela darbvirsmas programma (Python + Tkinter, tumšā tēma), kas ļauj rindu pa rindai pārbaudīt un labot `labels.csv`, vienlaikus redzot attēlu, un iesniegt labojumus Git. Mapē ir arī `verity_labels.py`, cits šī rīka variants; šis apraksts attiecas uz `verify_labels_git_commit.py`.

```text
┌───────────────────────────┬──────────────────────────────────┐
│ 42 / 4146                 │                                  │
│ 0042.png                  │                                  │
│ ┌───────────────────────┐ │        rindas attēls             │
│ │ rindas teksts         │ │   (tālummaiņa, pārvietošana,     │
│ │ (var labot)           │ │    apgriešana)                   │
│ └───────────────────────┘ │                                  │
│ [Crop] [Original]         │                                  │
│ [Deselect]                │                                  │
│ [Commit changes to Git]   │                                  │
└───────────────────────────┴──────────────────────────────────┘
```

**Palaišana** (mapē `latvian_htr/`):

```bash
pip install pillow
python verify_labels_git_commit.py
```

Tkinter ir Python standarta komplektā (Windows). Ceļi tiek noteikti attiecībā pret faila atrašanās vietu: rīks lasa `data/real_dataset/labels.csv` un attēlus no `data/real_dataset/images/`, tātad darbojas jebkurā datorā bez ceļu maiņas. Logā ir arī nelieli GIF lauki, kuru attēli tiek ņemti no mapes `latvian_htr/prikol/`.

**Taustiņi un darbības**

| Darbība | Taustiņš / vadīkla |
| --- | --- |
| Saglabāt labojumu un pāriet uz nākamo rindu | `Enter` |
| Iepriekšējā / nākamā rinda bez labojuma | `Alt+←` / `Alt+→` (vai `Ctrl+←` / `Ctrl+→`) |
| 10 rindas atpakaļ / uz priekšu | `PageUp` / `PageDown` |
| Pāriet uz numuru vai faila nosaukumu (piem., `0215`) | `Ctrl+F` |
| Saglabāt `labels.csv` nekavējoties | `Ctrl+S` |
| Atcelt labojumus pašreizējā rindā (atgriezt sākotnējo tekstu) | `Ctrl+Z` |
| Apgriezt attēlu pēc iezīmētās zonas | `Ctrl+R` vai poga «Crop» |
| Atgriezt neapgriezto oriģinālu | `Ctrl+B` vai poga «Original» |
| Noņemt iezīmējumu | `Esc` vai poga «Deselect» |
| Tuvināt / attālināt attēlu | `Ctrl` + peles ritenis, `Ctrl+=` / `Ctrl+-` |
| Atgriezt sākotnējo mērogu | `Ctrl+0` |
| Pārvietot attēlu | peles ritenis (`Shift` + ritenis horizontāli), vilkšana ar labo vai vidējo pogu |
| Saglabāt un iesniegt izmaiņas Git | poga «Commit changes to Git» |

**Attēla apgriešana.** Attēlā ar nospiestu kreiso peles pogu jāiezīmē zona (dzeltens raustīts rāmis) un jānospiež `Ctrl+R`. Apgrieztais attēls **uzreiz pārraksta failu** mapē `images/`; pirms pirmās apgriešanas oriģināls tiek nokopēts mapē `images/_originals/`. Pie faila nosaukuma parādās atzīme «[apgriezts]». Atgriežot oriģinālu (`Ctrl+B`), kopija no `_originals/` tiek atjaunota un dzēsta.

**Saglabāšana.**

- Attēla apgriešana tiek saglabāta uz diska **nekavējoties**.
- Teksta labojumi tiek ierakstīti `labels.csv` pie `Enter`, `Ctrl+S` un loga aizvēršanas. Pārvietojoties ar `Alt+←/→` vai `PageUp/PageDown`, labojums tiek atcerēts programmā, bet failā parādās pie nākamā `Enter` / `Ctrl+S` / aizvēršanas.
- Pirms pirmās saglabāšanas tiek izveidota rezerves kopija `labels.csv.bak`.
- Pozīcija tiek atcerēta failā `images/.verify_progress`, loga iestatījumi failā `images/.verify_settings.json`; nākamreiz programma turpina no tās pašas rindas.
- Poga «Commit changes to Git» vispirms saglabā `labels.csv`, tad izpilda `git add` mainītajiem failiem un `git commit` ar ziņojumu `real data 0.<N> checked till <fails>`. Izmaiņas netiek nosūtītas uz GitHub (`git push` jāizpilda pašam).

> Ja pēc apgriešanas atkārtoti palaižat attēlu sagriešanas skriptu, tas var pārrakstīt apgrieztos failus; oriģināli paliek `_originals/`.

### Sintētiskie dati

`synthetic_data.py` (versija 3.1) katru reizi ģenerē jaunu piemēru:

1. izvēlas fontu no `fonts/` (ap 20 fontu) un tekstu, ko šis fonts **pilnībā** spēj uzzīmēt (`font_charset` izslēdz rakstzīmes, kurām fontā nav glifa);
2. salasa nejaušu «rindu» no `data/corpus.txt` vārdiem (ap 7 % gadījumu skaitļi, retāk romiešu cipari);
3. 80 % rindu zīmē **burtu pa burtam** ar nelielām izmēra, augstuma un slīpuma svārstībām; puse no tām ar savienotiem burtiem (kursīvs);
4. 20 % rindu satur pārsvītrotus vārdus (līnija, murgojums, krusti, vairākas līnijas). 60 % svītrojumu ir «kļūdas» (pārrakstīšanās, trūkstoša vai lieka diakritiskā zīme, sajaukti burti), virs kurām mazākiem burtiem uzrakstīts pareizais vārds; etiķetē ir pareizais vārds. Svītrojumi bez labojuma etiķetē tiek aizvietoti ar `¤`;
5. pievieno burtnīcas lapas foto imitāciju: rūtiņu vai līniju papīru, nevienmērīgu apgaismojumu, viļņainu rindu, troksni, izplūdumu un JPEG artefaktus (pēc samazināšanas līdz 64 px).

Sintētiskie dati ir paredzēti, lai modelis varētu mācīties, kamēr trūkst reālā rokraksta. **Tie neaizvieto īstu cilvēku rokrakstu.**

---

## Web lietotne

React + Vite + TypeScript lietotne un Flask serveris mapē `web-app/`. Visi lietotāja teksti ir latviešu valodā (`src/i18n/lv.ts`, garie teksti `TermsDialog.tsx` un `PrivacyNotice.tsx`); izstrādātāju dokumentācija ir angļu valodā failā [`web-app/README.md`](web-app/README.md).

**Lietotāja ceļš:** pirmajā apmeklējumā «Lietošanas noteikumi» (piekrišana AI apmācībai ir brīvprātīga) → izvēlēties vienu vai vairākus attēlus vai uzņemt foto → priekšskatījums (var pagriezt, noņemt, pievienot vēl) → **Atpazīt tekstu** → progress ar atcelšanu → rezultāts, ko var labot (vairākiem foto katram sava cilne) → **Kopēt tekstu** / **Kopēt visu**.

Svarīgākais:

- attēls pārlūkā tiek pagriezts un samazināts līdz 1920 × 1080 (portretā 1080 × 1920);
- pārlūks sadala foto **rindās** (`src/image/lineSegmentation.ts`): izlīdzina apgaismojumu, atrod tintes formas, izmet malu objektus (galds, blakus lapa, pirksti), izlīdzina slīpumu līdz 12°, sagrupē formas rindās un katrā rindas attēlā atstāj tikai šīs rindas tinti. Pilnais foto netiek sūtīts;
- serverim tiek nosūtīti rindu PNG faili `0001.png`, `0002.png`, ..., `labels.csv` (kolonna `text` tukša) un lietotāja piekrišanas izvēle; serveris aizpilda `text` un atgriež CSV;
- serveris (`server/app.py`) palaiž modeli bez PyTorch (`server/htr_numpy.py`), jo PyTorch neietilpst bezmaksas PythonAnywhere diskā. Rakstzīmju tabula (`server/alphabet_table.py`) tiek izvēlēta pēc klašu skaita čekpointā: 137 (`crnn_best_migrated.pt`) vai 131 (`crnn_best.pt`);
- rindu attēli tiek saglabāti AI apmācībai tikai tad, ja lietotājs piekrita visiem noteikumiem;
- atpazīšanas laika limits: 60 s katram foto;
- pēc veiksmīgas atpazīšanas attēli tiek dzēsti no atmiņas, paliek tikai teksts;
- izstrādātāja panelis: pievieno adresei `?dev=1` (servera vai parauga teksta režīms, kļūdu scenāriji, pēdējā pieprasījuma rindu attēli un CSV).

Saskarne ar atpazīšanu ir atdalīta ar interfeisu `HandwritingRecognitionService` (`src/recognition/types.ts`); servisu izvēlas `src/recognition/createRecognitionService.ts`. Kļūdām tiek izmantoti kodi no `src/recognition/errors.ts`, un lietotājs tos redz kā latviskus paziņojumus.

> Modelis atgriež tekstu bez ticamības vērtībām katram vārdam, tāpēc nedrošo vārdu izcelšana (`config.highlightUncertainWords`) pagaidām ir izslēgta.

---

## Projekta struktūra

```text
LVRA/
├── README.md
├── Proekta-skaidrošana.md        # detalizēts modeļa un sistēmas skaidrojums (krievu val.)
│
├── latvian_htr/                  # Python daļa (modelis)
│   ├── alphabet.py               # rakstzīmju kopa (137 klases), kodēšana, dekodēšana
│   ├── old_alphabet.py           # iepriekšējā rakstzīmju kopa (131 klase), crnn_best.pt
│   ├── model.py                  # CRNN, čekpointu saglabāšana un ielāde
│   ├── dataset.py                # datu kopas, batch no līdzīga platuma rindām
│   ├── synthetic_data.py         # sintētisko attēlu ģenerators
│   ├── make_synth_dataset.py     # sintētiskās kopas ģenerēšana uz diska
│   ├── packed_lines.py           # kopas iepakošana lasīšanai no RAM
│   ├── train.py                  # apmācība
│   ├── infer.py                  # viena attēla atpazīšana
│   ├── evaluate.py               # CER novērtēšana
│   ├── migrate_checkpoint.py     # čekpointa pārnešana uz jaunu alfabētu
│   ├── verify_labels_git_commit.py  # marķējuma pārbaudes rīks (GUI, apgriešana, Git commit)
│   ├── verity_labels.py          # cits pārbaudes rīka variants
│   ├── requirements.txt
│   ├── fonts/                    # fonti sintētiskajiem datiem
│   ├── samples/                  # synthetic_data.py paraugi
│   ├── prikol/                   # attēli un GIF pārbaudes rīka laukiem
│   ├── checkpoints_v2/           # crnn_best_migrated.pt (izmanto visur), crnn_best.pt (vecāks), ≈ 35 MB katrs
│   └── data/
│       ├── corpus.txt            # latviešu teksts sintētisko datu vārdiem
│       └── real_dataset/         # reālie dati (labels.csv, images/)
│           └── images/           # rindu attēli; pārbaudes rīks šeit izveido _originals/, .verify_progress, .verify_settings.json
│
└── web-app/                      # web lietotne un serveris
    ├── src/
    │   ├── app/                  # ekrānu stāvoklis
    │   ├── components/           # UI komponentes
    │   ├── consent/              # lietošanas noteikumu izvēles saglabāšana
    │   ├── image/                # attēla apstrāde un sadalīšana rindās pārlūkā
    │   ├── recognition/          # atpazīšanas interfeiss, servera serviss, parauga serviss, kļūdas
    │   ├── editor/, utils/       # rezultāta labošana un kopēšana
    │   ├── i18n/lv.ts            # īsie lietotāja teksti
    │   └── config/config.ts      # fiksētās vērtības (limiti, sliekšņi, noteikumu versija)
    ├── server/                   # Flask serveris un modelis uz numpy
    │   ├── app.py
    │   ├── htr_numpy.py
    │   └── alphabet_table.py
    ├── public/
    ├── dist/                     # kompilētā lietotne (npm run build), ko izsniedz serveris
    ├── requirements.txt          # servera Python bibliotēkas
    └── package.json
```

---

## Zināmie ierobežojumi

- **Atpazīšanas kvalitāte uz reāla rokraksta** vēl ir zema (skat. [Testēšanas rezultāti](#testēšanas-rezultāti)); lietotne lietotājam to norāda.
- **Jaunās rakstzīmes `° ≈ ö ü Ç Ć`** čekpointā `crnn_best_migrated.pt` ir tikai pārnestas, nevis iemācītas (`migrate_checkpoint.py` tām piešķir sākuma vērtības). Kamēr modelis nav apmācīts tālāk, tās var parādīties atpazītajā tekstā nejauši.
- **Maz reālu datu:** 4146 rindas joprojām ir maz, lai modelis labi vispārinātu uz dažādiem rokrakstiem. Sintētiskie dati nav īsts rokraksts.
- **Rindu sadalīšana pārlūkā** darbojas ar parastām, arī slīpām (līdz 12°) un cieši izvietotām rindām, bet stipri izliektas rindas vai rindas, kuru burti pieskaras viena otrai lielā garumā, var tikt sadalītas neprecīzi. Teksts no blakus lapas, kas pilnībā redzams foto (nepieskaras foto malai), var nonākt rindu attēlos.
- **CTC garuma ierobežojums:** izvades garums `T = W / 2` (ar `x_stride=4`: `W / 4`) nedrīkst būt mazāks par teksta garumu; pārāk šauri attēli ar garu tekstu netiks pareizi apmācīti (`train.py` par to brīdina).
- **`--resume`** atjauno tikai svarus, nevis pilnu apmācības stāvokli.
- **Novērtēšana** `evaluate.py` šobrīd tikai CER ar greedy dekodēšanu; WER un Beam Search novērtējuma nav.
- **Fontu licences:** pirms projekta publiskas izplatīšanas jāpārbauda mapē `fonts/` esošo fontu licences.
- **Pārbaudes rīks (`verity_labels.py`):** nepārbauda, vai etiķete satur tikai alfabēta rakstzīmes; apgriešana pārraksta attēlu failus (oriģināli paliek `_originals/`); projektā nav automātisku testu šim rīkam.
- **Pārsvītrotie vārdi:** dažās marķēto datu partijās tie ir izlaisti, nevis atzīmēti ar `¤`, tāpēc modelis no šīm rindām neiemācās atpazīt svītrojumus.

---

## Ceļvedis

1. Apmācīt `crnn_best_migrated.pt` tālāk, lai modelis iemācītos jaunās rakstzīmes un labāk lasītu reālo rokrakstu.
2. Savākt un marķēt vēl lielāku, daudzveidīgāku reālā rokraksta datu kopu (dažādi cilvēki, pildspalvas, papīrs, foto kvalitāte), tostarp no lietotājiem, kas piekrita AI apmācībai; visu jauno marķējumu pirms apmācības pārbaudīt ar `verify_labels_git_commit.py`.
3. Turēt reālo datu pārbaudes kopu atsevišķi no apmācības un regulāri mērīt CER.
4. Pievienot WER un salīdzināt Greedy ar Beam Search.
5. Papildināt augmentāciju reālajiem datiem (papīra tekstūra, ēnas, perspektīva, apgaismojums).
6. Saglabāt čekpointā arī optimizētāja un soļu stāvokli, lai apmācību varētu pilnībā atsākt.
7. Ticamības vērtības katram vārdam, lai lietotnē varētu atkal ieslēgt nedrošo vārdu izcelšanu.

---

## Papildu dokumentācija

- [`web-app/README.md`](web-app/README.md): web lietotnes un servera uzbūve, izstrādātāja panelis, izvietošana PythonAnywhere (angļu val.).
- [`Proekta-skaidrošana.md`](Proekta-skaidrošana.md): padziļināts modeļa un sistēmas skaidrojums (krievu val.).

---

## Komanda un licence

Projektu izstrādā komanda **Slepie**. Licence pašlaik nav norādīta.

## Testēšanas rezultāti

=== REĀLIE paraugi: 3007 paraugi, CER = 0.622  (precīzi sakrīt: 2/3007) ===
  pareizi : Ekonomika izdala (3 vai 4) tirgus dalībniekus
  modelis : - uunM N°olLa b“ 4) A VVU) lLLlwLXwI)

  pareizi : 1,3,5,7 - reala ekonomika,
  modelis : D257 - MUX DhVMA

  pareizi : 2,4,6,8 - nereala (virtuala) ekonomika,
  modelis : Z.)U) 6(8 - C MNLXLO CLMM≈OLD) LhOIwM

  pareizi : M - money,
  modelis : -II- MONV

  pareizi : Monopolija uz eiro ražošana pieder tikai ECB,
  modelis : MoMja Uz vā Mx HilX thx ECl)


=== SINTĒTISKIE (svaigi): 400 paraugi, CER = 0.010  (precīzi sakrīt: 325/400) ===
  pareizi : vieglāk dzintara aprīkojuma
  modelis : vieglāk dzintara aprīkojuma

  pareizi : prodekāns, veicinātu pulveris.
  modelis : prodekāns, veicinātu pulveris.

  pareizi : centās durvīm arī šādas izstāties, kad Labu atspoguļojums,
  modelis : centās durvīm arī šādas izstāties, kad Labu atspoguļojums,

  pareizi : vecākiem - ii ir kur (3087)
  modelis : vecākiem - ii ir kur (3087)

  pareizi : un ¤ izlutināta,
  modelis : un ¤ izlutināta,

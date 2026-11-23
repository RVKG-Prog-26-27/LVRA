# LVRA, Latviešu valodas rokraksta atpazīšana

**LVRA** ir eksperimentāla sistēma, kas no rokrakstā rakstīta latviešu teksta attēla iegūst rediģējamu digitālu tekstu. Projektu veido divas daļas:

| Daļa | Mape | Tehnoloģijas | Uzdevums |
| --- | --- | --- | --- |
| Atpazīšanas modelis | `latvian_htr/` | Python, PyTorch | Apmācīt un palaist CRNN + CTC modeli, kas nolasa vienu teksta rindu no attēla |
| Web lietotne un serveris | `web-app/` | React 18, TypeScript, Vite; Flask, numpy | Telefonam pielāgota saskarne latviešu valodā: foto → rindas → teksts → labošana → kopēšana. Serveris palaiž modeli bez PyTorch |

> **Statuss: beta.** Web lietotne ir savienota ar reālo modeli: pārlūks sadala foto rindās, serveris (`web-app/server`, PythonAnywhere) katru rindu nolasa ar modeli `crnn_best_migrated.pt`, kas pārrakstīts uz numpy. Atpazīšanas kvalitāte uz reāla rokraksta vēl nav pietiekama gala produktam (skat. [Zināmie ierobežojumi](#zināmie-ierobežojumi)).
>
> **Svarīgi: alfabēts ir paplašināts.** `latvian_htr/alphabet.py` tagad satur vairāk pieturzīmju un simbolu (pašlaik 176 klases), tāpēc iepriekšējais čekpoints `checkpoints_v2/crnn_best_migrated.pt` (137 klases) ar šo kodu vairs nav ielādējams: modelis ir jāapmāca no jauna. Līdz tam web serveris turpina strādāt ar veco čekpointu un savu veco rakstzīmju tabulu (`web-app/server/alphabet_table.py`). Pēc pārtrenēšanas šī tabula jāatjauno (skat. [Alfabēta maiņa](#alfabēta-maiņa)).

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
11. [Vēsturiskie testēšanas rezultāti](#vēsturiskie-testēšanas-rezultāti)

---

## Statuss

| Komponente | Statuss |
| --- | --- |
| Rakstzīmju kopa (pašlaik 176 klases, ieskaitot CTC `blank`), paplašināta ar pieturzīmēm un simboliem | ✅ (modelis jāpārtrenē) |
| Sintētisko datu ģenerators (rokraksta imitācija, svītrojumi ar labojumiem, burtnīcas papīrs, simbolu izteiksmes) | ✅ |
| Rezerves fonts rakstzīmēm, kuru nav rindas fontā (`default_font()`) | ✅ |
| Sintētiskās datu kopas ģenerēšana uz diska un ielāde no RAM (`make_synth_dataset.py`, `packed_lines.py`) | ✅ |
| Reālo datu ielāde (`LineFolderDataset`) un to sadalīšana apmācības / pārbaudes daļās (`split_real`) | ✅ |
| CRNN modelis (CNN + 2×BiLSTM + Linear), `x_stride` 2 vai 4 | ✅ |
| Mainīga attēla platuma atbalsts (individuāli `input_lengths`, batch no līdzīga platuma rindām) | ✅ |
| Apmācība ar CTC, Adam, `ReduceLROnPlateau`, gradientu ierobežošana, mixed precision | ✅ |
| Pārbaude apmācības laikā (loss, CER, WER; sintētiskā un reālā pārbaudes kopa), labākā modeļa saglabāšana, agrā apstāšanās | ✅ |
| Viena attēla atpazīšana (`infer.py`) ar CTC Prefix Beam Search | ✅ |
| Novērtēšana: CER un WER (`evaluate.py`), neobligāts Greedy / Beam Search salīdzinājums | ✅ |
| Marķējuma pārbaudes rīks ar apgriešanu, tālummaiņu un Git commit + push (`verity_labels.py`) | ✅ |
| Web lietotnes saskarne latviešu valodā, vairāki foto, rezultāti cilnēs | ✅ |
| Foto sadalīšana rindās pārlūkā (ar slīpuma izlīdzināšanu) | ✅ |
| Web lietotnes savienojums ar modeli (Flask serveris, modelis uz numpy) | ✅ |
| Pietiekami liela reāla rokraksta datu kopa | daļēji (5677 rindu attēli, skaits aug) |
| Optimizētāja stāvokļa saglabāšana (pilna apmācības atsākšana) | ❌ |

---

## Ātrais starts

### Prasības

- **Python 3.10 vai jaunāks** (kodā tiek lietota sintakse `list[str] | None`)
- Python bibliotēkas modelim: `torch>=2.2`, `numpy`, `pillow`, `opencv-python-headless` (skat. `latvian_htr/requirements.txt`; `opencv` vajadzīgs tikai marķējuma pārbaudes rīkam)
- **Node.js (LTS)**, tikai web lietotnes kompilēšanai
- NVIDIA GPU nav obligāts; ja CUDA ir pieejama, tā tiek izmantota automātiski, citādi apmācība notiek uz CPU
- `alphabet.py` un `metrics.py` neprasa PyTorch, tos var importēt arī vidē bez tā

### Python daļa

```bash
cd latvian_htr
pip install -r requirements.txt

# 1. uzģenerēt sintētisko kopu un apmācīt modeli (skat. «Modeļa izmantošana»)
python make_synth_dataset.py --out data/synth_30k --n 30000
python train.py --synth-data data/synth_30k --real-data data/real_dataset --steps 20000

# 2. pārbaudīt apmācītu modeli uz vienas reāla rokraksta rindas
python infer.py --checkpoint checkpoints/crnn_best.pt --image data/real_dataset/images/0001.png
```

Rezultāts tiek izdrukāts formā `Text: '...'`.

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

Izveido 20 paraugus mapē `samples/` (`sample_0.png` … `sample_19.png`) un izdrukā to tekstus. Noder vizuālai ģeneratora pārbaudei. Mape `samples/` ir `.gitignore`, tā netiek iesniegta Git.

### 2. Sintētiskās datu kopas sagatavošana (ieteicams)

```bash
# vienreiz uzģenerēt kopu uz diska (izmanto visus procesora kodolus)
python make_synth_dataset.py --out data/synth_30k --n 30000

# neobligāti: iepakot kopu vienā failā, ko apmācība lasa no RAM (pēc katras jaunas ģenerēšanas)
python packed_lines.py data/synth_30k
```

`make_synth_dataset.py` saglabā attēlus mapē `<out>/images/` un `<out>/labels.csv` ar kolonnām `filename,text,width`. Vienāds `--seed` dod vienādu kopu. `packed_lines.py` saglabā gatavus attēlus mapē `<kopa>/packed/`; ja `labels.csv`, `dataset.py` (arī komentāru izmaiņas), alfabēts vai `TARGET_HEIGHT` pēc iepakošanas mainās, `train.py` to pamana, lasa PNG failus un iesaka komandu pārpakošanai. Ģenerētās kopas (`data/synth_*/`) ir `.gitignore`.

### 3. Apmācība

```bash
# no gatavas sintētiskās kopas (500 piemēri automātiski tiek atdalīti pārbaudei)
python train.py --synth-data data/synth_30k --steps 20000

# sintētiskie + reālie dati (10 % reālo rindu paliek tikai pārbaudei)
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
| `--val-data` | nav | Atsevišķa sintētiskā pārbaudes kopa (neobligāti) |
| `--val-size` | `500` | Sintētisko pārbaudes piemēru skaits (atdala no `--synth-data`, ja `--val-data` nav norādīta; ģenerēšanas režīmā tiek uzģenerēti ar fiksētu `--seed`) |
| `--real-data` | nav | Mape ar reālajiem datiem; apmācības daļa tiek pievienota sintētiskajiem |
| `--real-val-fraction` | `0.1` | Reālo rindu daļa, kas **netiek izmantota apmācībā** un kalpo par godīgu pārbaudi (`0` = visas rindas apmācībai) |
| `--real-fraction` | nav | Tikai ģenerēšanai apmācības laikā: reālo rindu daļa batch (0..1) |
| `--synth-per-epoch` | `5000` | Tikai ģenerēšanai apmācības laikā: vajadzīgs reālo datu daļas aprēķinam |
| `--x-stride` | `2` jaunam modelim | Platuma saspiešana CNN: `2` vai `4` (ātrāk). Ar `--resume` pēc noklusējuma tiek ņemta čekpointa vērtība; atšķirīga vērtība ir kļūda |
| `--no-amp` | izslēgts | Izslēdz mixed precision (fp16) |
| `--num-workers` | atkarīgs no datora | Paralēlie datu sagatavotāji |
| `--log-every` | `500` | Cik bieži izdrukāt zudumus |
| `--checkpoint-every` | `1000` | Cik bieži saglabāt svarus un veikt pārbaudi |
| `--checkpoint-dir` | `checkpoints` | Kur saglabāt čekpointus |
| `--resume` | nav | Ielādē svarus no esoša čekpointa un turpina apmācību |
| `--seed` | `0` | Nejaušības sākumvērtība, lai palaišanu varētu atkārtot |
| `--early-stop` | `0` | Apstāties, ja pārbaude N reizes pēc kārtas neuzlabojas (`0` = izslēgts) |

Tabulā «nav» nozīmē, ka parametram nav noklusējuma vērtības.

**Pārbaude un labākais čekpoints.** Pie katras čekpointa saglabāšanas modelis tiek pārbaudīts uz piemēriem, uz kuriem tas nemācās, un tiek izdrukāti loss, CER un WER. Ja norādīta `--real-data`, ir divas pārbaudes kopas: `synth` (sintētiskā) un `real` (atdalītās reālās rindas); `crnn_best.pt` tiek izvēlēts pēc **reālās** kopas CER. Ja reālo datu nav, to nosaka sintētiskā kopa (tā ir optimistiska: modelis, kas lasa sintētiku ar CER 0.01, var lasīt reālu rokrakstu ar CER 0.6).

**Reālo datu sadalījums.** Sadalījumu nosaka tikai faila nosaukums (`split_real` failā `dataset.py`), tāpēc tas **nemainās, kad datu kopai pievieno jaunas rindas**, un `evaluate.py` izmanto tieši to pašu pārbaudes daļu. Faili, kuru nosaukums beidzas ar numuru, tiek grupēti blokos pa 50 (`REAL_SPLIT_BLOCK`) un bloks nonāk vienā daļā kopumā, jo blakus rindas parasti nāk no vienas lapas un viena rokraksta. Ja `--real-val-fraction` vai kopas izmērs ir tāds, ka pārbaudes daļa sanāk tukša, apmācībā tiek izmantotas visas rindas (par to tiek izdrukāts brīdinājums).

Piezīmes par `--resume`: tiek ielādēti tikai **modeļa svari** (nevis optimizētājs vai soļu skaitītājs), tāpēc soļu skaitītājs sākas no 0 un jauni čekpointi var pārrakstīt vecos, ja `--checkpoint-dir` nav mainīts. Ja čekpoints ir apmācīts ar citu alfabētu, `train.py`, `infer.py` un `evaluate.py` apstājas ar skaidru kļūdas ziņojumu.

### 4. Atpazīšana (inference)

```bash
python infer.py \
  --checkpoint checkpoints/crnn_best.pt \
  --image data/real_dataset/images/0001.png \
  --beam-width 10
```

`--beam-width` (noklusējums `5`) nosaka, cik labāko teksta variantu tiek saglabāti dekodēšanas laikā. Lielāka vērtība nozīmē vairāk pārbaudītu variantu un nedaudz lēnāku darbību. Parametra maiņa **neprasa modeļa pārtrenēšanu**.

Modelis paredzēts **vienai teksta rindai**, nevis visai lapai. Visas lapas sadalīšanu rindās veic web lietotne.

### 5. Kvalitātes novērtēšana (CER, WER)

```bash
python evaluate.py --checkpoint checkpoints/crnn_best.pt
python evaluate.py --checkpoint checkpoints/crnn_best.pt --real-data data/real_dataset --synthetic 500
python evaluate.py --checkpoint checkpoints/crnn_best.pt --beam-width 5   # papildus Beam Search (lēns)
```

Skripts izdrukā **CER** (Character Error Rate = labojumu skaits / pareizo rakstzīmju skaits) un **WER** (tas pats vārdu līmenī); mazāks ir labāk. Rezultāti tiek doti atsevišķi reālajām rindām un svaigi ģenerētiem sintētiskajiem paraugiem (`--synthetic`, noklusējums `200`, `0` = nepārbaudīt), kā arī dažiem piemēriem «correct / model».

- Noklusējumā reālajiem datiem tiek izmantota **tikai atdalītā pārbaudes daļa** (`--real-split val`). `--real-split train` mēra uz apmācības daļas, `--real-split all` uz visām rindām (rezultāts ir optimistisks, jo modelis daļu rindu ir redzējis).
- `--real-val-fraction` jābūt tādai pašai vērtībai kā `train.py` palaišanā (noklusējums abiem `0.1`).
- Pamata mērījums izmanto **greedy** dekodēšanu; ar `--beam-width N` papildus tiek izdrukāts arī Beam Search rezultāts.

### Alfabēta maiņa

Ja alfabētam (`alphabet.py`) pievieno vai noņem rakstzīmes, mainās klašu skaits un rakstzīmju numuri, tāpēc vecais čekpoints vairs neder: modelis jāapmāca no jauna. Kopš alfabēta paplašināšanas čekpointu pārnešanas rīka (`migrate_checkpoint.py`) vairs nav. Pēc alfabēta maiņas:

1. pārgenerēt un pārpakot sintētiskās kopas (`python make_synth_dataset.py ...`, `python packed_lines.py ...`);
2. apmācīt modeli no nulles (`python train.py ...`);
3. atjaunot rakstzīmju tabulu `web-app/server/alphabet_table.py` un nomainīt čekpointu serverī. Serverim ir sava kopija gan rakstzīmju tabulai, gan arhitektūrai (`htr_numpy.py`); abām jāsakrīt ar `alphabet.py` un `model.py`.

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
[T, B, NUM_CLASSES]   rakstzīmju varbūtības katrā solī (pašlaik 176)
   │
   ▼  CTC dekodēšana (Greedy vai Beam Search)
Teksts
```

| Daļa | Fails | Apraksts |
| --- | --- | --- |
| Alfabēts | `alphabet.py` | Rakstzīmju ↔ indeksu tabula, `encode`, `decode_greedy`, `decode_beam_search`, `strip_struck` |
| CNN | `model.py` → `CNNBackbone` | Konvolūcijas + MaxPool; platumu saspiež vienreiz (×2) vai divreiz (×4, `x_stride=4`), augstumu līdz 1 |
| BiLSTM + Linear | `model.py` → `BiLSTMHead` | 2 slāņi, divvirzienu; izvada rezultātu katrai rakstzīmei |
| Modelis | `model.py` → `CRNN` | Savieno abas daļas; `output_length(W)` = `max(W // 2, 1)` vai `max(W // 4, 1)` |
| Čekpointi | `model.py` → `save_checkpoint`, `load_checkpoint`, `load_model`, `check_num_classes` | Čekpointā tiek saglabāti svari, `x_stride` un soļa numurs; tiek nolasīts arī vecais formāts (tikai svari); alfabēta neatbilstība tiek konstatēta ar skaidru kļūdu |
| Datu ielāde | `dataset.py` | `SyntheticHTRDataset`, `LineFolderDataset`, `split_real`, `BucketBatchSampler`, `SyntheticBucketStream`, `collate_batch`, `image_to_tensor`, `open_line_image` |
| Metrikas | `metrics.py` | `edit_distance`, `cer`, `wer` |

Čekpointos, kas pašlaik atrodas repozitorijā, `x_stride = 4`. Jauna modeļa `x_stride` noklusējums ir `2` (precīzāk), `--x-stride 4` apmācību paātrina aptuveni divas reizes. `x_stride` tiek saglabāts čekpointā, un serveris abus režīmus nolasa no tā.

### Attēla sagatavošana

Gan apmācībā, gan atpazīšanā attēls tiek apstrādāts vienādi (`open_line_image` un `image_to_tensor` failā `dataset.py`):

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

- **Greedy** (`decode_greedy`): katrā solī ņem visvarbūtīgāko klasi, sakļauj atkārtojumus, noņem `blank`. Izmanto pārbaude apmācības laikā un `evaluate.py`.
- **CTC Prefix Beam Search** (`decode_beam_search`): saglabā `beam_width` labākos teksta prefiksus. Izmanto `infer.py` un serveris; `evaluate.py` to izmanto ar `--beam-width`.

### Apmācības cikls

`train.py` izmanto `Adam`, `CTCLoss(blank=0, zero_infinity=True)`, gradientu ierobežošanu (norma 5.0) un `PlateauScheduler` (ik pēc 200 soļiem pārbauda vidējo zudumu; ja tas neuzlabojas, `lr` tiek samazināts uz pusi). Uz CUDA tiek izmantota mixed precision (fp16) ar `GradScaler`. Pārbaudes un labākā čekpointa izvēle ir aprakstīta sadaļā [Apmācība](#3-apmācība); pārējie čekpointi tiek saglabāti kā `crnn_step<N>.pt`; ar `--early-stop N` apmācība apstājas, ja pārbaude N reizes pēc kārtas neuzlabojas.

---

## Dati

### Rakstzīmju kopa

`alphabet.py` definē rakstzīmju grupas un no tām izveido sakārtotu sarakstu `CHARS`; kopā ar `blank` (indekss 0) tas dod `NUM_CLASSES` klases (pašlaik **176**). Grupas:

- latviešu burti ar diakritiskajām zīmēm (`ā č ē ģ ī ķ ļ ņ š ū ž`), lielie un mazie;
- `q w x y` (un lielie);
- cipari `0` līdz `9`;
- reti burti `ö ü Ç Ć`;
- atstarpe un pamata pieturzīmes `. , ! ? - : ; ( ) " ' / •`;
- tipogrāfiskās zīmes: pēdiņas `„ “ ” ‚ ‘ ’ ‹ › « »`, domuzīmes `– — −`, daudzpunkte `…`, `·`, `′ ″`, `` ` ´ ``, `¡ ¿`;
- iekavas `[ ] { }` un tastatūras simboli `* _ + = < > @ # % & ^ ~ | \`;
- matemātikas un citi simboli `± × ÷ ≠ ≤ ≥ ≈ ∞ √ ° ‰ ← → ↑ ↓ ⇒`, augšraksti un daļskaitļi `² ³ ¹ ¼ ½ ¾`;
- `§ ¶ © ® ™ №` un valūtas `€ $ £ ¥ ¢`;
- speciālais simbols **`¤`** (`STRIKE_MARK`), kas norāda **pārsvītrotu vārdu**.

Funkcija `strip_struck(text)` noņem `¤` no gala teksta (`"Ekonomika ¤ tirgus"` → `"Ekonomika tirgus"`).

Ja teksts satur rakstzīmi ārpus šīs kopas, `encode` izmet `ValueError`; `LineFolderDataset` šādas rindas izlaiž un izdrukā brīdinājumu ar to sarakstu.

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

Pašlaik mapē `images/` ir **5677 rindu attēli** (`0001.png` līdz `5677.png`), katrs attēls ir viena rinda. Datu kopa tiek papildināta, tāpēc precīzs rindu skaits jāskata `labels.csv` (ierakstu skaits tur var atšķirties no failu skaita, ja daļa rindu vēl nav marķēta vai ir izlaistas).

`labels.csv` formāts (UTF-8):

```csv
filename,text
0001.png,Ekonomika izdala (3 vai 4) tirgus dalībniekus
0002.png,"1,3,5,7 - reala ekonomika,"
```

Teksts nedrīkst saturēt rakstzīmes ārpus alfabēta. Pārsvītrotu vārdu marķēšanai izmanto `¤`.

> Attēli, `labels.csv` un čekpointi repozitorijā tiek glabāti ar **Git LFS** (skat. `.gitattributes`). Pēc klonēšanas jāizpilda `git lfs pull`, citādi tie būs tikai ~130 baitu atsauces faili.

### Marķēšanas noteikumi

Reālo datu rindas tiek marķētas pēc šādiem principiem:

- teksts tiek pārrakstīts **tieši tā, kā tas uzrakstīts attēlā**: bez gramatikas un pareizrakstības labošanas un bez vārdu «pielāgošanas» pareiziem latviešu vārdiem; autora kļūdas paliek etiķetē;
- viens attēla fails = viena teksta rinda; faila nosaukums sastāv no četriem cipariem (`0042.png`), numerācija turpinās bez atkārtojumiem;
- tiek pārrakstīts tikai **audzēkņa rokraksts**: skolotāja sarkanie labojumi, atzīmes un punkti etiķetē netiek iekļauti;
- **pārsvītrotie vārdi** dažās partijās etiķetē tika **izlaisti** (nevis aizvietoti ar `¤`), bet rindas, kurās pārsvītrota lielākā daļa teksta, netika iekļautas vispār. Ja vēlaties, lai modelis mācītos atpazīt svītrojumus, šajās vietās jāliek `¤` (skat. [Rakstzīmju kopa](#rakstzīmju-kopa));
- vārdi, kurus attēlā nevar droši izlasīt, ir pārrakstīti pēc labākā minējuma; tos vēlams pārbaudīt ar zemāk aprakstīto rīku;
- etiķetē drīkst būt tikai alfabēta rakstzīmes. Pārbaudes rīks alfabētu **nepārbauda**, tāpēc pirms apmācības jāpārliecinās, ka nav brīdinājumu «skipped … lines with characters that are not in the alphabet».

### Marķējuma pārbaudes rīks

`verity_labels.py` ir neliela darbvirsmas programma (Python + Tkinter, tumšā tēma), kas ļauj rindu pa rindai pārbaudīt un labot `labels.csv`, vienlaikus redzot attēlu, un iesniegt labojumus Git.

```text
┌───────────────────────────┬──────────────────────────────────┐
│ 42 / 5677                 │                                  │
│ 0042.png                  │                                  │
│ ┌───────────────────────┐ │        rindas attēls             │
│ │ rindas teksts         │ │   (tālummaiņa, pārvietošana,     │
│ │ (var labot)           │ │    apgriešana)                   │
│ └───────────────────────┘ │                                  │
│ [Crop] [Original]         │                                  │
│ [Deselect] [Delete]       │                                  │
│ [Commit & Push to Git]    │                                  │
└───────────────────────────┴──────────────────────────────────┘
```

**Palaišana** (mapē `latvian_htr/`):

```bash
pip install pillow opencv-python-headless
python verity_labels.py
```

Tkinter ir Python standarta komplektā (Windows). Ceļi tiek noteikti attiecībā pret faila atrašanās vietu: rīks lasa `data/real_dataset/labels.csv` un attēlus no `data/real_dataset/images/`, tātad darbojas jebkurā datorā bez ceļu maiņas. Logā ir arī nelieli GIF / video lauki, kuru faili tiek ņemti no mapes `latvian_htr/prikol/` (tāpēc vajadzīgs `opencv`). Git komandām rīks meklē `git` sistēmas `PATH` un standarta Windows ceļos (`C:\Program Files\Git\cmd\git.exe` u. c.).

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
| Dzēst rindu **un attēla failu** | poga «Delete» (ar apstiprinājumu) |
| Saglabāt, izveidot commit un nosūtīt (push) uz Git | poga «Commit & Push to Git» |

**Attēla apgriešana.** Attēlā ar nospiestu kreiso peles pogu jāiezīmē zona (dzeltens raustīts rāmis) un jānospiež `Ctrl+R`. Apgrieztais attēls **uzreiz pārraksta failu** mapē `images/`; pirms pirmās apgriešanas oriģināls tiek nokopēts mapē `images/_originals/`. Pie faila nosaukuma parādās atzīme «[apgriezts]». Atgriežot oriģinālu (`Ctrl+B`), kopija no `_originals/` tiek atjaunota un dzēsta.

**Saglabāšana.**

- Attēla apgriešana tiek saglabāta uz diska **nekavējoties**.
- Teksta labojumi tiek ierakstīti `labels.csv` pie `Enter`, `Ctrl+S` un loga aizvēršanas. Pārvietojoties ar `Alt+←/→` vai `PageUp/PageDown`, labojums tiek atcerēts programmā, bet failā parādās pie nākamā `Enter` / `Ctrl+S` / aizvēršanas.
- Pirms pirmās saglabāšanas tiek izveidota rezerves kopija `labels.csv.bak`.
- Pozīcija tiek atcerēta failā `images/.verify_progress`, loga iestatījumi failā `images/.verify_settings.json`; nākamreiz programma turpina no tās pašas rindas.
- Poga «Commit & Push to Git» vispirms saglabā `labels.csv`, tad izpilda `git add` mainītajiem failiem (`labels.csv` un `images/`, bez `_originals/`, `.verify_progress` un `.verify_settings.json`) un `git commit` ar ziņojumu `real data 0.<N> checked till <fails>`, kur `<N>` ir par 1 lielāks par lielāko līdz šim izmantoto versiju vēsturē. Pēc tam tiek izpildīts `git push` (ja tas neizdodas, tiek mēģināts `git push -u origin <zars>`); ja push neizdodas, commit paliek lokāli un rīks parāda kļūdu.

`labels.csv.bak`, `.verify_progress`, `.verify_settings.json` un `_originals/` ir lokāli darba faili, un tie ir pievienoti `.gitignore`.

> Ja pēc apgriešanas atkārtoti palaižat attēlu sagriešanas skriptu, tas var pārrakstīt apgrieztos failus; oriģināli paliek `_originals/`.

### Sintētiskie dati

`synthetic_data.py` katru reizi ģenerē jaunu piemēru:

1. izvēlas fontu no `fonts/` un tekstu, ko šis fonts kopā ar **rezerves fontu** spēj uzzīmēt. Ja rindas fontam nav glifa kādai rakstzīmei (piemēram, `ö`, `§`, `€`), šī rakstzīme tiek uzzīmēta ar rezerves fontu, kura izmērs tiek pielāgots rindas fonta mazo burtu augstumam; rinda šādā gadījumā vienmēr tiek zīmēta «burtu pa burtam». Rakstzīme tiek izslēgta tikai tad, ja tās nav **nevienā** no abiem fontiem (`font_charset`, `drawable_charset`);
2. salasa nejaušu «rindu» no `data/corpus.txt` vārdiem (ap 7 % gadījumu skaitļi, ap 3 % simbolu izteiksmes, piemēram, `§ 12`, `25°C`, `x ≥ 3`, `5 × 4`, `½`, retāk romiešu cipari);
3. 80 % rindu zīmē **burtu pa burtam** ar nelielām izmēra, augstuma un slīpuma svārstībām; puse no tām ar savienotiem burtiem (kursīvs);
4. 20 % rindu satur pārsvītrotus vārdus (līnija, murgojums, krusti, vairākas līnijas). 60 % svītrojumu ir «kļūdas» (pārrakstīšanās, trūkstoša vai lieka diakritiskā zīme, sajaukti burti), virs kurām mazākiem burtiem uzrakstīts pareizais vārds; etiķetē ir pareizais vārds. Svītrojumi bez labojuma etiķetē tiek aizvietoti ar `¤`;
5. pievieno burtnīcas lapas foto imitāciju: rūtiņu vai līniju papīru, nevienmērīgu apgaismojumu, viļņainu rindu, troksni, izplūdumu un JPEG artefaktus (pēc samazināšanas līdz 64 px).

**Rezerves fonta izvēle** (`default_font()`), kārtībā:

1. fails `fonts/default.ttf` vai `fonts/default.otf`, ja tāds ir (ieteicams: tad dati izskatās vienādi uz katra datora; piemērots, piemēram, DejaVu Sans, kas aptver visas alfabēta rakstzīmes). Šis fonts netiek izmantots kā parasts rokraksta fonts;
2. sistēmas fonts ar vislielāko rakstzīmju pārklājumu (Linux: DejaVu Sans; Windows: Arial / Segoe UI; macOS: Arial);
3. fonts no `fonts/` ar vislielāko pārklājumu.

Rezerves fonts tiek meklēts tikai pirmās vajadzības brīdī, tāpēc `synthetic_data.py` imports vairs nekrīt, ja sistēmā nav piemērota fonta. Kļūda tiek izmesta tikai tad, ja nav neviena fonta vispār.

Mapē `fonts/` pašlaik esošie fonti neaptver visas jaunās rakstzīmes (piemēram, bultiņas `← → ↑ ↓ ⇒` nav nevienā no tiem). Šīs rakstzīmes parādās sintētiskajos datos tikai tad, ja tās ir rezerves fontā; ar DejaVu Sans tās ir.

Sintētiskie dati ir paredzēti, lai modelis varētu mācīties, kamēr trūkst reālā rokraksta. **Tie neaizvieto īstu cilvēku rokrakstu.**

---

## Web lietotne

React + Vite + TypeScript lietotne un Flask serveris mapē `web-app/`. Visi lietotāja teksti ir latviešu valodā (`src/i18n/lv.ts`, garie teksti `TermsDialog.tsx` un `PrivacyNotice.tsx`); izstrādātāju dokumentācija ir angļu valodā failā [`web-app/README.md`](web-app/README.md).

**Lietotāja ceļš:** pirmajā apmeklējumā «Lietošanas noteikumi» (piekrišana AI apmācībai ir brīvprātīga) → izvēlēties vienu vai vairākus attēlus vai uzņemt foto → priekšskatījums (var pagriezt, noņemt, pievienot vēl) → **Atpazīt tekstu** → progress ar atcelšanu → rezultāts, ko var labot (vairākiem foto katram sava cilne) → **Kopēt tekstu** / **Kopēt visu**.

Svarīgākais:

- attēls pārlūkā tiek pagriezts un samazināts līdz 1920 × 1080 (portretā 1080 × 1920);
- pārlūks sadala foto **rindās** (`src/image/lineSegmentation.ts`): izlīdzina apgaismojumu, atrod tintes formas, izmet malu objektus (galds, blakus lapa, pirksti), izlīdzina slīpumu līdz 12°, sagrupē formas rindās un katrā rindas attēlā atstāj tikai šīs rindas tinti. Pilnais foto netiek sūtīts;
- serverim tiek nosūtīti rindu PNG faili `0001.png`, `0002.png`, ..., `labels.csv` (kolonna `text` tukša) un lietotāja piekrišanas izvēle; serveris aizpilda `text` un atgriež CSV;
- serveris (`server/app.py`) palaiž modeli bez PyTorch (`server/htr_numpy.py`), jo PyTorch neietilpst bezmaksas PythonAnywhere diskā. Rakstzīmju tabula (`server/alphabet_table.py`) ir roku darba kopija no `alphabet.py`; tā tiek izvēlēta pēc klašu skaita čekpointā (pašlaik 137 klases, `crnn_best_migrated.pt`). **Pēc alfabēta maiņas un pārtrenēšanas tā jāatjauno** (skat. [Alfabēta maiņa](#alfabēta-maiņa));
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
│   ├── alphabet.py               # rakstzīmju kopa, kodēšana, dekodēšana (bez PyTorch)
│   ├── metrics.py                # CER, WER (bez PyTorch)
│   ├── model.py                  # CRNN, čekpointu saglabāšana un ielāde
│   ├── dataset.py                # datu kopas, reālo datu sadalīšana, batch no līdzīga platuma rindām
│   ├── synthetic_data.py         # sintētisko attēlu ģenerators
│   ├── make_synth_dataset.py     # sintētiskās kopas ģenerēšana uz diska
│   ├── packed_lines.py           # kopas iepakošana lasīšanai no RAM
│   ├── train.py                  # apmācība
│   ├── infer.py                  # viena attēla atpazīšana
│   ├── evaluate.py               # CER / WER novērtēšana
│   ├── verity_labels.py          # marķējuma pārbaudes rīks (GUI, apgriešana, Git commit + push)
│   ├── requirements.txt
│   ├── fonts/                    # fonti sintētiskajiem datiem (+ neobligāts default.ttf)
│   ├── prikol/                   # attēli, GIF un video pārbaudes rīka laukiem
│   ├── checkpoints_v2/           # crnn_best_migrated.pt (137 klases; pašlaik to izmanto web serveris), ≈ 35 MB
│   └── data/
│       ├── corpus.txt            # latviešu teksts sintētisko datu vārdiem
│       └── real_dataset/         # reālie dati (labels.csv, images/)
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

Lokāli darba faili, kas netiek iesniegti Git (skat. `.gitignore`): `latvian_htr/checkpoints/` (apmācības rezultāti), `latvian_htr/samples/`, `latvian_htr/data/synth_*/`, `latvian_htr/data/**/packed/`, kā arī marķējuma rīka `labels.csv.bak`, `images/.verify_progress`, `images/.verify_settings.json` un `images/_originals/`.

---

## Zināmie ierobežojumi

- **Atpazīšanas kvalitāte uz reāla rokraksta** vēl ir zema (skat. [Vēsturiskie testēšanas rezultāti](#vēsturiskie-testēšanas-rezultāti)); lietotne lietotājam to norāda.
- **Modelis jāpārtrenē:** alfabēts ir paplašināts, tāpēc `crnn_best_migrated.pt` ar pašreizējo `alphabet.py` nav ielādējams. Jaunās rakstzīmes modelis iemācīsies tikai pēc apmācības ar jauno sintētisko kopu.
- **Maz reālu datu:** ap 5700 rindu joprojām ir maz, lai modelis labi vispārinātu uz dažādiem rokrakstiem. Sintētiskie dati nav īsts rokraksts. Atdalot 10 % pārbaudei, apmācībai paliek vēl mazāk.
- **Reālās pārbaudes kopas ierobežojumi:** sadalījums notiek pa failu numuru blokiem, jo marķējumā nav informācijas par autoru vai lapu. Ja viena un tā pati persona ir uzrakstījusi rindas, kas nonāk gan apmācībā, gan pārbaudē, rezultāts var būt nedaudz optimistisks.
- **Rindu sadalīšana pārlūkā** darbojas ar parastām, arī slīpām (līdz 12°) un cieši izvietotām rindām, bet stipri izliektas rindas vai rindas, kuru burti pieskaras viena otrai lielā garumā, var tikt sadalītas neprecīzi. Teksts no blakus lapas, kas pilnībā redzams foto (nepieskaras foto malai), var nonākt rindu attēlos.
- **CTC garuma ierobežojums:** izvades garums `T = W / 2` (ar `x_stride=4`: `W / 4`) nedrīkst būt mazāks par teksta garumu; pārāk šauri attēli ar garu tekstu netiks pareizi apmācīti (`train.py` par to brīdina).
- **`--resume`** atjauno tikai svarus, nevis pilnu apmācības stāvokli.
- **Fontu licences:** pirms projekta publiskas izplatīšanas jāpārbauda mapē `fonts/` esošo fontu licences (arī `default.ttf`, ja to pievienojat).
- **Pārbaudes rīks (`verity_labels.py`):** nepārbauda, vai etiķete satur tikai alfabēta rakstzīmes; apgriešana pārraksta attēlu failus (oriģināli paliek `_originals/`); projektā nav automātisku testu šim rīkam.
- **Pārsvītrotie vārdi:** dažās marķēto datu partijās tie ir izlaisti, nevis atzīmēti ar `¤`, tāpēc modelis no šīm rindām neiemācās atpazīt svītrojumus.
- **Automātisku testu nav:** koda izmaiņas pašlaik nav pārbaudāmas ar testiem; pēc izmaiņām `train.py` un `evaluate.py` jāpalaiž ar nelielu kopu.

---

## Ceļvedis

1. Apmācīt modeli no jauna ar paplašināto alfabētu (sintētika + reālie dati), pārbaudīt rezultātu uz atdalītās reālās kopas.
2. Atjaunot web serveri jaunajam modelim: `alphabet_table.py`, čekpoints, pēc vajadzības `htr_numpy.py`.
3. Savākt un marķēt vēl lielāku, daudzveidīgāku reālā rokraksta datu kopu (dažādi cilvēki, pildspalvas, papīrs, foto kvalitāte), tostarp no lietotājiem, kas piekrita AI apmācībai; visu jauno marķējumu pirms apmācības pārbaudīt ar `verity_labels.py`. Ja iespējams, marķējumā saglabāt informāciju par autoru, lai pārbaudes kopu varētu atdalīt pa personām.
4. Papildināt augmentāciju reālajiem datiem (papīra tekstūra, ēnas, perspektīva, apgaismojums).
5. Saglabāt čekpointā arī optimizētāja un soļu stāvokli, lai apmācību varētu pilnībā atsākt.
6. Ticamības vērtības katram vārdam, lai lietotnē varētu atkal ieslēgt nedrošo vārdu izcelšanu.

---

## Papildu dokumentācija

- [`web-app/README.md`](web-app/README.md): web lietotnes un servera uzbūve, izstrādātāja panelis, izvietošana PythonAnywhere (angļu val.).
- [`Proekta-skaidrošana.md`](Proekta-skaidrošana.md): padziļināts modeļa un sistēmas skaidrojums (krievu val.). **Šis dokuments vēl nav atjaunināts** pēc alfabēta paplašināšanas un citām izmaiņām, tajā var būt novecojusi informācija.

---

## Komanda un licence

Projektu izstrādā komanda **Slepie**. Licence pašlaik nav norādīta.

## Vēsturiskie testēšanas rezultāti

Zemāk redzams iepriekšējā modeļa (`crnn_best_migrated.pt`, 137 klases) mērījums ar `evaluate.py`, kas bija pieejams pirms atdalītās reālās pārbaudes kopas ieviešanas. **Tie nav salīdzināmi ar jauno modeli** un tiek saglabāti tikai kā atskaites punkts.

| Kopa | Paraugu skaits | CER |
| --- | --- | --- |
| Reālie paraugi (visa reālā kopa, tobrīd 3007 rindas) | 3007 | 0.622 (precīzi sakrīt 2 no 3007) |
| Sintētiskie (svaigi) | 400 | 0.010 (precīzi sakrīt 325 no 400) |

Atšķirība starp sintētiku un reālu rokrakstu parāda, ka modelis labi lasa ģeneratora rakstu, bet reālo rokrakstu vēl nolasa slikti. Pēc pārtrenēšanas rezultāti jāpārmēra ar `python evaluate.py --checkpoint ... --real-split val`.

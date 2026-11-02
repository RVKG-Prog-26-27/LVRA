# LVRA — Latviešu valodas rokraksta atpazīšana

**LVRA** ir eksperimentāla sistēma, kas no rokrakstā rakstīta latviešu teksta attēla iegūst rediģējamu digitālu tekstu. Projektu veido divas daļas:

| Daļa | Mape | Tehnoloģijas | Uzdevums |
| --- | --- | --- | --- |
| Atpazīšanas modelis | `latvian_htr/` | Python, PyTorch | Apmācīt un palaist CRNN + CTC modeli, kas nolasa vienu teksta rindu no attēla |
| Web lietotne | `web-app/` | React 18, TypeScript, Vite | Telefonam pielāgota saskarne latviešu valodā: foto → teksts → labošana → kopēšana |

> **Statuss: alpha.** Python modelis un web lietotne darbojas katrs atsevišķi, bet **web lietotne vēl nav savienota ar reālo modeli** — tā izmanto `mock` servisu, kas vienmēr atgriež vienu un to pašu demonstrācijas tekstu. Atpazīšanas kvalitāte uz reāla rokraksta vēl nav pietiekama gala produktam (skat. [Zināmie ierobežojumi](#zināmie-ierobežojumi)).

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

---

## Statuss

| Komponente | Statuss |
| --- | --- |
| Latviešu rakstzīmju kopa (130 klases, ieskaitot CTC `blank`) | ✅ |
| Sintētisko datu ģenerators (rokraksta imitācija, svītrojumi, papīra fons) | ✅ |
| Reālo datu ielāde (`RealHTRDataset`) | ✅ |
| CRNN modelis (CNN + 2×BiLSTM + Linear) | ✅ |
| Mainīga attēla platuma atbalsts (individuāli `input_lengths`) | ✅ |
| Apmācība ar CTC, Adam, `ReduceLROnPlateau`, gradientu ierobežošana | ✅ |
| Viena attēla atpazīšana (`infer.py`) ar CTC Prefix Beam Search | ✅ |
| Novērtēšana ar CER (`evaluate.py`, greedy dekodēšana) | ✅ |
| Web lietotnes saskarne latviešu valodā | ✅ |
| WER, kā arī Greedy un Beam Search salīdzinājums | ❌ |
| Web lietotnes savienojums ar modeli (backend API) | ❌ |
| Pietiekami liela reāla rokraksta datu kopa | ❌ |
| Optimizētāja stāvokļa saglabāšana (pilna apmācības atsākšana) | ❌ |

---

## Ātrais starts

### Prasības

- **Python 3.10 vai jaunāks** (kodā tiek lietota sintakse `list[str] | None`)
- Python bibliotēkas: `torch>=2.2`, `numpy`, `pillow` (skat. `latvian_htr/requirements.txt`)
- **Node.js (LTS)** — tikai web lietotnei
- NVIDIA GPU nav obligāts; ja CUDA ir pieejama, tā tiek izmantota automātiski, citādi apmācība notiek uz CPU

### Python daļa

```bash
cd latvian_htr
pip install -r requirements.txt

# Pārbaudīt jau apmācītu modeli uz vienas reāla rokraksta rindas
python infer.py \
  --checkpoint checkpoints_v2/crnn_step30000.pt \
  --image data/real_dataset/images/0001.png
```

Rezultāts tiek izdrukāts formā `Teksts: '...'`.

### Web lietotne

```bash
cd web-app
npm install
npm run dev
```

Atver `http://localhost:5173`. Lai pārbaudītu uz telefona (dators un telefons vienā Wi-Fi tīklā): `npm run dev:phone` un atver terminālī parādīto `Network` adresi.

---

## Modeļa izmantošana

Visas komandas tiek izpildītas mapē `latvian_htr/`.

### 1. Sintētisko paraugu ģenerēšana

```bash
python synthetic_data.py
```

Izveido 20 paraugus mapē `samples/` (`sample_0.png` … `sample_19.png`) un izdrukā to tekstus. Noder vizuālai ģeneratora pārbaudei.

### 2. Apmācība

```bash
# tikai sintētiskie dati
python train.py --steps 2000 --batch-size 32

# sintētiskie + reālie dati
python train.py --steps 20000 --real-data data/real_dataset
```

| Parametrs | Noklusējums | Nozīme |
| --- | --- | --- |
| `--steps` | `20000` | Apmācības soļu skaits (1 solis = 1 batch) |
| `--batch-size` | `64` | Piemēru skaits vienā solī |
| `--lr` | `1e-3` | Sākotnējais mācīšanās ātrums |
| `--synth-per-epoch` | `5000` | Sintētisko piemēru skaits vienā «epohā» |
| `--real-data` | — | Ceļš uz reālo datu mapi; ja norādīts, tiek pievienoti sintētiskajiem |
| `--num-workers` | `2` | Paralēlie datu sagatavotāji |
| `--log-every` | `500` | Cik bieži izdrukāt zudumus |
| `--checkpoint-every` | `1000` | Cik bieži saglabāt svarus |
| `--checkpoint-dir` | `checkpoints` | Kur saglabāt `crnn_step<N>.pt` |
| `--resume` | — | Ielādē svarus no esoša checkpoint un turpina apmācību |

Piezīmes par `--resume`: tiek ielādēti tikai **modeļa svari** (nevis optimizētājs vai soļu skaitītājs), tāpēc soļu skaitītājs sākas no 0 un jauni checkpoint faili var pārrakstīt vecos, ja `--checkpoint-dir` nav mainīts.

### 3. Atpazīšana (inference)

```bash
python infer.py \
  --checkpoint checkpoints_v2/crnn_step30000.pt \
  --image data/real_dataset/images/0001.png \
  --beam-width 10
```

`--beam-width` (noklusējums `5`) nosaka, cik labāko teksta variantu tiek saglabāti dekodēšanas laikā. Lielāka vērtība nozīmē vairāk pārbaudītu variantu un nedaudz lēnāku darbību. Parametra maiņa **neprasa modeļa pārtrenēšanu**.

Modelis paredzēts **vienai teksta rindai** — nevis visai lapai.

### 4. Kvalitātes novērtēšana (CER)

```bash
python evaluate.py --checkpoint checkpoints_v2/crnn_step30000.pt
python evaluate.py --checkpoint checkpoints_v2/crnn_step30000.pt --synthetic 500
```

Skripts izdrukā **CER** (Character Error Rate = labojumu skaits / pareizo rakstzīmju skaits; mazāks ir labāk) atsevišķi reālajiem un svaigi ģenerētajiem sintētiskajiem paraugiem, kā arī dažus piemērus «pareizi / modelis».

Svarīgi:
- Novērtēšana izmanto **greedy** dekodēšanu, nevis Beam Search.
- Ja reālie dati tika iekļauti apmācībā (`--real-data`), CER uz tiem ir **optimistisks**. Godīgam rezultātam reālos paraugus vēlams atdalīt no apmācības.

---

## Kā modelis darbojas

Izmantota **CRNN** arhitektūra (konvolucionāls tīkls + rekurents tīkls) ar **CTC** zudumu funkciju. CNN un BiLSTM ir savienoti **secīgi**: CNN izvade ir BiLSTM ievade.

```text
Attēls [B, 1, 64, W]
   │
   ▼  CNN — vizuālo pazīmju iegūšana, augstums 64 → 1
[B, 512, 1, W/2]
   │
   ▼  attēla platums kļūst par laika asi T = W / 2
[T, B, 512]
   │
   ▼  2 × BiLSTM (hidden = 256) — konteksts abos virzienos
[T, B, 512]
   │
   ▼  Linear → log_softmax
[T, B, 130]   rakstzīmju varbūtības katrā solī
   │
   ▼  CTC dekodēšana (Beam Search)
Teksts
```

| Daļa | Fails | Apraksts |
| --- | --- | --- |
| Alfabēts | `alphabet.py` | Rakstzīmju ↔ indeksu tabula, `encode`, `decode_greedy`, `decode_beam_search`, `strip_struck` |
| CNN | `model.py` → `CNNBackbone` | Konvolūcijas + MaxPool; platumu saspiež tikai vienreiz (×2), augstumu — līdz 1 |
| BiLSTM + Linear | `model.py` → `BiLSTMHead` | 2 slāņi, divvirzienu; izvada rezultātu katrai rakstzīmei |
| Modelis | `model.py` → `CRNN` | Savieno abas daļas; `output_length(W)` = `max(W // 2, 1)` |
| Datu ielāde | `dataset.py` | `SyntheticHTRDataset`, `RealHTRDataset`, `collate_batch`, `image_to_tensor` |

### Attēla sagatavošana

Gan apmācībā, gan atpazīšanā attēls tiek apstrādāts vienādi:

1. pārveidošana pelēktoņos;
2. mērogošana līdz **64 px augstumam** (proporcijas saglabātas);
3. automātiskais kontrasts (`autocontrast`), lai pelēks vai tumšs fons kļūtu gaišs;
4. krāsu apgriešana (fons = 0, tinte ≈ 1);
5. tensors `[1, 64, W]`.

### Mainīga platuma attēli

Vienā batch attēli tiek papildināti ar nullēm līdz platākā attēla platumam. Lai CTC neuztvertu šo papildinājumu kā saturu, `train.py` katram piemēram aprēķina individuālu `input_length = model.output_length(W)`.

> **Ja maināt CNN** (pooling vai kernel izmērus), jāpielāgo arī `CRNN.output_length`, citādi CTC saņems nepareizus garumus.

### CTC un dekodēšana

CTC ļauj apmācīt modeli, nenorādot, kur attēlā atrodas katrs burts. Modelis katrā laika solī izvēlas rakstzīmi vai `blank`.

- **Greedy** (`decode_greedy`): katrā solī ņem visvarbūtīgāko klasi, sakļauj atkārtojumus, noņem `blank`.
- **CTC Prefix Beam Search** (`decode_beam_search`): saglabā `beam_width` labākos teksta prefiksus. Izmanto `infer.py`.

### Apmācības cikls

`train.py` izmanto `Adam`, `CTCLoss(blank=0, zero_infinity=True)`, gradientu ierobežošanu (norma 5.0) un `PlateauScheduler` (ik pēc 200 soļiem pārbauda vidējo zudumu; ja tas neuzlabojas, `lr` tiek samazināts uz pusi). Svari tiek saglabāti kā `crnn_step<N>.pt`.

---

## Dati

### Rakstzīmju kopa

`alphabet.py` satur 129 rakstzīmes + `blank` (indekss 0), kopā **130 klases**:

- latviešu burti ar diakritiskajām zīmēm (`ā č ē ģ ī ķ ļ ņ š ū ž`), lielie un mazie;
- `q w x y` (un lielie);
- cipari `0–9`;
- atstarpe, pieturzīmes, pēdiņas (arī `„ “ ” « »`), domuzīmes, iekavas, simboli, `€ $ № …`;
- speciālais simbols **`¤`** (`STRIKE_MARK`) — norāda **pārsvītrotu vārdu**.

Funkcija `strip_struck(text)` noņem `¤` no gala teksta (`"Ekonomika ¤ tirgus"` → `"Ekonomika tirgus"`).

Ja teksts satur rakstzīmi ārpus šīs kopas, `encode` izmet `ValueError`.

### Reālie dati

Atrodas `latvian_htr/data/real_dataset/`:

```text
data/real_dataset/
├── labels.csv          # kolonnas: filename,text
├── corpus.txt          # latviešu teksts sintētisko datu vārdu avotam
└── images/
    ├── 0001.png
    └── ...
```

Pašlaik: **143 marķētas teksta rindas** (kopā ≈ 5700 rakstzīmju), katrs attēls ir viena rinda.

`labels.csv` formāts (UTF-8):

```csv
filename,text
0001.png,Ekonomika izdala (3 vai 4) tirgus dalībniekus
0002.png,"1,3,5,7 - reala ekonomika"
```

Teksts nedrīkst saturēt rakstzīmes ārpus alfabēta. Pārsvītrotu vārdu marķēšanai izmanto `¤`.

### Sintētiskie dati

`synthetic_data.py` katru reizi ģenerē jaunu piemēru:

1. izvēlas fontu no `fonts/` (ap 20 fontu) un tekstu, ko šis fonts **pilnībā** spēj uzzīmēt (`font_charset` izslēdz rakstzīmes, kurām fontā nav glifa);
2. salasa nejaušu «rindu» no vārdiem (ap 7 % gadījumu — skaitļi);
3. 80 % rindu zīmē **burtu pa burtam** ar nelielām izmēra, augstuma un slīpuma svārstībām; puse no tām ar savienotiem burtiem (kursīvs);
4. 20 % rindu satur pārsvītrotus vārdus (4 veidi: līnija, murgojums, krusti, vairākas līnijas), kas etiķetē tiek aizvietoti ar `¤`;
5. pievieno burtnīcas lapas foto imitāciju: rūtiņu papīru, nevienmērīgu apgaismojumu, viļņainu rindu, troksni un izplūdumu.

Sintētiskie dati ir paredzēti, lai modelis varētu mācīties, kamēr trūkst reālā rokraksta. **Tie neaizvieto īstu cilvēku rokrakstu.**

---

## Web lietotne

React + Vite + TypeScript lietotne mapē `web-app/`. Visi lietotāja teksti ir latviešu valodā (`src/i18n/lv.ts`); izstrādātāju dokumentācija ir angļu valodā failā [`web-app/README.md`](web-app/README.md).

**Lietotāja ceļš:** izvēlēties attēlu vai uzņemt foto → priekšskatījums (var pagriezt) → **Atpazīt tekstu** → progress ar atcelšanu → rezultāts, ko var labot → **Kopēt tekstu**.

Svarīgākais:

- attēls pārlūkā tiek pagriezts, samazināts līdz 1920 × 1080 (portretā 1080 × 1920) un pārvērsts par JPEG;
- atpazīšanas laika limits — 60 s;
- vārdi ar ticamību zem 70 % tiek izcelti; izcēlums pazūd, kad vārds tiek labots;
- pēc veiksmīgas atpazīšanas attēls tiek dzēsts no atmiņas, paliek tikai teksts;
- izstrādātāja panelis: pievieno adresei `?dev=1`.

### Kā pieslēgt reālo modeli

Saskarne ar atpazīšanu ir atdalīta ar interfeisu `HandwritingRecognitionService` (`src/recognition/types.ts`): metode `recognize(image: Blob, { signal })` atgriež `{ text, words: [{ text, confidence }] }`. Lai savienotu ar Python modeli:

1. izveidot backend API (piem., Python serveris, kas pieņem JPEG, palaiž `infer.py` loģiku un atgriež JSON);
2. uzrakstīt jaunu klasi, kas implementē šo interfeisu un izsauc API;
3. izvēlēties to failā `src/recognition/createRecognitionService.ts` (šobrīd tur atgriezts `MockRecognitionService`).

Kļūdām jāizmanto kodi no `src/recognition/errors.ts`.

> Esošais modelis atgriež tekstu bez ticamības vērtībām katram vārdam; šīs vērtības (`confidence`) backend pusē vēl jāimplementē.

---

## Projekta struktūra

```text
LVRA/
├── README.md
├── Proekta-skaidrošana.md        # detalizēts modeļa skaidrojums (krievu val., daļēji novecojis)
│
├── latvian_htr/                  # Python daļa
│   ├── alphabet.py               # rakstzīmju kopa, kodēšana, dekodēšana
│   ├── model.py                  # CRNN
│   ├── dataset.py                # datu kopas un batch veidošana
│   ├── synthetic_data.py         # sintētisko attēlu ģenerators
│   ├── train.py                  # apmācība
│   ├── infer.py                  # viena attēla atpazīšana
│   ├── evaluate.py               # CER novērtēšana
│   ├── requirements.txt
│   ├── fonts/                    # fonti sintētiskajiem datiem
│   ├── checkpoints_v2/           # apmācītā modeļa svari (crnn_step30000.pt, ≈ 35 MB)
│   └── data/real_dataset/        # reālie dati (labels.csv, images/, corpus.txt)
│
├── web-app/                      # React/Vite lietotne
│   ├── src/
│   │   ├── app/                  # ekrānu stāvoklis
│   │   ├── components/           # UI komponentes
│   │   ├── image/                # attēla apstrāde pārlūkā
│   │   ├── recognition/          # atpazīšanas interfeiss, mock serviss, kļūdas
│   │   ├── editor/, utils/       # rezultāta labošana un kopēšana
│   │   ├── i18n/lv.ts            # visi lietotāja teksti
│   │   └── config/config.ts      # fiksētās vērtības (limiti, sliekšņi)
│   ├── public/
│   └── package.json
│
└── web-app_easter-egg/           # web-app variants ar paslēptu «easter egg»; darbam nav nepieciešams
```

---

## Zināmie ierobežojumi

- **Web lietotne neizmanto reālo modeli** — rezultāts ir fiksēts demonstrācijas teksts.
- **Maz reālu datu:** 143 rindas ir ļoti maz, lai modelis labi vispārinātu uz dažādiem rokrakstiem. Sintētiskie dati nav īsts rokraksts.
- **Korpusa ceļš:** `synthetic_data.load_words()` meklē vārdus failā `latvian_htr/data/corpus.txt`, bet korpuss šobrīd atrodas `latvian_htr/data/real_dataset/corpus.txt`. Ja pirmā faila nav, tiek izmantots neliels iebūvēts vārdu saraksts — sintētiskie dati kļūst mazāk daudzveidīgi. Risinājums: pārvietot vai nokopēt failu uz `latvian_htr/data/corpus.txt`, vai nomainīt ceļu kodā.
- **Viena rinda uz attēlu:** nav automātiskas teksta zonu vai rindu atdalīšanas pilnai lapai.
- **CTC garuma ierobežojums:** izvades garums `T = W / 2` nedrīkst būt mazāks par teksta garumu; pārāk šauri attēli ar garu tekstu netiks pareizi apmācīti (`train.py` par to brīdina).
- **`--resume`** atjauno tikai svarus, nevis pilnu apmācības stāvokli.
- **Novērtēšana** šobrīd tikai CER ar greedy dekodēšanu; WER un Beam Search novērtējuma nav.
- **Fontu licences:** pirms projekta publiskas izplatīšanas jāpārbauda mapē `fonts/` esošo fontu licences.
- `web-app/package.json` satur tikai skriptus `dev` un `dev:phone` (nav `build`).

---

## Ceļvedis

1. Savākt un marķēt lielāku, daudzveidīgāku reālā rokraksta datu kopu (dažādi cilvēki, pildspalvas, papīrs, foto kvalitāte).
2. Atdalīt validācijas kopu no apmācības un regulāri mērīt CER.
3. Pievienot WER un salīdzināt Greedy ar Beam Search.
4. Papildināt augmentāciju reālajiem datiem (papīra tekstūra, ēnas, perspektīva, apgaismojums).
5. Izveidot backend API un savienot web lietotni ar modeli.
6. Saglabāt checkpoint ar optimizētāja un soļu stāvokli, lai apmācību varētu pilnībā atsākt.
7. Nākotnē — rindu atdalīšana, lai atpazītu visu lapu.

---

## Papildu dokumentācija

- [`web-app/README.md`](web-app/README.md) — web lietotnes uzbūve, izstrādātāja panelis, ierobežojumi (angļu val.).
- [`Proekta-skaidrošana.md`](Proekta-skaidrošana.md) — padziļināts modeļa skaidrojums (krievu val.). **Uzmanību:** tas apraksta agrāku versiju (attēla augstums 32 px, vienāds `T` visam batch, tikai greedy dekodēšana), tāpēc daļa detaļu vairs neatbilst kodam; autoritatīvs avots ir pats kods un šis README.

---

## Komanda un licence

Projektu izstrādā komanda **Slepie**. Licence pašlaik nav norādīta.

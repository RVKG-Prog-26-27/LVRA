# LVRA, Latvijas Valodas Rokraksta Atpazīšana

A responsive web app (phone first) that turns a photo of handwritten Latvian text into digital text the user can read, correct and copy. Made by the Konstantins, team Slepie.

> **Version 0.2 (beta).** The web app and the AI run together on PythonAnywhere (EU). The AI is the CRNN model from `../latvian_htr` (`checkpoints_v2/crnn_best_migrated.pt`), run with numpy.

## User flow

1. **First visit:** the "Lietošanas noteikumi" window opens and cannot be skipped. The user chooses **Piekrītu visiem noteikumiem** (includes consent to AI training) or **Piekrītu tikai obligātajiem** (no training). The choice is remembered in the browser. The footer link "Lietošanas noteikumi" reopens the window to change it.
2. The user taps **Izvēlēties attēlus** (photo library, several photos at once) or **Uzņemt foto** (camera). More photos can be added from the preview with **Uzņemt vēl foto** and **Pievienot no galerijas**. All photos together may be up to 50 MB.
3. The app shows a preview of each photo. The user can rotate or remove each one and confirms with **Atpazīt tekstu**.
4. The photos are recognised one after another. For each photo, in the browser, the photo is rotated, scaled to fit 1920 x 1080 (1080 x 1920 for portrait), straightened if the writing is tilted, and **split into one image per text line**: `0001.png` is the first line, `0002.png` the second, and so on. The full photo is never sent.
5. The line PNGs, a `labels.csv` (header `filename,text`, text column empty) and the consent choice are sent to the server. The server fills in the text column and sends the CSV back.
6. The editor shows the text, one line per PNG, in order. With several photos each photo has its own tab ("1. attēls", "2. attēls", ...); a photo that failed shows its error in its tab. The user can correct the text. **Kopēt tekstu** copies one photo's text, **Kopēt visu** copies all of them. Corrections are not sent anywhere.
7. After recognition the photos are deleted from the browser's memory and only the text remains.

## Folder layout

```
LVRA/
  latvian_htr/            the AI (training code, checkpoints).
  web-app/
    src/                  the website (React + TypeScript)
    server/app.py         Flask server: serves the website and the AI endpoint
    server/htr_numpy.py   the CRNN model run with numpy
    server/alphabet_table.py  copy of the character tables and CTC decoding from latvian_htr
    requirements.txt      Python packages for the server (pythonanywhere.com)
    dist/                 the built website ("npm run build"), this is what the server serves
```

The server reads one file from `latvian_htr` without changing it: `checkpoints_v2/crnn_best_migrated.pt` (the trained weights). It does not import any code from `latvian_htr`, because that code needs PyTorch.

### Keep the server in sync with latvian_htr

The server has its own copies of things the AI side may change. When one of these files changes in `latvian_htr`, the matching server file has to be updated:

| latvian_htr | server |
|---|---|
| `alphabet.py` (letters, punctuation, decoding) | `server/alphabet_table.py` |
| `model.py` (layers, pooling sizes) | `forward()` and `check_architecture()` in `server/htr_numpy.py` |
| `synthetic_data.py` `TARGET_HEIGHT` (now 64) | `TARGET_HEIGHT` in `server/htr_numpy.py` |
| `infer.py` / `dataset.py` (image preparation, beam width) | `prepare_line()` and `BEAM_WIDTH` in `server/htr_numpy.py` |

`server/alphabet_table.py` holds two character tables: 137 classes (the current `alphabet.py`, used by `crnn_best_migrated.pt`) and 131 classes (`old_alphabet.py`, used by the older `crnn_best.pt`). The server counts the classes in the checkpoint and picks the matching table, so both checkpoints read the right letters. `x_stride` (2 or 4) is read from the checkpoint as well.

If the model file no longer fits the server's copy, the server does not return wrong text: recognition fails, users see "Atpazīšana pašlaik nav pieejama", and the error log names the layer or the class count that does not match.

## Running it on your computer

### 1. Install Node.js and Python (one time)

- Node.js **LTS** from https://nodejs.org (check with `node -v`)
- Python 3.10 or newer from https://www.python.org (check with `python --version`)

### 2. Install the packages (one time)

Open a terminal in the `web-app` folder:

```
npm install
pip install -r requirements.txt
```

### 3. Build and run the server

```
npm run build
python server/app.py
```

Open http://127.0.0.1:5000. This is exactly what runs on PythonAnywhere.

For development with automatic reload, keep `python server/app.py` running and start `npm run dev` in a second terminal, then open http://localhost:5173. Requests to `/api` are passed on to the server on port 5000.

### Trying it on a phone

1. Connect the computer and the phone to the same Wi-Fi network.
2. Run the server (`python server/app.py`) and, in a second terminal, `npm run dev:phone`. The terminal prints a "Network" address such as `http://192.168.1.20:5173`.
3. Open that address on the phone.

### Developer panel

Add `?dev=1` to the address, for example http://localhost:5173/?dev=1. A panel appears at the bottom of the page where you can:

- switch between the **real AI server** and the **mock** (sample text, no server needed),
- choose a mock scenario (success, no handwriting found, poor quality, image rejected, service unavailable, never answers, unexpected error) and the simulated delay,
- shorten the timeout to see the timeout message quickly,
- see the **last request**: every line PNG the browser cut out, the `labels.csv` that was sent and the one that came back. Use this to check the line splitting.

The panel is a developer tool.

## Deploying on PythonAnywhere (EU)

The free plan allows one web app, so the website and the AI run as one Flask app.

### 1. Build the website

In the `web-app` folder: `npm run build`. This creates `web-app/dist`.

### 2. Upload the files

Sign in at https://eu.pythonanywhere.com. In the **Files** tab create `/home/user.name/LVRA` and upload into it:

```
/home/user.name/LVRA/web-app/server/        app.py, htr_numpy.py, alphabet_table.py
/home/user.name/LVRA/web-app/dist/          everything from dist (index.html, favicon.svg, assets/)
/home/user.name/LVRA/web-app/requirements.txt
/home/user.name/LVRA/latvian_htr/checkpoints_v2/crnn_best_migrated.pt
```

Replace `user.name` with your PythonAnywhere username, here and in the steps below. Do not upload `node_modules`. The free plan has little disk space, so upload only the one checkpoint the server uses. An easier way for many files: zip them on your computer, upload the zip, then in a **Bash console** go to the folder with `cd ~` and unpack it with `unzip` followed by the zip's name.

### 3. Install the Python packages

Open a **Bash console** (Consoles tab) and run:

```
pip3.10 install --user -r ~/LVRA/web-app/requirements.txt
```

Flask, numpy and Pillow are usually already installed there; the command then just confirms them.

### 4. Create the web app

**Web** tab, then **Add a new web app**, then **Manual configuration**, then **Python 3.10**.

### 5. Edit the WSGI file

On the Web tab, open the link **WSGI configuration file**, delete everything in it and paste:

```python
import os
import sys

os.environ.setdefault("LVRA_DATA_DIR", "/home/user.name/lvra_data/real_dataset")

path = "/home/user.name/LVRA/web-app/server"
if path not in sys.path:
    sys.path.insert(0, path)

from app import app as application
```

Save it.

### 6. Turn on HTTPS and reload

On the Web tab switch on **Force HTTPS** (copying to the clipboard needs HTTPS), then press **Reload**.

### 7. Check

- https://user.name.eu.pythonanywhere.com/api/health should show `true` for `website_built` and `checkpoint_found`. `model_loaded` turns `true` after the first recognition.
- https://user.name.eu.pythonanywhere.com shows the app.
- If something fails, the Web tab has an **Error log** link.

### Updating

- Website changed: `npm run build`, upload the new `dist` folder (replace the old one), press **Reload**.
- New model: the simplest way is to upload the new weights into `latvian_htr/checkpoints_v2/` under the name `crnn_best_migrated.pt` (replacing the old file) and press **Reload**. To use a file with another name, add an `LVRA_CHECKPOINT` line to the WSGI file, written like the `LVRA_DATA_DIR` line, with the full path of the new file. If `alphabet.py`, `model.py`, `TARGET_HEIGHT` or the image preparation changed, update the server first (see "Keep the server in sync with latvian_htr").

## Server API

`POST /api/recognize`, `multipart/form-data`:

| Field | Content |
|---|---|
| `labels` | `labels.csv`, header `filename,text`, text column empty |
| `images` | one PNG per line, named `0001.png`, `0002.png`, ... |
| `training_consent` | `true` or `false` |

Answer `200`: `text/csv`, the same `labels.csv` with the text filled in (crossed-out word marks `¤` removed). Errors: JSON `{"error": "<code>"}` with `INVALID_IMAGE` (400/413), `NO_TEXT_FOUND` (400) or `SERVICE_UNAVAILABLE` (503). The web app shows each code as a Latvian message.

Limits: 100 lines, 16 MB per request, line images up to 4000 x 1000 px.

`GET /api/health`: JSON with `website_built`, `checkpoint_found`, `model_loaded`.

Settings (environment variables, all optional): `LVRA_HTR_DIR` (folder of the AI code, default `../latvian_htr`), `LVRA_CHECKPOINT` (default `checkpoints_v2/crnn_best_migrated.pt` in `LVRA_HTR_DIR`), `LVRA_DATA_DIR` (default `~/lvra_data/real_dataset`).

## Training data

Only for users who chose **Piekrītu visiem noteikumiem** (`training_consent=true`), the server adds the lines to `/home/user.name/lvra_data/real_dataset`:

```
real_dataset/
  labels.csv     filename,text
  images/        20261108_161530_k3f9a2_0001.png, ...
```

This is the format `latvian_htr/train.py --real-data` reads. The `text` column holds the **AI's guess** (crossed-out word marks `¤` kept). The Slepie team checks the images and corrects the text before training. Download the folder from the Files tab (or zip it in a Bash console: `cd ~/lvra_data && zip -r real_dataset.zip real_dataset`).

With `training_consent=false` nothing is written to disk.

## Architecture

```
User interface          src/App.tsx, src/components/*
                        (short Latvian texts in src/i18n/lv.ts,
                         terms in TermsDialog.tsx, privacy notice in PrivacyNotice.tsx)
Consent                 src/consent/consentStorage.ts (browser localStorage, versioned)
Screen state            src/app/appState.ts, src/app/useLvraApp.ts
Image handling          src/image/imageProcessing.ts, src/image/imageSize.ts
Line splitting          src/image/lineSegmentation.ts
labels.csv              src/recognition/labelsCsv.ts
Recognition entry       src/recognition/recognizeHandwriting.ts (CSV, timeout, cancel)
Recognition service     src/recognition/types.ts (interface)
                        src/recognition/serverRecognitionService.ts (the real AI)
                        src/recognition/mockRecognitionService.ts (sample text)
                        src/recognition/createRecognitionService.ts (picks the service)
Recognised text         src/recognition/segments.ts (uncertain word highlighting, switched off)
Editing and copying     src/editor/editorDom.ts, src/components/ResultEditor.tsx,
                        src/components/ResultTabs.tsx, src/utils/clipboard.ts
Server                  server/app.py, server/htr_numpy.py, server/alphabet_table.py
```

Errors from every layer use one set of codes (`src/recognition/errors.ts`). The UI turns each code into a plain Latvian message, so no technical error text reaches the user.

Fixed product values are in `src/config/config.ts`: backend (`server` or `mock`), 60 s timeout per photo, uncertain word highlighting (off, because the AI returns only text per line), 1920 x 1080 limit, 50 MB for all photos together, 100 lines, terms version, privacy contact email.

## Line splitting

`src/image/lineSegmentation.ts` splits each photo into line images in the browser:

1. **Lighting:** evens out the lighting, so paper becomes white and shadows disappear.
2. **Ink shapes:** marks the ink and finds the separate ink shapes. Shapes that are not writing are thrown away: anything touching the left or right edge of the photo (table, neighbouring page, fingers), large shapes at the top or bottom edge, a finger coming in from below, page outlines, long printed or ruled lines and filled patches.
3. **Straightening:** measures how much the writing is tilted (up to 12 degrees either way) and straightens the photo, so the lines run level. The line images are cut from the straightened photo.
4. **Grouping:** groups the shapes into lines. Two shapes belong to the same line when they sit side by side at about the same height; shapes stacked above each other are never joined. Parts of one line with a wide gap between words are joined when they meet at the same height.
5. **Small marks:** Latvian diacritics, dots and commas, and loose pieces of faint letters, go to the nearest line, preferring the line below them. A tall shape that reaches into two lines is cut between them.
6. **Cutting out:** cuts out each line with a margin. Only the ink of that line is drawn; parts of the lines above and below that fall inside the rectangle are painted white, so every image holds exactly one line.

Tested on photos of a lined notebook and plain paper, also tilted by up to 6 degrees and with lines squeezed closer together. Very strongly curved lines, or lines whose letters touch along a long stretch, can still come out split or partly wrong. Text from a neighbouring page that is fully inside the photo (not touching its edge) can end up in the line images.

# LVRA, Latvijas Valodas Rokraksta Atpazīšana

A responsive web app (phone first) that turns a photo of handwritten Latvian text into digital text the user can read, correct and copy. Made by the Slepie team.

The user interface is entirely in Latvian. This documentation is in English for developers.

> **Version 0.2.** The web app and the AI run together on PythonAnywhere (EU). The AI is the CRNN model from `../latvian_htr` (`checkpoints_v2/crnn_step30000.pt`), run with numpy. It reads computer-drawn handwriting fonts almost perfectly but real handwriting mostly wrongly, so users see the notice "AI vēl mācās, tāpēc atpazītais teksts var būt neprecīzs."

## User flow

1. **First visit:** the "Lietošanas noteikumi" window opens and cannot be skipped. The user chooses **Piekrītu visiem noteikumiem** (includes consent to AI training) or **Piekrītu tikai obligātajiem** (no training). The choice is remembered in the browser. The footer link "Lietošanas noteikumi" reopens the window to change it.
2. The user taps **Izvēlēties attēlu** (photo library) or **Uzņemt foto** (camera).
3. The app shows a preview. The user can rotate it and confirms with **Atpazīt tekstu**.
4. In the browser, the photo is rotated, scaled to fit 1920 x 1080 (1080 x 1920 for portrait) and **split into one image per text line**: `0001.png` is the first line, `0002.png` the second, and so on. The full photo is never sent.
5. The line PNGs, a `labels.csv` (header `filename,text`, text column empty) and the consent choice are sent to the server. The server fills in the text column and sends the CSV back.
6. The editor shows the text, one line per PNG, in order. The user can correct it. **Kopēt tekstu** copies it. Corrections are not sent anywhere.
7. After recognition the photo is deleted from the browser's memory and only the text remains.

## Folder layout

```
LVRA/
  latvian_htr/            the AI (training code, checkpoints). Not changed by the web app.
  web-app/
    src/                  the website (React + TypeScript)
    server/app.py         Flask server: serves the website and the AI endpoint
    server/htr_numpy.py   the CRNN model run with numpy (no PyTorch)
    server/alphabet_table.py  copy of the character table and CTC decoding (no PyTorch)
    requirements.txt      Python packages for the server
    dist/                 the built website ("npm run build"), this is what the server serves
```

The server reads one file from `latvian_htr` without changing it: `checkpoints_v2/crnn_step30000.pt` (the trained weights). It does not import any code from `latvian_htr`, because that code needs PyTorch.

### Keep the server in sync with latvian_htr

The server has its own copies of things the AI side may change. When one of these files changes in `latvian_htr`, the matching server file has to be updated:

| latvian_htr | server |
|---|---|
| `alphabet.py` (letters, punctuation, decoding) | `server/alphabet_table.py` |
| `model.py` (layers, pooling sizes) | `forward()` and `check_architecture()` in `server/htr_numpy.py` |
| `synthetic_data.py` `TARGET_HEIGHT` (now 64) | `TARGET_HEIGHT` in `server/htr_numpy.py` |
| `infer.py` / `dataset.py` (image preparation, beam width) | `prepare_line()` and `BEAM_WIDTH` in `server/htr_numpy.py` |

If the model file no longer fits the server's copy, the server does not return wrong text: recognition fails, users see "Atpazīšana pašlaik nav pieejama", and the error log names the layer that does not match.

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

### 3a. Quick way: build and run the server

```
npm run build
python server/app.py
```

Open http://127.0.0.1:5000. This is exactly what runs on PythonAnywhere.

### 3b. While changing the website: two terminals

Terminal 1: `python server/app.py` (the AI on port 5000)
Terminal 2: `npm run dev` (the website on http://localhost:5173 with live reload; requests to `/api` are passed to port 5000)

### Trying it on a phone

1. Connect the computer and the phone to the same Wi-Fi network.
2. Run the server (terminal 1 above) and `npm run dev:phone`. The terminal prints a "Network" address such as `http://192.168.1.20:5173`.
3. Open that address on the phone.

### Developer panel

Add `?dev=1` to the address, for example http://localhost:5173/?dev=1. A panel appears at the bottom of the page where you can:

- switch between the **real AI server** and the **mock** (sample text, no server needed),
- choose a mock scenario (success, no handwriting found, poor quality, image rejected, service unavailable, never answers, unexpected error) and the simulated delay,
- shorten the timeout to see the timeout message quickly,
- see the **last request**: every line PNG the browser cut out, the `labels.csv` that was sent and the one that came back. Use this to check the line splitting.

The panel is a developer tool and is in English on purpose. It records images only while it is enabled.

## Deploying on PythonAnywhere (EU), account cloudy1337

The free plan allows one web app, so the website and the AI run as one Flask app at https://cloudy1337.eu.pythonanywhere.com.

### 1. Build the website on your computer

In the `web-app` folder: `npm run build`. This creates `web-app/dist`.

### 2. Upload the files

Sign in at https://eu.pythonanywhere.com. In the **Files** tab create `/home/cloudy1337/LVRA` and upload into it:

```
/home/cloudy1337/LVRA/web-app/server/        app.py, htr_numpy.py, alphabet_table.py
/home/cloudy1337/LVRA/web-app/dist/          everything from dist (index.html, favicon.svg, assets/)
/home/cloudy1337/LVRA/web-app/requirements.txt
/home/cloudy1337/LVRA/latvian_htr/checkpoints_v2/crnn_step30000.pt
```

Do not upload `node_modules`. The free plan has little disk space, so upload only the one checkpoint the server uses. An easier way for many files: zip them on your computer, upload the zip, then in a **Bash console** go to the folder with `cd ~` and unpack it with `unzip` followed by the zip's name.

### 3. Install the Python packages

Open a **Bash console** (Consoles tab) and run:

```
pip3.10 install --user -r ~/LVRA/web-app/requirements.txt
```

Flask, numpy and Pillow are usually already installed there; the command then just confirms them. PyTorch is not needed.

### 4. Create the web app

**Web** tab, then **Add a new web app**, then **Manual configuration**, then **Python 3.10**. (If you choose a newer Python, use the matching pip in step 3, for example `pip3.12`.)

### 5. Edit the WSGI file

On the Web tab, open the link **WSGI configuration file**, delete everything in it and paste:

```python
import os
import sys

os.environ.setdefault("LVRA_DATA_DIR", "/home/cloudy1337/lvra_data/real_dataset")

path = "/home/cloudy1337/LVRA/web-app/server"
if path not in sys.path:
    sys.path.insert(0, path)

from app import app as application
```

Save it.

### 6. Turn on HTTPS and reload

On the Web tab switch on **Force HTTPS** (copying to the clipboard needs HTTPS), then press **Reload**.

### 7. Check

- https://cloudy1337.eu.pythonanywhere.com/api/health should show `true` for `website_built` and `checkpoint_found`. `model_loaded` turns `true` after the first recognition.
- https://cloudy1337.eu.pythonanywhere.com shows the app.
- If something fails, the Web tab has an **Error log** link.

### Keeping it running

Free web apps switch off after a while. The Web tab shows the date; press the button there to extend it before that date.

### Updating

- Website changed: `npm run build`, upload the new `dist` folder (replace the old one), press **Reload**.
- New model: the simplest way is to upload the new weights into `latvian_htr/checkpoints_v2/` under the name `crnn_step30000.pt` (replacing the old file) and press **Reload**. To use a file with another name, add an `LVRA_CHECKPOINT` line to the WSGI file, written like the `LVRA_DATA_DIR` line, with the full path of the new file. If `alphabet.py`, `model.py`, `TARGET_HEIGHT` or the image preparation changed, update the server first (see "Keep the server in sync with latvian_htr").

## Server API

`POST /api/recognize`, `multipart/form-data`:

| Field | Content |
|---|---|
| `labels` | `labels.csv`, header `filename,text`, text column empty |
| `images` | one PNG per line, named `0001.png`, `0002.png`, ... |
| `training_consent` | `true` or `false` |

Answer `200`: `text/csv`, the same `labels.csv` with the text filled in. Errors: JSON `{"error": "<code>"}` with `INVALID_IMAGE` (400/413), `NO_TEXT_FOUND` (400) or `SERVICE_UNAVAILABLE` (503). The web app shows each code as a Latvian message.

Limits: 100 lines, 16 MB per request, line images up to 4000 x 1000 px.

`GET /api/health`: JSON with `website_built`, `checkpoint_found`, `model_loaded`.

## Training data

Only for users who chose **Piekrītu visiem noteikumiem** (`training_consent=true`), the server adds the lines to `/home/cloudy1337/lvra_data/real_dataset`:

```
real_dataset/
  labels.csv     filename,text
  images/        20261108_161530_k3f9a2_0001.png, ...
```

This is the format `latvian_htr/train.py --real-data` reads. The `text` column holds the **AI's guess**. The Slepie team checks the images and corrects the text before training. Download the folder from the Files tab (or zip it in a Bash console: `cd ~/lvra_data && zip -r real_dataset.zip real_dataset`).

With `training_consent=false` nothing is written to disk: uploads are kept in memory only and are gone when the request ends.

**Privacy promises in the app that the team must keep:** only Slepie team members see this data; it stays on PythonAnywhere EU servers; it is **deleted at the end of the LVRA project**. If any of this changes, update the texts in `src/components/TermsDialog.tsx` and `src/components/PrivacyNotice.tsx`, and change `terms.version` in `src/config/config.ts` so every user is asked again.

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
                        src/utils/clipboard.ts
Server                  server/app.py, server/htr_numpy.py, server/alphabet_table.py
```

Errors from every layer use one set of codes (`src/recognition/errors.ts`). The UI turns each code into a plain Latvian message, so no technical error text reaches the user.

Fixed product values are in `src/config/config.ts`: backend (`server` or `mock`), 60 s timeout, uncertain word highlighting (off, because the AI returns only text per line), 1920 x 1080 limit, 100 lines, terms version, privacy contact email.

## Line splitting

`src/image/lineSegmentation.ts` evens out the lighting, marks the ink, removes shapes that touch the photo's edge (table, page border), and separates lines by the white gaps between them. Latvian diacritics above and below letters are joined to their line. It works for fairly straight lines with some space between them. Strongly slanted or touching lines come out merged into one image. If that happens too often, the splitting should move to the server and use a stronger method.

## Known limitations

- The AI (`crnn_step30000.pt`) reads new sentences drawn with its handwriting fonts with almost no errors, but on the 41 real handwritten lines in `latvian_htr/data/real_dataset` about 78% of the characters come out wrong. It needs to be trained on real handwriting (collected with consent, see Training data).
- Speed: about 0.5 to 0.8 s per line on a fast PC, so a page of 15 to 20 lines takes roughly 10 to 15 s there. PythonAnywhere is usually slower. The app gives up after 60 s (`recognitionTimeoutMs` in `src/config/config.ts`).
- Very large photos may fail to open on older phones with little memory. The user then sees a message to try another image.
- HEIC photos open only in browsers that support HEIC (mainly Safari).

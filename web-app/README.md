# LVRA, Latvijas Valodas Rokraksta Atpazīšana

A responsive web app (phone first) that turns a photo of handwritten Latvian text into digital text the user can read, correct and copy. Made by the Slepie team.

The user interface is entirely in Latvian. This documentation is in English for developers.

> **Alpha version.** The handwriting recognition model is not connected yet. The app runs with a **temporary mock service** that always returns the same Latvian sample text, whatever image is chosen. While the mock is active, users see a "Demonstrācijas režīms" notice.

## User flow

1. The user taps **Izvēlēties attēlu** (photo library) or **Uzņemt foto** (camera).
2. The app shows a preview. The user can rotate it and confirms with **Atpazīt tekstu**.
3. The image is rotated, scaled to fit 1920 x 1080 (1080 x 1920 for portrait) and converted to JPEG in the browser.
4. A progress state with **Atcelt** is shown. After 60 seconds the attempt stops with a timeout message.
5. The recognised text appears with line breaks preserved. Words with confidence below 70% are highlighted, and a highlight disappears once the user edits that word.
6. **Kopēt tekstu** copies the text and shows **Teksts nokopēts!**
7. After successful recognition the image is deleted from memory and only the text remains.

## Setup

### 1. Install Node.js (one time)

Download the **LTS** version from https://nodejs.org and run the installer with the default options. Check it in a terminal:

```
node -v
```

### 2. Install and start

Open a terminal in the `web-app` folder:

```
npm install
npm run dev
```

Open http://localhost:5173 in your browser. Stop the server with Ctrl+C.

### Trying it on a phone

1. Connect the computer and the phone to the same Wi-Fi network.
2. Run `npm run dev:phone`. The terminal prints a "Network" address such as `http://192.168.1.20:5173`.
3. Open that address on the phone.

The camera button uses the phone's own camera app, and copying works without HTTPS. Your firewall may ask for permission the first time.

### Developer panel

Add `?dev=1` to the address, for example http://localhost:5173/?dev=1. A panel appears at the bottom of the page where you can:

- choose a mock scenario (success, all words confident, no handwriting found, poor quality, image rejected, service unavailable, never answers, unexpected error),
- change the simulated processing delay,
- shorten the timeout to see the timeout message quickly.

The panel is a temporary developer tool and is in English on purpose.

## Architecture

Each layer knows the one below it only through a small interface, so the recognition backend can be replaced without touching the UI.

```
User interface          src/App.tsx, src/components/*
                        (all Latvian texts in src/i18n/lv.ts)
Screen state            src/app/appState.ts, src/app/useLvraApp.ts
Image handling          src/image/imageProcessing.ts, src/image/imageSize.ts
Recognition entry       src/recognition/recognizeHandwriting.ts (timeout, cancel)
Recognition service     src/recognition/types.ts (interface)
                        src/recognition/mockRecognitionService.ts (TEMPORARY)
                        src/recognition/createRecognitionService.ts (picks the service)
Recognised text         src/recognition/segments.ts (uncertain word highlighting)
Editing and copying     src/editor/editorDom.ts, src/components/ResultEditor.tsx,
                        src/utils/clipboard.ts
```

Errors from every layer use one set of codes (`src/recognition/errors.ts`). The UI turns each code into a plain Latvian message, so no technical error text reaches the user.

Fixed product values (60 s timeout, 70% confidence threshold, 1920 x 1080 limit, JPEG quality, privacy contact email) are in `src/config/config.ts`.

## Known limitations

- The mock ignores the image content and always returns the same sample text.
- Very large photos may fail to open on older phones with little memory. The user then sees a message to try another image.
- HEIC photos open only in browsers that support HEIC (mainly Safari).

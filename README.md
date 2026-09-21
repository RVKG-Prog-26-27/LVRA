<<<<<<< HEAD
# LVRA, Latvijas Valodas Rokraksta Atpazīšana

A responsive web app (phone first) that turns a photo of handwritten Latvian text into ordinary digital text the user can read, correct and copy. Made by the Slepie team.

The user interface is entirely in Latvian. This documentation is in English for developers.

> **Status:** the handwriting recognition model is not finished. The app currently runs with a **temporary mock service** that returns a fixed Latvian sample text. While the mock is active, users see a visible "Demonstrācijas režīms" notice, so the sample is never presented as real recognition.

## User flow

1. The user taps **Izvēlēties attēlu** (photo library) or **Uzņemt foto** (camera).
2. The app shows a preview; the user can rotate it and confirms with **Atpazīt tekstu**.
3. The image is rotated, scaled to fit 1920 x 1080 (1080 x 1920 for portrait, never enlarged) and converted to JPEG in the browser.
4. The JPEG is sent to the recognition service. A progress state with **Atcelt** is shown; after 60 seconds the attempt stops with a timeout message.
5. The recognised text appears with line breaks preserved. Words with confidence below 70% are highlighted; a highlight disappears once the user edits that word.
6. **Kopēt tekstu** copies the text and shows **Teksts nokopēts!**
7. After successful recognition the image is deleted from memory; only the text remains. After a failure the image is kept so the user can press **Mēģināt vēlreiz**.

## Running it on your computer

### 1. Install Node.js (one time)

1. Go to https://nodejs.org and download the **LTS** version for your system.
2. Run the installer with the default options.
3. Open a terminal (macOS: Terminal; Windows: Command Prompt or PowerShell) and check:
   ```
   node -v
   ```
   It should print a version number, for example v22.x.x.

### 2. Start the app

1. Unzip the project and open a terminal in the project folder, for example:
   ```
   cd path/to/20260921_LVRA_Web_App
   ```
2. Install the dependencies (one time, needs internet):
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm run dev
   ```
4. Open http://localhost:5173 in your browser.

Stop the server with Ctrl+C in the terminal.

### Trying it on your phone

1. Connect the computer and phone to the same Wi-Fi network.
2. Run `npm run dev:phone`. The terminal prints a "Network" address such as `http://192.168.1.20:5173`.
3. Open that address on the phone.

Both buttons work over the local network: the camera button uses the phone's own camera app through the file input, and copying falls back to a method that does not need HTTPS. Your firewall may ask for permission the first time.

### Developer panel

Open the app with `?dev=1` added to the address (for example http://localhost:5173/?dev=1). A panel appears at the bottom that lets you:

- choose a mock scenario (success, all words confident, no handwriting found, poor quality, image rejected, service unavailable, never answers, unexpected error),
- change the simulated delay,
- shorten the timeout to test the timeout message quickly.

The panel is a temporary developer tool and is in English on purpose.

## Tests

```
npm test
```

The tests cover the 1080p size calculation and rotation, the recognition response parser, HTTP error mapping, the API adapter (with a fake server), timeout and cancellation, the mock service, the screen state machine, the highlighting of uncertain words and the editor's "highlight disappears after editing" rule.

`npm run typecheck` checks the TypeScript types.

## Building for the web

```
npm run build
```

The finished site is written to `dist/`. It is a static site and can be hosted on any static web host. In production the site must be served over **HTTPS**; browsers restrict clipboard access on plain HTTP.

`npm run preview` serves the built site locally for a final check.

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Meaning |
|---|---|
| `VITE_RECOGNITION_MODE` | `mock` (default, temporary sample text) or `api` (real server). |
| `VITE_RECOGNITION_API_URL` | Base address of the recognition server. The app posts to `<address>/recognize`. |
| `VITE_PRIVACY_CONTACT_EMAIL` | Email shown in the privacy notice. **Currently a placeholder (kontakti@example.com); replace before publishing.** |

Fixed product values (60 s timeout, 70% confidence threshold, 1920 x 1080 limit, JPEG quality 0.92) are in `src/config/config.ts`.

Restart `npm run dev` after changing `.env`.

## Architecture

Each layer only knows the one below it through a small interface, so the recognition backend can change without touching the UI.

```
User interface          src/App.tsx, src/components/*
      |                 (Latvian texts only from src/i18n/lv.ts)
Screen state            src/app/appState.ts (pure state machine)
      |                 src/app/useLvraApp.ts (connects UI to the layers below)
Image handling          src/image/imageProcessing.ts (open, rotate, scale, JPEG)
      |                 src/image/imageSize.ts (pure size maths)
Recognition entry       src/recognition/recognizeHandwriting.ts (timeout, cancel)
      |
Recognition service     HandwritingRecognitionService interface (types.ts)
      |                   ApiRecognitionService   real server over HTTP
      |                   MockRecognitionService  TEMPORARY sample text
      |                 createRecognitionService.ts picks one
Recognised text         segments.ts (uncertain word highlighting)
      |
Editing and copying     src/editor/editorDom.ts, src/components/ResultEditor.tsx
                        src/utils/clipboard.ts
```

Errors from every layer use one set of codes (`src/recognition/errors.ts`); the UI turns each code into a plain Latvian message, so no technical error text reaches the user.

## Connecting the real model

**Option A: the model is behind an HTTP API (planned).**

1. Make the server follow the contract in [`docs/API.md`](docs/API.md).
2. In `.env` set `VITE_RECOGNITION_MODE=api` and `VITE_RECOGNITION_API_URL=https://your-server/...`.
3. If the server's format differs from the contract, adapt only `src/recognition/apiRecognitionService.ts` and `parseResponse.ts`.
4. Run `npm test` and try it with a few real images.

**Option B: a different kind of backend** (another request format, authentication, on-device inference, etc.). Write a new class implementing `HandwritingRecognitionService` from `src/recognition/types.ts` and return it from `createRecognitionService.ts`. No UI changes are needed.

**When the real model is live,** remove the temporary files: `src/recognition/mockRecognitionService.ts`, `src/recognition/mockSettings.ts`, `src/components/DevPanel.tsx`, the mock test cases, and the mock branch in `createRecognitionService.ts`. The demonstration notice disappears automatically in `api` mode.

## Privacy

- The browser keeps the image only in memory. It is never written to local storage, cookies or disk, and it is released right after successful recognition.
- The app uses no cookies, analytics, ads or third-party fonts and makes no requests other than to the recognition server.
- The privacy notice (footer link **Privātuma paziņojums**) promises that the image is also deleted on the server right after recognition. **The recognition server must honour this:** process the image in memory, do not store or log it, and delete any temporary copy once the response is sent. See `docs/API.md`.
- Replace the placeholder contact email before publishing, and have the notice text reviewed before public launch.

## Accepted images

JPEG and PNG are the formats the app asks for. Any other image the browser can open (for example WebP, or HEIC in Safari) is also accepted and converted to JPEG. If the browser cannot open the file, the user is asked to choose a JPEG or PNG image. There is no input size limit; very large photos are scaled down in the browser before sending.

## Project structure

```
index.html              page shell (lang="lv")
public/favicon.svg
src/
  App.tsx               screens
  main.tsx              entry point
  styles.css            visual design (exercise-book sheet, ruled result area)
  app/                  screen state machine and hook
  components/           UI pieces
  config/config.ts      settings
  editor/               result editor DOM logic
  i18n/lv.ts            all Latvian texts
  image/                image processing
  recognition/          service interface, API adapter, temporary mock, timeout
  utils/clipboard.ts
docs/API.md             contract for the recognition server
.env.example            configuration template
```

## Known limitations

- The mock ignores the image content; it always returns the same sample text.
- Very large photos (tens of megapixels) may fail to open on older phones with little memory; the user then sees a message to try another image.
- HEIC photos open only in browsers that support HEIC (mainly Safari). Other browsers show the JPEG or PNG message.
- Undo (Ctrl+Z) may not undo line breaks and pastes in the result editor, because these are inserted as plain text by the app.
=======
# LVRA
>>>>>>> 2dba99254eac5fc104c7b2fba6fad84de97fd166

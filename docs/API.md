# Recognition server API (proposed contract)

This is the format `ApiRecognitionService` expects. The model team may change it; if so, only `src/recognition/apiRecognitionService.ts` and `src/recognition/parseResponse.ts` need updating.

## Request

```
POST <VITE_RECOGNITION_API_URL>/recognize
Content-Type: multipart/form-data
```

| Field | Type | Description |
|---|---|---|
| `image` | file (`image/jpeg`) | The handwriting image, already rotated by the user and scaled to fit 1920 x 1080 (1080 x 1920 for portrait). |

No authentication is used.

The server must allow cross-origin requests (CORS) from the site's address if the app and the server are on different domains.

## Successful response

`200 OK`, `Content-Type: application/json`

```json
{
  "text": "Šodien no rīta devos uz tirgu.\nNopirku maizi, pienu un ābolus.",
  "words": [
    { "text": "Šodien", "confidence": 0.97 },
    { "text": "no", "confidence": 0.99 },
    { "text": "tirgu", "confidence": 0.62 }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `text` | yes | The recognised Latvian text exactly as written (no spelling correction). Lines separated by `\n`. |
| `words` | no | Words in reading order, each with `text` (must appear in `text` exactly) and `confidence` between 0 and 1. Words below 0.7 are highlighted in the app. Punctuation may be left out. |

An empty or whitespace-only `text` is shown to the user as "no handwriting found".

## Error response

Any non-2xx status. Optionally a JSON body naming the reason:

```json
{ "error": { "code": "POOR_QUALITY" } }
```

| `error.code` | Meaning | Suggested status |
|---|---|---|
| `NO_TEXT_FOUND` | No handwriting detected in the image | 422 |
| `POOR_QUALITY` | Image too blurry, dark or small to read | 422 |
| `INVALID_IMAGE` | Image unreadable or unsupported | 400 or 415 |

Without a known code the app uses the status: 400/413/415 as image rejected, 408/504 as timeout, 422 as no handwriting found, 404/429/5xx as service unavailable.

## Timing

The app waits at most 60 seconds, then cancels the request and shows a timeout message.

## Privacy requirements for the server

The app's privacy notice tells users that images are deleted right after recognition and stored nowhere. The server must therefore:

- process the image in memory only, or delete any temporary file as soon as the response is sent,
- not store images or recognised text in databases, logs, backups or training sets,
- not log request bodies.

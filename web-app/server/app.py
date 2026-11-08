"""
LVRA server: one Flask app that serves the website and runs the AI.

  GET  /                 the built website (web-app/dist, made with "npm run build")
  POST /api/recognize    line images + labels.csv in, filled labels.csv out
  GET  /api/health       quick check that the server and the model are in place

POST /api/recognize, multipart/form-data:
  labels            labels.csv, header "filename,text", text column empty
  images            one PNG per line, named 0001.png, 0002.png, ... (top line first)
  training_consent  "true" or "false"

Answer: 200 with text/csv (labels.csv, text column filled in), or an error
as JSON {"error": "<code>"} with a code the web app already knows:
INVALID_IMAGE, NO_TEXT_FOUND, SERVICE_UNAVAILABLE.

Privacy (must match the privacy notice):
  * Uploads are read into memory only, never into temporary files.
  * training_consent=false: nothing is written to disk; the images are gone
    when the request ends.
  * training_consent=true: the line images and the AI's text are added to
    DATA_DIR (images/ + labels.csv), in the format latvian_htr/train.py
    reads with --real-data. Files get unique names, so uploads from
    different people never overwrite each other. Nothing links a file to a
    person: no IP address, name or other identifier is stored.

Settings (environment variables, all optional):
  LVRA_HTR_DIR      folder of the AI code, default: ../../latvian_htr from this file
  LVRA_CHECKPOINT   model weights, default: checkpoints_v2/crnn_step30000.pt in LVRA_HTR_DIR
  LVRA_DATA_DIR     where training data is kept, default: ~/lvra_data/real_dataset
                    (on PythonAnywhere: /home/cloudy1337/lvra_data/real_dataset)
"""

import csv
import io
import os
import re
import secrets
import time

from flask import Flask, Request, jsonify, request, send_from_directory
from PIL import Image

from htr_numpy import Recognizer

try:
    import fcntl  # file locking on Linux (PythonAnywhere)
except ImportError:  # Windows, local testing only
    fcntl = None

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.dirname(HERE)
DIST_DIR = os.path.join(WEB_APP_DIR, "dist")
HTR_DIR = os.path.abspath(os.environ.get("LVRA_HTR_DIR", os.path.join(WEB_APP_DIR, "..", "latvian_htr")))
CHECKPOINT = os.environ.get("LVRA_CHECKPOINT", os.path.join(HTR_DIR, "checkpoints_v2", "crnn_step30000.pt"))
DATA_DIR = os.path.expanduser(os.environ.get("LVRA_DATA_DIR", "~/lvra_data/real_dataset"))

MAX_LINES = 100  # same as config.lines.maxLines in the web app
MAX_LINE_WIDTH = 4000
MAX_LINE_HEIGHT = 1000
FILENAME_RE = re.compile(r"^\d{4}\.png$")


class InMemoryRequest(Request):
    """Keeps uploaded files in memory. Flask would otherwise put large uploads into temporary files on disk."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


app = Flask(__name__, static_folder=None)
app.request_class = InMemoryRequest
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per request

recognizer = Recognizer(CHECKPOINT)


def error(code, status):
    return jsonify({"error": code}), status


@app.errorhandler(413)
def too_large(_):
    return error("INVALID_IMAGE", 413)


# ---------------------------------------------------------------- website

@app.get("/")
def index():
    if not os.path.isfile(os.path.join(DIST_DIR, "index.html")):
        return "The website is not built yet. Run 'npm run build' in the web-app folder.", 503
    return send_from_directory(DIST_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    if path.startswith("api/"):
        return error("UNKNOWN", 404)
    return send_from_directory(DIST_DIR, path)


# ---------------------------------------------------------------- API

@app.get("/api/health")
def health():
    return jsonify(
        {
            "website_built": os.path.isfile(os.path.join(DIST_DIR, "index.html")),
            "checkpoint_found": os.path.isfile(CHECKPOINT),
            "model_loaded": recognizer.loaded,
        }
    )


def read_labels(file_storage):
    text = file_storage.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "filename" not in reader.fieldnames or "text" not in reader.fieldnames:
        raise ValueError("labels.csv must have the columns filename,text")
    return [row["filename"] for row in reader]


@app.post("/api/recognize")
def recognize():
    labels = request.files.get("labels")
    uploads = request.files.getlist("images")
    consent = request.form.get("training_consent") == "true"

    if labels is None:
        return error("INVALID_IMAGE", 400)
    if not uploads:
        return error("NO_TEXT_FOUND", 400)
    if len(uploads) > MAX_LINES:
        return error("INVALID_IMAGE", 400)

    try:
        filenames = read_labels(labels)
    except (ValueError, UnicodeDecodeError, csv.Error):
        return error("INVALID_IMAGE", 400)

    # Check every image before doing any work.
    images = {}
    for upload in uploads:
        name = upload.filename or ""
        if not FILENAME_RE.match(name) or name in images:
            return error("INVALID_IMAGE", 400)
        data = upload.read()
        try:
            img = Image.open(io.BytesIO(data))
            if img.format != "PNG":
                return error("INVALID_IMAGE", 400)
            if img.width > MAX_LINE_WIDTH or img.height > MAX_LINE_HEIGHT:
                return error("INVALID_IMAGE", 400)
            img.load()
        except Exception:
            return error("INVALID_IMAGE", 400)
        images[name] = (img, data)
    if sorted(images) != sorted(filenames):
        return error("INVALID_IMAGE", 400)

    # Recognise every line.
    try:
        texts = {name: recognizer.recognize(images[name][0]) for name in filenames}
    except FileNotFoundError:
        app.logger.exception("Model files missing")
        return error("SERVICE_UNAVAILABLE", 503)
    except Exception:
        app.logger.exception("Recognition failed")
        return error("SERVICE_UNAVAILABLE", 503)

    if consent:
        try:
            save_for_training(filenames, images, texts)
        except Exception:
            # The user still gets the text; the failed save is only logged.
            app.logger.exception("Saving training data failed")

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["filename", "text"])
    for name in filenames:
        writer.writerow([name, texts[name]])
    return app.response_class(out.getvalue(), mimetype="text/csv; charset=utf-8")


def save_for_training(filenames, images, texts):
    """
    Adds the lines to DATA_DIR in latvian_htr's format:
      DATA_DIR/labels.csv   filename,text   (text = the AI's guess, the team corrects it)
      DATA_DIR/images/<YYYYMMDD_HHMMSS>_<random>_0001.png
    """
    images_dir = os.path.join(DATA_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    batch = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"

    rows = []
    for name in filenames:
        stored = f"{batch}_{name}"
        with open(os.path.join(images_dir, stored), "wb") as f:
            f.write(images[name][1])
        rows.append([stored, texts[name]])

    labels_path = os.path.join(DATA_DIR, "labels.csv")
    with open(labels_path, "a", encoding="utf-8", newline="") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.tell() == 0:
                f.write("filename,text\n")
            csv.writer(f, lineterminator="\n").writerows(rows)
        finally:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    # Local testing: python server/app.py, then open http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=False)

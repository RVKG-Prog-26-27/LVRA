import csv
import json
import os
import random
import shutil
import threading
import time
import subprocess
import re
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageSequence, ImageTk
from pathlib import Path

BASE = Path(__file__).resolve().parent
FOLDER = BASE / "data" / "real_dataset" / "images"
FOLDER_CSV = BASE / "data" / "real_dataset"
CSV_PATH = FOLDER_CSV / "labels.csv"

PROGRESS_PATH = os.path.join(FOLDER, ".verify_progress")
SETTINGS_PATH = os.path.join(FOLDER, ".verify_settings.json")
ORIG_DIR = os.path.join(FOLDER, "_originals")
PRIKOL_DIR = BASE / "prikol"   # GIFs for the slots are taken from this folder

BG = "#1e1e1e"
PANEL = "#252526"
FG = "#e6e6e6"
MUTED = "#8a8a8a"
INPUT_BG = "#2d2d30"
ACCENT = "#3a7bd5"
BTN_BG = "#3c3c3c"
BTN_ACTIVE = "#4a4a4a"
CANVAS_BG = "#121212"
SEL_COLOR = "#ffcc00"

GIF_SLOTS = 3                      # number of small media slots
GIF_HEIGHT = 96                    # height of the small slots row (width stretches automatically)
BIG_HEIGHT = 220                   # initial height of the big field (drag the divider to change)
BIG_MODE = "stretch"               # "stretch" = fill the whole field (like the small slots), "fit" = keep proportions
LEFT_WIDTH = 780                   # initial width of the left panel (drag the divider to change)
STATIC_MAX = 2500                  # photos are downscaled to this size once, then resized quickly
MAX_VIDEO_FPS = 60

IMAGE_EXTS = {".gif", ".webp", ".apng", ".png", ".jpg", ".jpeg", ".jfif", ".bmp",
              ".tif", ".tiff", ".ico", ".ppm", ".pgm"}
# videos need OpenCV:  pip install opencv-python-headless   (no sound, looped)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpg",
              ".mpeg", ".mpe", ".3gp", ".ogv", ".ts", ".mts", ".m2ts", ".vob", ".asf"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS   # everything in the prikol folder with these extensions

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
RESAMPLE_FAST = getattr(Image, "Resampling", Image).BILINEAR

# Git commit settings
REAL_DATA_VERSION_START = 8  # The current real-data commit number


def read_rows(path):
    for enc in ("utf-8-sig", "utf-8", "cp1257"):
        try:
            with open(path, newline="", encoding=enc) as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError("Could not read labels.csv")
    header = rows[0]
    data = [[r[0], ",".join(r[1:]) if len(r) > 2 else (r[1] if len(r) > 1 else "")]
            for r in rows[1:] if r]
    return header, data


def make_button(parent, text, command):
    return tk.Button(parent, text=text, command=command, bg=BTN_BG, fg=FG,
                     activebackground=BTN_ACTIVE, activeforeground=FG,
                     relief="flat", bd=0, padx=10, pady=6,
                     font=("Segoe UI", 10), cursor="hand2")


def make_copyable(parent, text, font, fg, height, width=80):
    """Read-only text that can be selected with the mouse and copied (Ctrl+C / right click)."""
    t = tk.Text(parent, height=height, width=width, wrap="word", font=font, bg=PANEL, fg=fg,
                relief="flat", bd=0, highlightthickness=0, padx=0, pady=0,
                selectbackground=ACCENT, selectforeground="white", cursor="xterm")
    t.insert("1.0", text)
    t.config(state="disabled")

    menu = tk.Menu(t, tearoff=0, bg=BTN_BG, fg=FG, activebackground=ACCENT,
                   activeforeground="white", bd=0)

    def copy(_=None):
        try:
            data = t.get("sel.first", "sel.last")
        except tk.TclError:
            data = t.get("1.0", "end-1c")
        t.clipboard_clear()
        t.clipboard_append(data)

    menu.add_command(label="Copy", command=copy)
    t.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
    return t


def set_readonly_text(widget, text):
    widget.config(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.config(state="disabled")


def target_size(src_w, src_h, tw, th, mode):
    if mode == "fit":
        r = min(tw / src_w, th / src_h)
        return max(1, int(src_w * r)), max(1, int(src_h * r))
    return tw, th


class MediaWorker(threading.Thread):
    """Decodes an image / GIF / video in the background and hands ready-to-show frames to the slot.
    The UI thread never decodes or resizes anything, so big files do not freeze the window."""

    def __init__(self, slot, path):
        super().__init__(daemon=True)
        self.slot = slot
        self.path = Path(path)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def publish(self, img):
        with self.slot.lock:
            if self.slot.worker is self:
                self.slot.pending = img

    def run(self):
        try:
            if self.path.suffix.lower() in VIDEO_EXTS:
                self.run_video()
            else:
                self.run_image()
        except Exception as ex:
            with self.slot.lock:
                if self.slot.worker is self:
                    self.slot.error = f"cannot open\n{self.path.name}\n{str(ex)[:60]}"

    # ---- pictures and GIFs ----
    def run_image(self):
        bgcol = (45, 45, 48, 255)
        with Image.open(self.path) as im:
            n = getattr(im, "n_frames", 1)
            if n <= 1:
                try:
                    im = ImageOps.exif_transpose(im)      # photos: respect EXIF rotation
                except Exception:
                    pass
                rgba = im.convert("RGBA")
                rgba.thumbnail((STATIC_MAX, STATIC_MAX), RESAMPLE)
                bg = Image.new("RGBA", rgba.size, bgcol)
                bg.alpha_composite(rgba)
                base = bg.convert("RGB")
                while not self.stop_event.is_set():
                    tgt = self.slot.target
                    size = target_size(base.width, base.height, tgt[0], tgt[1], self.slot.mode)
                    self.publish(base.resize(size, RESAMPLE))
                    while not self.stop_event.is_set() and self.slot.target == tgt:
                        self.stop_event.wait(0.1)         # wait until the slot is resized
                return
            i = 0
            next_time = time.perf_counter()
            while not self.stop_event.is_set():
                im.seek(i)
                if "transparency" in im.info or im.mode in ("RGBA", "LA"):
                    rgba = im.convert("RGBA")
                    bg = Image.new("RGBA", rgba.size, bgcol)
                    bg.alpha_composite(rgba)
                    frame = bg.convert("RGB")
                else:
                    frame = im.convert("RGB")
                tgt = self.slot.target
                size = target_size(frame.width, frame.height, tgt[0], tgt[1], self.slot.mode)
                self.publish(frame.resize(size, RESAMPLE_FAST, reducing_gap=2.0))
                delay = max(20, int(im.info.get("duration", 100) or 100)) / 1000.0
                next_time += delay
                wait = next_time - time.perf_counter()
                if wait > 0:
                    self.stop_event.wait(wait)
                else:
                    next_time = time.perf_counter()        # decoding is slower than the GIF - do not accumulate lag
                i = (i + 1) % n

    # ---- video ----
    def run_video(self):
        try:
            import cv2
        except ImportError:
            raise RuntimeError("videos need: pip install opencv-python-headless")
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise RuntimeError("video cannot be opened")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps != fps or fps < 1:
                fps = 25.0
            fps = min(fps, MAX_VIDEO_FPS)
            idx = 0
            t0 = time.perf_counter()
            while not self.stop_event.is_set():
                expected = int((time.perf_counter() - t0) * fps)
                while idx < expected - 1 and not self.stop_event.is_set():
                    if not cap.grab():                     # too slow -> skip frames, keep in sync
                        break
                    idx += 1
                ok, frame = cap.read()
                if not ok:                                  # end of the video -> loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx, t0 = 0, time.perf_counter()
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("no frames in video")
                idx += 1
                h, w = frame.shape[:2]
                tgt = self.slot.target
                size = target_size(w, h, tgt[0], tgt[1], self.slot.mode)
                if size != (w, h):
                    interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, size, interpolation=interp)
                self.publish(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                wait = idx / fps - (time.perf_counter() - t0)
                if wait > 0:
                    self.stop_event.wait(wait)
        finally:
            cap.release()


class MediaSlot:
    """Box for an image / GIF / video taken from the prikol folder.
    mode="stretch" - the picture fills the whole box, mode="fit" - it keeps its proportions.
    Left click - next file from the folder, right click - random file."""

    POLL_MS = 20

    def __init__(self, parent, scan, on_change, mode="stretch"):
        self.scan = scan
        self.on_change = on_change
        self.mode = mode
        self.path = None
        self.worker = None
        self.pending = None
        self.error = None
        self.lock = threading.Lock()
        self.target = (96, 96)
        self.photo = None

        self.frame = tk.Frame(parent, bg=INPUT_BG, highlightthickness=1,
                              highlightbackground="#3c3c3c")
        self.label = tk.Label(self.frame, bg=INPUT_BG, fg=MUTED, font=("Segoe UI", 8),
                              cursor="hand2", bd=0)
        self.label.place(x=0, y=0, relwidth=1, relheight=1)
        for w in (self.frame, self.label):
            w.bind("<Button-1>", self.next_file)
            w.bind("<Button-3>", self.random_file)
        self.frame.bind("<Configure>", self.on_resize)
        self.show_empty()
        self.frame.after(self.POLL_MS, self.poll)

    def show_empty(self):
        self.label.config(image="", text="put images / GIFs / videos\ninto 'prikol'")

    def on_resize(self, e):
        self.target = (max(e.width - 2, 8), max(e.height - 2, 8))

    def set_file(self, path):
        if self.worker is not None:
            self.worker.stop()
        with self.lock:
            self.pending, self.error = None, None
            self.path = Path(path) if path is not None else None
            self.worker = MediaWorker(self, path) if path is not None else None
        if self.worker is None:
            self.show_empty()
            return
        self.worker.start()

    def poll(self):
        img, err = None, None
        with self.lock:
            img, self.pending = self.pending, None
            err = self.error
        if err:
            self.label.config(image="", text=err)
        elif img is not None:
            self.photo = ImageTk.PhotoImage(img)
            self.label.config(image=self.photo, text="")
        self.frame.after(self.POLL_MS, self.poll)

    def next_file(self, _=None):
        files = self.scan()
        if not files:
            self.set_file(None)
            return
        idx = files.index(self.path) if self.path in files else -1
        self.set_file(files[(idx + 1) % len(files)])
        self.on_change()

    def random_file(self, _=None):
        files = self.scan()
        if not files:
            self.set_file(None)
            return
        self.set_file(random.choice(files))
        self.on_change()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Checking")
        root.geometry("1500x720")
        root.minsize(950, 420)
        root.configure(bg=BG)

        self.header, self.rows = read_rows(CSV_PATH)
        self.original = [r[1] for r in self.rows]
        self.dirty = False
        self.photo = None
        self.idx = self.load_progress()

        # img / view state
        self.img_size = (0, 0)        # file pixels
        self._cache = None            # (path, mtime, PIL image)
        self.zoom = 1.0               # 1.0 = fit to window
        self.view_ox = None           # image offset on canvas (None = centered)
        self.view_oy = None
        self.ox = 0.0                 # current layout
        self.oy = 0.0
        self.eff_scale = 1.0
        self.pan_start = None
        self.sel_start = None         # selection is stored in IMAGE pixels
        self.sel = None               # (x0, y0, x1, y1) in image pixels
        self.sel_id = None

        # ---------- layout: [ left panel | image ]  (dividers can be dragged) ----------
        hpane = tk.PanedWindow(root, orient="horizontal", bg=BG, bd=0, sashwidth=6,
                               sashrelief="flat")
        hpane.pack(fill="both", expand=True)

        # ---------- left ----------
        left = tk.Frame(hpane, bg=PANEL, padx=12, pady=12)

        header = tk.Frame(left, bg=PANEL)
        header.pack(fill="x", pady=(0, 8))
        labels = tk.Frame(header, bg=PANEL)
        labels.pack(side="left", anchor="n")
        self.info = tk.Label(labels, text="", font=("Segoe UI", 12), bg=PANEL, fg=MUTED,
                             anchor="w")
        self.info.pack(fill="x")
        self.name = tk.Label(labels, text="", font=("Consolas", 15, "bold"), bg=PANEL, fg=FG,
                             anchor="w")
        self.name.pack(fill="x")

        # small GIF slots (right of the number / file name); files come from the "prikol" folder
        os.makedirs(PRIKOL_DIR, exist_ok=True)
        gifs = tk.Frame(header, bg=PANEL, height=GIF_HEIGHT)
        gifs.pack(side="left", fill="both", expand=True, padx=(14, 0))
        gifs.grid_propagate(False)
        gifs.rowconfigure(0, weight=1)
        self.gif_slots = []
        for col in range(GIF_SLOTS):
            gifs.columnconfigure(col, weight=1, uniform="gif")
            slot = MediaSlot(gifs, self.scan_prikol, self.save_settings, mode="stretch")
            slot.frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            self.gif_slots.append(slot)

        # vertical divider: [ text + buttons ] / [ status + hints + big field ]
        vpane = tk.PanedWindow(left, orient="vertical", bg=PANEL, bd=0, sashwidth=6,
                               sashrelief="flat")
        vpane.pack(fill="both", expand=True)

        top = tk.Frame(vpane, bg=PANEL)
        self.text = tk.Text(top, width=40, height=8, wrap="word",
                            font=("Segoe UI", 20), undo=True, bg=INPUT_BG, fg=FG,
                            insertbackground=FG, selectbackground=ACCENT,
                            selectforeground="white", relief="flat", bd=0,
                            padx=10, pady=10, highlightthickness=1,
                            highlightbackground="#3c3c3c", highlightcolor=ACCENT)
        self.text.pack(fill="both", expand=True)

        btns = tk.Frame(top, bg=PANEL)
        btns.pack(fill="x", pady=(10, 0))
        make_button(btns, "Crop (Ctrl+R)", self.apply_crop).pack(side="left", padx=(0, 6))
        make_button(btns, "Original (Ctrl+B)", self.restore_original).pack(side="left", padx=(0, 6))
        make_button(btns, "Deselect (Esc)", self.clear_selection).pack(side="left")

        bottom = tk.Frame(vpane, bg=PANEL)
        notes = tk.Frame(bottom, bg=PANEL)
        notes.pack(side="left", fill="y", anchor="n", pady=(8, 0))
        self.status = make_copyable(notes, "", ("Segoe UI", 10), MUTED, 2, width=58)
        self.status.pack(fill="x")
        self.hints = make_copyable(
            notes,
            "Enter - continue  Spec - ¤  Alt+←/→ - back/forward\n"
            "PgUp/PgDn - ±10    Ctrl+F - search    Ctrl+S - save\n"
            "Ctrl+Wheel - zoom    Ctrl+0 - reset zoom    Wheel / RMB drag - pan\n"
            "Media slots (folder prikol: images, GIFs, videos): click - next, right click - random\n"
            "Drag the dividers between the panels to resize them",
            ("Segoe UI", 9), MUTED, 6, width=58)
        self.hints.pack(fill="x", pady=(6, 0))
        make_button(notes, "Commit changes to Git", self.commit_to_git).pack(
            fill="x", pady=(8, 0)
        )

        # big field: works like the small slots (stretches to the whole field)
        self.big_slot = MediaSlot(bottom, self.scan_prikol, self.save_settings, mode=BIG_MODE)
        self.big_slot.frame.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(8, 0))

        vpane.add(top, stretch="always", minsize=200)
        vpane.add(bottom, stretch="never", minsize=120, height=BIG_HEIGHT)
        hpane.add(left, stretch="never", minsize=560, width=LEFT_WIDTH)
        self.load_settings()

        # ---------- right panel ----------
        right = tk.Frame(hpane, bg=CANVAS_BG)
        hpane.add(right, stretch="always", minsize=300)
        self.canvas = tk.Canvas(right, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.show_image())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        # zoom (Ctrl + wheel) and pan (wheel / Shift+wheel / middle or right button drag)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom_wheel)
        self.canvas.bind("<Control-Button-4>", lambda e: self.zoom_at(e.x, e.y, ZOOM_STEP))
        self.canvas.bind("<Control-Button-5>", lambda e: self.zoom_at(e.x, e.y, 1 / ZOOM_STEP))
        self.canvas.bind("<MouseWheel>", self.on_wheel_pan)
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.on_wheel_pan(e, horizontal=True))
        self.canvas.bind("<Button-4>", lambda e: self.pan_by(0, 60))
        self.canvas.bind("<Button-5>", lambda e: self.pan_by(0, -60))
        for b in (2, 3):
            self.canvas.bind(f"<ButtonPress-{b}>", self.on_pan_press)
            self.canvas.bind(f"<B{b}-Motion>", self.on_pan_drag)

        # ---------- keys ----------
        self.text.bind("<Return>", self.on_enter)
        root.bind("<Return>", self.on_enter)
        for w in (self.text, root):
            w.bind("<Alt-Left>", lambda e: self.go(-1))
            w.bind("<Alt-Right>", lambda e: self.go(1))
            w.bind("<Control-Left>", lambda e: self.go(-1))
            w.bind("<Control-Right>", lambda e: self.go(1))
            w.bind("<Prior>", lambda e: self.go(-10))
            w.bind("<Next>", lambda e: self.go(10))
            w.bind("<Control-f>", self.goto)
            w.bind("<Control-s>", self.on_save_key)
            w.bind("<Control-r>", self.on_crop_key)
            w.bind("<Control-b>", self.on_restore_key)
            w.bind("<Escape>", self.on_escape)
            w.bind("<Control-Key-0>", self.on_zoom_reset_key)
            w.bind("<Control-equal>", lambda e: self.zoom_center(ZOOM_STEP))
            w.bind("<Control-plus>", lambda e: self.zoom_center(ZOOM_STEP))
            w.bind("<Control-minus>", lambda e: self.zoom_center(1 / ZOOM_STEP))
        self.text.bind("<Control-z>", self.undo_line)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.load_current()

    # ---------- status ----------
    def set_status(self, text):
        set_readonly_text(self.status, text)

    # ---------- settings (GIF slots) ----------
    def scan_prikol(self):
        try:
            return sorted(p for p in Path(PRIKOL_DIR).iterdir()
                          if p.is_file() and p.suffix.lower() in MEDIA_EXTS)
        except Exception:
            return []

    def load_settings(self):
        saved, saved_big = [], None
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            saved, saved_big = data.get("gifs", []), data.get("big")
        except Exception:
            pass
        files = self.scan_prikol()
        by_name = {p.name: p for p in files}
        for i, slot in enumerate(self.gif_slots):
            name = Path(saved[i]).name if i < len(saved) and saved[i] else None
            path = by_name.get(name)
            if path is None and files:
                path = files[i % len(files)]
            slot.set_file(path)
        path = by_name.get(Path(saved_big).name) if saved_big else None
        if path is None and files:
            path = files[len(self.gif_slots) % len(files)]
        self.big_slot.set_file(path)

    def save_settings(self):
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({"gifs": [s.path.name if s.path else None for s in self.gif_slots],
                           "big": self.big_slot.path.name if self.big_slot.path else None},
                          f, ensure_ascii=False)
        except Exception:
            pass
        self.text.focus_set()

    # ---------- data ----------
    def load_progress(self):
        try:
            with open(PROGRESS_PATH, encoding="utf-8") as f:
                i = int(f.read().strip())
            return min(max(i, 0), len(self.rows) - 1)
        except Exception:
            return 0

    def save_progress(self):
        try:
            with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
                f.write(str(self.idx))
        except Exception:
            pass

    def commit_text(self):
        new = self.text.get("1.0", "end-1c").replace("\n", " ").strip()
        if new != self.rows[self.idx][1]:
            self.rows[self.idx][1] = new
            self.dirty = True

    def save(self):
        self.commit_text()
        if not self.dirty:
            self.set_status("No changes")
            return
        bak = str(CSV_PATH) + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(CSV_PATH, bak)
        tmp = str(CSV_PATH) + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            w.writerows(self.rows)
        os.replace(tmp, CSV_PATH)
        self.dirty = False
        changed = sum(1 for a, r in zip(self.original, self.rows) if a != r[1])
        self.set_status(f"Saved. Changed lines: {changed}")

    # ---------- show ----------
    def current_path(self):
        return os.path.join(FOLDER, self.rows[self.idx][0])

    def reset_view(self):
        self.zoom = 1.0
        self.view_ox = None
        self.view_oy = None

    def load_current(self):
        fn, txt = self.rows[self.idx]
        self.info.config(text=f"{self.idx + 1} / {len(self.rows)}")
        cropped = os.path.exists(os.path.join(ORIG_DIR, fn))
        self.name.config(text=fn + ("   [cutted]" if cropped else ""))
        self.text.delete("1.0", "end")
        self.text.insert("1.0", txt)
        self.text.edit_reset()
        self.text.focus_set()
        self.text.mark_set("insert", "end")
        self.sel = None
        self.sel_start = None
        self.reset_view()
        self.show_image()
        self.save_progress()

    def get_image(self):
        path = self.current_path()
        if not os.path.exists(path):
            return None
        mtime = os.path.getmtime(path)
        if self._cache and self._cache[0] == path and self._cache[1] == mtime:
            return self._cache[2]
        im = Image.open(path).convert("RGB")
        self._cache = (path, mtime, im)
        return im

    def layout(self):
        """Compute image position / scale on the canvas for the current zoom."""
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        W, H = self.img_size
        base = max(min((cw - 20) / W, (ch - 20) / H, 3.0), 0.01)
        s = base * self.zoom
        w, h = W * s, H * s
        if w <= cw or self.view_ox is None:
            ox = (cw - w) / 2
        else:
            ox = min(0.0, max(cw - w, self.view_ox))
        if h <= ch or self.view_oy is None:
            oy = (ch - h) / 2
        else:
            oy = min(0.0, max(ch - h, self.view_oy))
        self.view_ox = ox if w > cw else None
        self.view_oy = oy if h > ch else None
        self.ox, self.oy, self.eff_scale = ox, oy, s
        return cw, ch

    def show_image(self):
        self.canvas.delete("all")
        self.sel_id = None
        im = self.get_image()
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        if im is None:
            self.img_size = (0, 0)
            self.canvas.create_text(cw // 2, ch // 2, text=f"No file:\n{self.current_path()}",
                                    font=("Segoe UI", 14), fill="#ff6b6b")
            return
        self.img_size = im.size
        W, H = im.size
        cw, ch = self.layout()
        s, ox, oy = self.eff_scale, self.ox, self.oy
        # render only the visible part of the image
        xi0 = max(0, int(-ox / s))
        yi0 = max(0, int(-oy / s))
        xi1 = min(W, int((cw - ox) / s) + 2)
        yi1 = min(H, int((ch - oy) / s) + 2)
        if xi1 > xi0 and yi1 > yi0:
            part = im.crop((xi0, yi0, xi1, yi1))
            size = (max(1, int(round((xi1 - xi0) * s))), max(1, int(round((yi1 - yi0) * s))))
            part = part.resize(size, RESAMPLE)
            self.photo = ImageTk.PhotoImage(part)
            self.canvas.create_image(int(round(ox + xi0 * s)), int(round(oy + yi0 * s)),
                                     image=self.photo, anchor="nw")
        self.draw_selection()

    # ---------- zoom / pan ----------
    def zoom_at(self, cx, cy, factor):
        if self.img_size[0] == 0:
            return "break"
        self.layout()
        s_old = self.eff_scale
        px, py = (cx - self.ox) / s_old, (cy - self.oy) / s_old
        new_zoom = min(ZOOM_MAX, max(1.0, self.zoom * factor))
        if new_zoom == self.zoom:
            return "break"
        s_new = s_old * new_zoom / self.zoom
        self.zoom = new_zoom
        self.view_ox = cx - px * s_new
        self.view_oy = cy - py * s_new
        self.show_image()
        self.set_status(f"Zoom: x{self.zoom:.2f}" if self.zoom > 1.0 else "Zoom: fit")
        return "break"

    def zoom_center(self, factor):
        self.zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, factor)
        return "break"

    def on_zoom_wheel(self, e):
        return self.zoom_at(e.x, e.y, ZOOM_STEP if e.delta > 0 else 1 / ZOOM_STEP)

    def on_zoom_reset_key(self, e):
        self.reset_view()
        self.show_image()
        self.set_status("Zoom: fit")
        return "break"

    def pan_by(self, dx, dy):
        if self.zoom <= 1.0 or self.img_size[0] == 0:
            return "break"
        self.layout()
        self.view_ox = self.ox + dx
        self.view_oy = self.oy + dy
        self.show_image()
        return "break"

    def on_wheel_pan(self, e, horizontal=False):
        step = 80 if e.delta > 0 else -80
        return self.pan_by(step, 0) if horizontal else self.pan_by(0, step)

    def on_pan_press(self, e):
        if self.zoom > 1.0 and self.img_size[0]:
            self.layout()
            self.pan_start = (e.x, e.y, self.ox, self.oy)
            self.canvas.config(cursor="fleur")

    def on_pan_drag(self, e):
        if self.pan_start is None:
            return
        sx, sy, ox0, oy0 = self.pan_start
        self.view_ox = ox0 + (e.x - sx)
        self.view_oy = oy0 + (e.y - sy)
        self.show_image()

    # ---------- selection / crop ----------
    def to_img(self, x, y):
        W, H = self.img_size
        ix = min(max((x - self.ox) / self.eff_scale, 0), W)
        iy = min(max((y - self.oy) / self.eff_scale, 0), H)
        return ix, iy

    def on_press(self, e):
        if self.img_size[0] == 0:
            return
        self.sel_start = self.to_img(e.x, e.y)
        self.sel = None
        self.draw_selection()

    def on_drag(self, e):
        if self.sel_start is None:
            return
        x, y = self.to_img(e.x, e.y)
        sx, sy = self.sel_start
        self.sel = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
        self.draw_selection()

    def on_release(self, e):
        self.sel_start = None
        if self.pan_start is not None:
            self.pan_start = None
            self.canvas.config(cursor="crosshair")
        s = self.eff_scale
        if self.sel and ((self.sel[2] - self.sel[0]) * s < 4 or (self.sel[3] - self.sel[1]) * s < 4):
            self.sel = None
        self.draw_selection()
        if self.sel:
            self.set_status("Highlighted. Ctrl+R - cut, Esc - deselect")

    def draw_selection(self):
        if self.sel_id is not None:
            self.canvas.delete(self.sel_id)
            self.sel_id = None
        if self.sel:
            s = self.eff_scale
            x0, y0, x1, y1 = self.sel
            self.sel_id = self.canvas.create_rectangle(
                self.ox + x0 * s, self.oy + y0 * s, self.ox + x1 * s, self.oy + y1 * s,
                outline=SEL_COLOR, width=2, dash=(5, 3))

    def clear_selection(self):
        self.sel = None
        self.sel_start = None
        self.draw_selection()

    def apply_crop(self):
        if not self.sel:
            self.set_status("First, select an area on the image using your mouse.")
            return
        path = self.current_path()
        if not os.path.exists(path):
            return
        W, H = self.img_size
        x0 = max(0, int(round(self.sel[0])))
        y0 = max(0, int(round(self.sel[1])))
        x1 = min(W, int(round(self.sel[2])))
        y1 = min(H, int(round(self.sel[3])))
        if x1 - x0 < 5 or y1 - y0 < 5:
            self.set_status("Area too small")
            return
        os.makedirs(ORIG_DIR, exist_ok=True)
        backup = os.path.join(ORIG_DIR, self.rows[self.idx][0])
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        im = Image.open(path)
        im.crop((x0, y0, x1, y1)).save(path)
        self.set_status(f"Cropped: {x1 - x0}x{y1 - y0} px (original in _originals)")
        self.sel = None
        self.reset_view()
        self.name.config(text=self.rows[self.idx][0] + "   [cropped]")
        self.show_image()

    def restore_original(self):
        fn = self.rows[self.idx][0]
        backup = os.path.join(ORIG_DIR, fn)
        if not os.path.exists(backup):
            self.set_status("There is no saved original for this image")
            return
        shutil.copy2(backup, self.current_path())
        os.remove(backup)
        self.name.config(text=fn)
        self.sel = None
        self.reset_view()
        self.show_image()
        self.set_status("Original restored")


    # ---------- Git commit ----------
    def commit_to_git(self):
        """Save the current labels and image changes, then commit them to Git."""
        self.commit_text()
        if self.dirty:
            try:
                self.save()
            except Exception as ex:
                messagebox.showerror("Git commit", f"Could not save labels.csv:\n{ex}")
                return

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(BASE),
                capture_output=True,
                text=True,
                check=True,
            )
            repo_root = Path(result.stdout.strip()).resolve()

            # Stage labels.csv and dataset images, but never temporary UI files
            # or the _originals crop-backup directory.
            files_to_stage = [CSV_PATH.resolve()]
            if FOLDER.exists():
                for path in FOLDER.rglob("*"):
                    if not path.is_file():
                        continue
                    relative_parts = path.relative_to(FOLDER).parts
                    if "_originals" in relative_parts:
                        continue
                    if path.suffix.lower() in IMAGE_EXTS:
                        files_to_stage.append(path.resolve())

            relative_paths = []
            for path in files_to_stage:
                try:
                    relative_paths.append(str(path.relative_to(repo_root)))
                except ValueError:
                    raise RuntimeError(f"File is outside the Git repository: {path}")

            subprocess.run(
                ["git", "add", "--", *relative_paths],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=True,
            )

            changed = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            if changed.returncode == 0:
                self.set_status("Git: no staged changes to commit")
                return
            if changed.returncode != 1:
                raise RuntimeError(changed.stderr or "Could not check staged changes")

            # Find the most recent real-data commit number and increment it.
            history = subprocess.run(
                ["git", "log", "--all", "--format=%s", "-100"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=True,
            )
            version = REAL_DATA_VERSION_START
            for subject in history.stdout.splitlines():
                match = re.match(r"^real data 0\.(\d+)(?:\s|$)", subject, re.IGNORECASE)
                if match:
                    version = max(version, int(match.group(1)))
                    break
            version += 1

            current_file = self.rows[self.idx][0]
            commit_message = f"real data 0.{version} checked till {current_file}"
            commit = subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            if commit.returncode != 0:
                details = (commit.stderr or commit.stdout).strip()
                raise RuntimeError(details or "Git commit failed")

            self.set_status(f"Committed: {commit_message}")
        except FileNotFoundError:
            messagebox.showerror(
                "Git commit",
                "Git was not found. Install Git and make sure it is available in PATH."
            )
        except subprocess.CalledProcessError as ex:
            details = (ex.stderr or ex.stdout or str(ex)).strip()
            messagebox.showerror("Git commit", details)
        except Exception as ex:
            messagebox.showerror("Git commit", str(ex))

    # ---------- key handlers ----------
    def on_save_key(self, e):
        self.save()
        return "break"

    def on_crop_key(self, e):
        self.apply_crop()
        return "break"

    def on_restore_key(self, e):
        self.restore_original()
        return "break"

    def on_escape(self, e):
        self.clear_selection()
        return "break"

    def go(self, step):
        self.commit_text()
        new = min(max(self.idx + step, 0), len(self.rows) - 1)
        if new != self.idx:
            self.idx = new
            self.load_current()
        return "break"

    def on_enter(self, event):
        self.commit_text()
        if self.dirty:
            self.save()
        if self.idx >= len(self.rows) - 1:
            self.set_status("This is the last line")
        else:
            self.idx += 1
            self.load_current()
        return "break"

    def undo_line(self, event):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.original[self.idx])
        return "break"

    def goto(self, event):
        win = tk.Toplevel(self.root, bg=PANEL, padx=16, pady=14)
        win.title("Go to")
        win.transient(self.root)
        win.resizable(False, False)
        tk.Label(win, text="Line number or file name (e.g., 6767):",
                 bg=PANEL, fg=FG, font=("Segoe UI", 11)).pack(anchor="w")
        var = tk.StringVar()
        ent = tk.Entry(win, textvariable=var, font=("Segoe UI", 14), bg=INPUT_BG, fg=FG,
                       insertbackground=FG, relief="flat", width=24)
        ent.pack(fill="x", pady=8, ipady=4)
        ent.focus_set()

        def go_now(_=None):
            q = var.get().strip()
            win.destroy()
            if not q:
                return
            target = None
            for i, (fn, _t) in enumerate(self.rows):
                if fn == q or fn.startswith(q.zfill(4)):
                    target = i
                    break
            if target is None and q.isdigit() and 1 <= int(q) <= len(self.rows):
                target = int(q) - 1
            if target is None:
                self.set_status("String not found")
            else:
                self.commit_text()
                self.idx = target
                self.load_current()

        ent.bind("<Return>", go_now)
        win.bind("<Escape>", lambda e: win.destroy())
        make_button(win, "Go to", go_now).pack(anchor="e")
        win.grab_set()
        return "break"

    def on_close(self):
        self.commit_text()
        if self.dirty:
            self.save()
        self.save_progress()
        self.root.destroy()


def main():
    if not os.path.exists(CSV_PATH):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"File not found:\n{CSV_PATH}")
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
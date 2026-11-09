import csv
import json
import os
import random
import shutil
import threading
import time
import subprocess
import re
import math
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFilter
from pathlib import Path

BASE = Path(__file__).resolve().parent
FOLDER = BASE / "data" / "real_dataset" / "images"
FOLDER_CSV = BASE / "data" / "real_dataset"
CSV_PATH = FOLDER_CSV / "labels.csv"

PROGRESS_PATH = os.path.join(FOLDER, ".verify_progress")
SETTINGS_PATH = os.path.join(FOLDER, ".verify_settings.json")
ORIG_DIR = os.path.join(FOLDER, "_originals")
PRIKOL_DIR = BASE / "prikol"   # GIFs for the slots are taken from this folder

# Cosmic Purple theme
BG = "#130924"
PANEL = "#1d1035"
PANEL_BORDER = "#4a2b82"
FG = "#f3e8ff"
MUTED = "#a78bfa"
INPUT_BG = "#261546"
ACCENT = "#8b5cf6"
ACCENT_HOVER = "#9333ea"
BTN_BG = "#2e1854"
BTN_ACTIVE = "#5b21b6"
CANVAS_BG = "#0b0518"
SEL_COLOR = "#c084fc"
OK_COLOR = "#4ade80"
ERR_COLOR = "#f87171"

GIF_SLOTS = 3                      # number of small media slots
GIF_HEIGHT = 96                    # height of the small slots row
BIG_HEIGHT = 220                   # initial height of the big field
BIG_MODE = "stretch"               # "stretch" or "fit"
LEFT_WIDTH = 780                   # initial width of the left panel
STATIC_MAX = 2500                  # downscale max size
MAX_VIDEO_FPS = 60

IMAGE_EXTS = {".gif", ".webp", ".apng", ".png", ".jpg", ".jpeg", ".jfif", ".bmp",
              ".tif", ".tiff", ".ico", ".ppm", ".pgm"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpg",
              ".mpeg", ".mpe", ".3gp", ".ogv", ".ts", ".mts", ".m2ts", ".vob", ".asf"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
RESAMPLE_FAST = getattr(Image, "Resampling", Image).BILINEAR

# Zoom settings
ZOOM_STEP = 1.2
ZOOM_MAX = 10.0

# Starfield settings
STAR_COUNT = 260
STAR_TWINKLE_SHADES = 8

# Git commit settings
REAL_DATA_VERSION_START = 8


def _round_polygon_points(x0, y0, x1, y1, r, steps=24):
    """Flat list of polygon points approximating a rounded rectangle."""
    r = max(1, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = []
    corners = ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90),
               (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270))
    for cx, cy, a0 in corners:
        for i in range(steps + 1):
            a = math.radians(a0 + 90.0 * i / steps)
            pts.extend((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


class RoundedFrame(tk.Frame):
    """Frame with rounded corners and outline."""

    def __init__(self, master, radius=14, fill=PANEL, outline=PANEL_BORDER, ow=1, **kw):
        try:
            bg = kw.pop("bg", master.cget("bg"))
        except Exception:
            bg = kw.pop("bg", BG)
        super().__init__(master, bg=bg, **kw)
        self._radius, self._fill, self._outline, self._ow = radius, fill, outline, ow
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self._canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._rect = self._canvas.create_polygon(0, 0, fill=fill, outline=outline, width=ow)
        
        self.inner = tk.Frame(self._canvas, bg=fill, bd=0, highlightthickness=0)
        self._win = self._canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self._canvas.tag_lower(self._rect)
        self._canvas.bind("<Configure>", self._redraw)

    def set_outline(self, color):
        self._outline = color
        self._canvas.itemconfig(self._rect, outline=color)

    def _redraw(self, e):
        w, h = max(e.width, 4), max(e.height, 4)
        r = max(1, min(self._radius, (w - 2) // 2, (h - 2) // 2))
        self._canvas.coords(self._rect, _round_polygon_points(1, 1, w - 1, h - 1, r))
        m = self._ow + max(1, int(r - r / math.sqrt(2)))
        self._canvas.coords(self._win, m, m)
        self._canvas.itemconfigure(self._win, width=max(w - 2 * m, 1), height=max(h - 2 * m, 1))


class RoundedButton(tk.Canvas):
    """Button with rounded corners."""

    def __init__(self, master, text, command, fill=BTN_BG, hover=BTN_ACTIVE,
                 fg=FG, radius=12, font=("Segoe UI", 10, "bold")):
        try:
            bg = master.cget("bg")
        except Exception:
            bg = BG
        self._font = tkfont.Font(font=font)
        tw = self._font.measure(text)
        th = self._font.metrics("linespace")
        w, h = tw + 30, th + 16
        super().__init__(master, width=w, height=h, bg=bg, highlightthickness=0,
                         bd=0, cursor="hand2")
        self._command = command
        self._fill, self._hover, self._fg, self._radius = fill, hover, fg, radius
        self._rect = self.create_polygon(0, 0, fill=fill, outline=PANEL_BORDER, width=1)
        self._txt = self.create_text(0, 0, text=text, font=self._font, fill=fg)
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", lambda _e: self.itemconfig(self._rect, fill=self._hover))
        self.bind("<Leave>", lambda _e: self.itemconfig(self._rect, fill=self._fill))
        self.bind("<Button-1>", lambda _e: self._command())
        self._redraw(None)

    def _redraw(self, _e):
        try:
            w = self.winfo_width()
            h = self.winfo_height()
            if w <= 2:
                w = int(self.cget("width"))
            if h <= 2:
                h = int(self.cget("height"))
        except Exception:
            return
        r = max(1, min(self._radius, (h - 2) // 2))
        self.coords(self._rect, _round_polygon_points(1, 1, w - 1, h - 1, r))
        self.coords(self._txt, w / 2, h / 2)


def find_git():
    found = shutil.which("git")
    if found and os.path.isfile(found):
        return found
    candidates = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\bin\git.exe",
    ]
    for env_name in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name)
        if base:
            candidates.extend([
                os.path.join(base, "Programs", "Git", "cmd", "git.exe"),
                os.path.join(base, "Git", "cmd", "git.exe"),
            ])
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


GIT_EXE = find_git()


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
    return RoundedButton(parent, text, command)


def make_copyable(parent, text, font, fg, height, width=80):
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

    def run_image(self):
        bgcol = (45, 45, 48, 255)
        with Image.open(self.path) as im:
            n = getattr(im, "n_frames", 1)
            if n <= 1:
                try:
                    im = ImageOps.exif_transpose(im)
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
                        self.stop_event.wait(0.1)
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
                    next_time = time.perf_counter()
                i = (i + 1) % n

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
                    if not cap.grab():
                        break
                    idx += 1
                ok, frame = cap.read()
                if not ok:
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

        self.frame = RoundedFrame(parent, radius=10, fill=INPUT_BG, outline=PANEL_BORDER)
        self.label = tk.Label(self.frame.inner, bg=INPUT_BG, fg=MUTED, font=("Segoe UI", 8),
                              cursor="hand2", bd=0)
        self.label.place(x=0, y=0, relwidth=1, relheight=1)
        for w in (self.frame, self.frame.inner, self.label):
            w.bind("<Button-1>", self.next_file)
            w.bind("<Button-3>", self.random_file)
        self.frame.bind("<Enter>", lambda _e: self.frame.set_outline(ACCENT))
        self.frame.bind("<Leave>", lambda _e: self.frame.set_outline(PANEL_BORDER))
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
        root.title("🌌 Checking — Cosmic Purple Edition")
        root.geometry("1500x720")
        root.minsize(950, 420)
        root.configure(bg=BG)

        self.header, self.rows = read_rows(CSV_PATH)
        self.original = [r[1] for r in self.rows]
        self.dirty = False
        self.photo = None
        self.idx = self.load_progress()

        # Starfield animation state
        self.stars = []
        self.star_phase = 0.0
        self.space_base = None
        self.space_photo = None
        self._bg_key = None
        self.meteors = []
        self.next_meteor = time.perf_counter() + random.uniform(4, 10)

        # img / view state
        self.img_size = (0, 0)
        self._cache = None
        self.zoom = 1.0
        self.view_ox = None
        self.view_oy = None
        self.ox = 0.0
        self.oy = 0.0
        self.eff_scale = 1.0
        self.pan_start = None
        self.sel_start = None
        self.sel = None
        self.sel_id = None

        # ---------- Layout ----------
        hpane = tk.PanedWindow(root, orient="horizontal", bg=BG, bd=0, sashwidth=6, sashrelief="flat")
        hpane.pack(fill="both", expand=True)

        left_outer = RoundedFrame(hpane, radius=18, fill=PANEL, outline=PANEL_BORDER, bg=BG)
        left = left_outer.inner
        left.config(padx=10, pady=10)

        header = tk.Frame(left, bg=PANEL)
        header.pack(fill="x", pady=(0, 6))
        
        labels = tk.Frame(header, bg=PANEL)
        labels.pack(side="left", anchor="n")
        self.info = tk.Label(labels, text="", font=("Segoe UI", 12), bg=PANEL, fg=MUTED, anchor="w")
        self.info.pack(fill="x")
        self.name = tk.Label(labels, text="", font=("Consolas", 15, "bold"), bg=PANEL, fg=FG, anchor="w")
        self.name.pack(fill="x")
        tk.Frame(labels, bg=ACCENT, height=2).pack(fill="x", pady=(4, 0))

        os.makedirs(PRIKOL_DIR, exist_ok=True)
        gifs = tk.Frame(header, bg=PANEL, height=GIF_HEIGHT)
        gifs.pack(side="left", fill="both", expand=True, padx=(10, 0))
        gifs.grid_propagate(False)
        gifs.rowconfigure(0, weight=1)
        self.gif_slots = []
        for col in range(GIF_SLOTS):
            gifs.columnconfigure(col, weight=1, uniform="gif")
            slot = MediaSlot(gifs, self.scan_prikol, self.save_settings, mode="stretch")
            slot.frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            self.gif_slots.append(slot)

        vpane = tk.PanedWindow(left, orient="vertical", bg=PANEL, bd=0, sashwidth=6, sashrelief="flat")
        vpane.pack(fill="both", expand=True)

        top = tk.Frame(vpane, bg=PANEL)
        text_wrap = RoundedFrame(top, radius=12, fill=INPUT_BG, outline=PANEL_BORDER, bg=PANEL, ow=1)
        text_wrap.pack(fill="both", expand=True)
        text_wrap.inner.config(padx=0, pady=0)
        self.text = tk.Text(text_wrap.inner, width=40, height=6, wrap="word",
                            font=("Segoe UI", 20), undo=True, bg=INPUT_BG, fg=FG,
                            insertbackground=FG, selectbackground=ACCENT,
                            selectforeground="white", relief="flat", bd=0,
                            padx=10, pady=10, highlightthickness=0)
        self.text.pack(fill="both", expand=True)

        btns = tk.Frame(top, bg=PANEL)
        btns.pack(fill="x", pady=(8, 0))
        make_button(btns, "Crop (Ctrl+R)", self.apply_crop).pack(side="left", padx=(0, 6))
        make_button(btns, "Original (Ctrl+B)", self.restore_original).pack(side="left", padx=(0, 6))
        make_button(btns, "Deselect (Esc)", self.clear_selection).pack(side="left")
        make_button(btns, "🗑 Delete", self.delete_current).pack(side="right")

        bottom = tk.Frame(vpane, bg=PANEL)
        notes = tk.Frame(bottom, bg=PANEL)
        notes.pack(side="left", fill="both", expand=True, anchor="n", pady=(6, 0))
        self.status = make_copyable(notes, "", ("Segoe UI", 10), MUTED, 2, width=40)
        self.status.pack(fill="x")
        self.hints = make_copyable(
            notes,
            "Enter - continue  Spec - ¤  Alt+←/→ - back/forward\n"
            "PgUp/PgDn - ±10    Ctrl+F - search    Ctrl+S - save\n"
            "Ctrl+Wheel - zoom    Ctrl+0 - reset zoom    Wheel / RMB drag - pan\n"
            "🗑 Delete (right button) - deletes line AND file\n"
            "Media slots: click - next, right click - random",
            ("Segoe UI", 9), MUTED, 6, width=40)
        self.hints.pack(fill="both", expand=True, pady=(4, 0))
        make_button(notes, "💜 Commit & Push to Git", self.commit_to_git).pack(fill="x", pady=(6, 0))

        self.big_slot = MediaSlot(bottom, self.scan_prikol, self.save_settings, mode=BIG_MODE)
        self.big_slot.frame.pack(side="right", fill="both", expand=True, padx=(8, 0), pady=(6, 0))

        vpane.add(top, stretch="always", minsize=150)
        vpane.add(bottom, stretch="never", minsize=120, height=BIG_HEIGHT)
        hpane.add(left_outer, stretch="never", minsize=450, width=LEFT_WIDTH)
        self.load_settings()

        right_outer = RoundedFrame(hpane, radius=18, fill=CANVAS_BG, outline=PANEL_BORDER, bg=BG)
        hpane.add(right_outer, stretch="always", minsize=300)
        right = right_outer.inner
        self.canvas = tk.Canvas(right, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.show_image())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
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

        self.init_stars()
        self.animate_stars()
        self.load_current()

    @staticmethod
    def _shade(rgb, k):
        k = 0.15 + 0.85 * k
        return "#%02x%02x%02x" % tuple(int(c * k) for c in rgb)

    def init_stars(self):
        self.stars = []
        for _ in range(STAR_COUNT):
            tint = random.choice([(255, 255, 255), (255, 244, 255), (233, 213, 255),
                                  (196, 132, 252), (214, 226, 255)])
            self.stars.append({
                "x": random.random(),
                "y": random.random(),
                "r": random.uniform(0.5, 1.7),
                "speed": random.uniform(0.5, 2.4),
                "phase": random.uniform(0, 2 * math.pi),
                "tint": tint,
                "glow": random.random() < 0.10,
                "shades": [self._shade(tint, i / (STAR_TWINKLE_SHADES - 1))
                           for i in range(STAR_TWINKLE_SHADES)],
            })

    def render_space_base(self, w, h):
        grad = Image.linear_gradient("L").resize((w, h))
        top = Image.new("RGB", (w, h), (27, 12, 48))
        bottom = Image.new("RGB", (w, h), (6, 3, 14))
        base = Image.composite(bottom, top, grad)

        rng = random.Random((w << 16) | h)
        neb = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        nd = ImageDraw.Draw(neb)
        palette = [((124, 58, 237), 55), ((76, 29, 149), 65), ((192, 132, 252), 32),
                   ((49, 18, 92), 72), ((147, 51, 234), 38)]
        for _ in range(max(7, (w * h) // 85000)):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            rx = rng.uniform(0.08, 0.30) * w
            ry = rx * rng.uniform(0.35, 0.85)
            col, amax = palette[rng.randrange(len(palette))]
            nd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col + (rng.randint(amax // 3, amax),))
        neb = neb.filter(ImageFilter.GaussianBlur(min(w, h) // 8))
        base = Image.alpha_composite(base.convert("RGBA"), neb).convert("RGB")

        self.space_base = (w, h, base)
        self.space_photo = ImageTk.PhotoImage(base)

    def draw_starfield(self, width, height):
        self.canvas.delete("stars")
        s = max(0.6, min(width, height) / 800.0)

        # Определяем координаты отображаемой картинки на холсте (если она есть)
        img_box = None
        if self.img_size[0] > 0 and self.img_size[1] > 0:
            W, H = self.img_size
            img_w = W * self.eff_scale
            img_h = H * self.eff_scale
            # Добавим небольшой отступ (например, 4 пикселя), чтобы звёзды не липли вплотную к краям картинки
            pad = 4
            img_box = (self.ox - pad, self.oy - pad, self.ox + img_w + pad, self.oy + img_h + pad)

        for st in self.stars:
            x, y = st["x"] * width, st["y"] * height

            # Если звезда попадает внутрь области картинки — пропускаем её отрисовку
            if img_box and img_box[0] <= x <= img_box[2] and img_box[1] <= y <= img_box[3]:
                continue

            r = max(0.6, st["r"] * s)
            k = (math.sin(self.star_phase * st["speed"] + st["phase"]) + 1) / 2
            idx = int(k * (STAR_TWINKLE_SHADES - 1))
            if st["glow"]:
                gr = r * 3.4
                self.canvas.create_oval(x - gr, y - gr, x + gr, y + gr,
                                        fill=st["shades"][max(0, idx - 4)],
                                        outline="", tags="stars")
            self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                    fill=st["shades"][idx], outline="", tags="stars")

        for m in self.meteors:
            x0 = m["x"] - m["vx"] * 0.45
            y0 = m["y"] - m["vy"] * 0.45
            # Падающие звезды тоже не рисуем поверх картинки
            if img_box and img_box[0] <= m["x"] <= img_box[2] and img_box[1] <= m["y"] <= img_box[3]:
                continue
            fade = min(1.0, m["life"] * 2)
            self.canvas.create_line(m["x"], m["y"], x0, y0,
                                    fill="#7e22ce", width=2,
                                    stipple="gray50" if fade < 0.8 else "",
                                    tags="stars")
            self.canvas.create_line(m["x"], m["y"], (m["x"] + x0) / 2, (m["y"] + y0) / 2,
                                    fill="#f3e8ff", width=2, tags="stars")

    def animate_stars(self):
        self.star_phase += 0.12
        now = time.perf_counter()
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        if now > self.next_meteor and len(self.meteors) < 2:
            self.meteors.append({
                "x": random.uniform(0.25, 1.05) * cw,
                "y": random.uniform(0.0, 0.35) * ch,
                "vx": -random.uniform(0.10, 0.22) * cw,
                "vy": random.uniform(0.05, 0.12) * ch,
                "life": random.uniform(0.6, 1.2),
            })
            self.next_meteor = now + random.uniform(6, 15)
        dt = 0.08
        for m in self.meteors:
            m["x"] += m["vx"] * dt
            m["y"] += m["vy"] * dt
            m["life"] -= dt
        self.meteors = [m for m in self.meteors
                        if m["life"] > 0 and m["x"] > -100 and m["y"] < ch + 100]
        if self.canvas.winfo_exists():
            self.draw_starfield(cw, ch)
            self.root.after(80, self.animate_stars)

    def set_status(self, text, color=None):
        set_readonly_text(self.status, text)
        self.status.config(fg=color if color else MUTED)

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
        if not self.rows:
            return
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
        self.set_status(f"Saved. Changed lines: {changed}", OK_COLOR)

    def current_path(self):
        if not self.rows:
            return os.path.join(FOLDER, "")
        return os.path.join(FOLDER, self.rows[self.idx][0])

    def reset_view(self):
        self.zoom = 1.0
        self.view_ox = None
        self.view_oy = None

    def load_current(self):
        if not self.rows:
            self.info.config(text="0 / 0")
            self.name.config(text="—")
            self.text.delete("1.0", "end")
            self.text.edit_reset()
            self.sel = None
            self.sel_start = None
            self.reset_view()
            self.show_image()
            self.save_progress()
            return
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
        if not self.rows:
            return None
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
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        W, H = self.img_size
        if W == 0 or H == 0:
            return cw, ch
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
        self.canvas.delete("img", "msg", "sel")
        self.sel_id = None
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        if self._bg_key != (cw, ch):
            self.canvas.delete("bg", "stars")
            self.render_space_base(cw, ch)
            self.canvas.create_image(0, 0, image=self.space_photo, anchor="nw", tags="bg")
            self._bg_key = (cw, ch)
        
        im = self.get_image()
        if im is not None:
            self.img_size = im.size
        else:
            self.img_size = (0, 0)

        # Сначала рассчитываем положение картинки, чтобы звёзды учитывали её актуальные границы
        self.layout()
        self.draw_starfield(cw, ch)

        if im is None:
            text = "All entries deleted 🌌" if not self.rows else f"No file:\n{self.current_path()}"
            self.canvas.create_text(cw // 2, ch // 2, text=text,
                                    font=("Segoe UI", 14), fill=ERR_COLOR, tags="msg")
            return

        W, H = im.size
        s, ox, oy = self.eff_scale, self.ox, self.oy
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
                                     image=self.photo, anchor="nw", tags="img")
        self.draw_selection()

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
                outline=SEL_COLOR, width=2, dash=(5, 3), tags="sel")

    def clear_selection(self):
        self.sel = None
        self.sel_start = None
        self.draw_selection()

    def apply_crop(self):
        if not self.rows:
            return
        if not self.sel:
            self.set_status("First, select an area on the image using your mouse.", ERR_COLOR)
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
            self.set_status("Area too small", ERR_COLOR)
            return
        os.makedirs(ORIG_DIR, exist_ok=True)
        backup = os.path.join(ORIG_DIR, self.rows[self.idx][0])
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        im = Image.open(path)
        im.crop((x0, y0, x1, y1)).save(path)
        self.set_status(f"Cropped: {x1 - x0}x{y1 - y0} px (original in _originals)", OK_COLOR)
        self.sel = None
        self.reset_view()
        self.name.config(text=self.rows[self.idx][0] + "   [cropped]")
        self.show_image()

    def restore_original(self):
        if not self.rows:
            return
        fn = self.rows[self.idx][0]
        backup = os.path.join(ORIG_DIR, fn)
        if not os.path.exists(backup):
            self.set_status("There is no saved original for this image", ERR_COLOR)
            return
        shutil.copy2(backup, self.current_path())
        os.remove(backup)
        self.name.config(text=fn)
        self.sel = None
        self.reset_view()
        self.show_image()
        self.set_status("Original restored", OK_COLOR)

    def delete_current(self):
        if not self.rows:
            self.set_status("Nothing to delete", ERR_COLOR)
            return
        fn = self.rows[self.idx][0]
        if not messagebox.askyesno("Delete", f"Delete the entry and the file?\n\n{fn}"):
            return
        path = self.current_path()
        try:
            if os.path.exists(path) and os.path.isfile(path):
                os.remove(path)
        except Exception as ex:
            messagebox.showerror("Delete", f"Could not delete the file:\n{ex}")
            return
        backup = os.path.join(ORIG_DIR, fn)
        if os.path.exists(backup):
            try:
                os.remove(backup)
            except Exception:
                pass
        del self.rows[self.idx]
        del self.original[self.idx]
        self._cache = None
        self.sel = None
        self.dirty = True
        self.save()
        if self.rows:
            self.idx = min(self.idx, len(self.rows) - 1)
        else:
            self.idx = 0
        self.load_current()
        self.set_status(f"Deleted: {fn}" + ("" if self.rows else ". The list is empty."),
                        OK_COLOR)

    def git_push(self, repo_root):
        push = subprocess.run([GIT_EXE, "push"], cwd=str(repo_root),
                              capture_output=True, text=True)
        if push.returncode == 0:
            return None
        branch = subprocess.run([GIT_EXE, "rev-parse", "--abbrev-ref", "HEAD"],
                                cwd=str(repo_root), capture_output=True,
                                text=True).stdout.strip() or "main"
        push = subprocess.run([GIT_EXE, "push", "-u", "origin", branch],
                              cwd=str(repo_root), capture_output=True, text=True)
        if push.returncode == 0:
            return None
        return (push.stderr or push.stdout or "git push failed").strip()

    def commit_to_git(self):
        global GIT_EXE

        if not GIT_EXE or not os.path.isfile(GIT_EXE):
            GIT_EXE = find_git()

        if not GIT_EXE:
            messagebox.showerror(
                "Git commit",
                "Не удалось найти git.exe.\n\n"
                r"Ожидаемый путь: C:\Program Files\Git\cmd\git.exe"
            )
            return

        self.commit_text()

        if self.dirty:
            try:
                self.save()
            except Exception as ex:
                messagebox.showerror(
                    "Git commit",
                    f"Не удалось сохранить labels.csv:\n{ex}"
                )
                return

        try:
            result = subprocess.run(
                [GIT_EXE, "rev-parse", "--show-toplevel"],
                cwd=str(BASE),
                capture_output=True,
                text=True,
                check=True
            )

            repo_root = Path(result.stdout.strip()).resolve()

            csv_relative = CSV_PATH.resolve().relative_to(repo_root).as_posix()
            images_relative = FOLDER.resolve().relative_to(repo_root).as_posix()

            pathspecs = [
                csv_relative,
                images_relative + "/",
                f":(exclude,glob){images_relative}/_originals/**",
                f":(exclude){images_relative}/.verify_progress",
                f":(exclude){images_relative}/.verify_settings.json",
            ]

            subprocess.run(
                [GIT_EXE, "add", "-A", "--", *pathspecs],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=True
            )

            changed = subprocess.run(
                [GIT_EXE, "diff", "--cached", "--quiet", "--", *pathspecs],
                cwd=str(repo_root),
                capture_output=True,
                text=True
            )

            if changed.returncode == 0:
                self.set_status("Git: нет новых изменений для коммита")
                return

            if changed.returncode != 1:
                raise RuntimeError(
                    changed.stderr or "Не удалось проверить изменения Git"
                )

            history = subprocess.run(
                [GIT_EXE, "log", "--all", "--format=%s"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=True
            )

            version = REAL_DATA_VERSION_START

            for subject in (history.stdout or "").splitlines():
                match = re.match(
                    r"^real data 0\.(\d+)(?:\s|$)",
                    subject.strip(),
                    re.IGNORECASE
                )

                if match:
                    version = max(version, int(match.group(1)))

            version += 1

            current_file = self.rows[self.idx][0] if self.rows else "end"

            commit_message = (
                f"real data 0.{version} checked till {current_file}"
            )

            commit = subprocess.run(
                [
                    GIT_EXE,
                    "commit",
                    "-m",
                    commit_message,
                    "--only",
                    "--",
                    *pathspecs
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True
            )

            if commit.returncode != 0:
                details = (commit.stderr or commit.stdout).strip()
                raise RuntimeError(details or "Git commit failed")

            self.set_status(f"Committed: {commit_message}")

            push_error = self.git_push(repo_root)

            if push_error is None:
                self.set_status(f"Committed & pushed: {commit_message}", OK_COLOR)
                messagebox.showinfo(
                    "Git commit",
                    f"Коммит создан и успешно отправлен (push) в репозиторий:\n\n{commit_message}"
                )
            else:
                self.set_status(f"Committed (push failed): {commit_message}", ERR_COLOR)
                messagebox.showwarning(
                    "Git commit",
                    f"Коммит создан, но автоматический push не удался:\n\n{push_error}\n\n"
                    f"Проверьте настройки remote (git remote -v, авторизацию)."
                )

        except subprocess.CalledProcessError as ex:
            details = (ex.stderr or ex.stdout or str(ex)).strip()
            messagebox.showerror("Git commit", details)

        except FileNotFoundError as ex:
            messagebox.showerror(
                "Git commit",
                f"Не удалось запустить Git.\n\n"
                f"Git: {GIT_EXE}\n"
                f"Путь: {ex.filename}\n"
                f"Ошибка: {ex}"
            )

        except Exception as ex:
            messagebox.showerror("Git commit", str(ex))

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
        if not self.rows:
            return "break"
        self.commit_text()
        new = min(max(self.idx + step, 0), len(self.rows) - 1)
        if new != self.idx:
            self.idx = new
            self.load_current()
        return "break"

    def on_enter(self, event):
        if not self.rows:
            return "break"
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
        if not self.rows:
            return "break"
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
                self.set_status("String not found", ERR_COLOR)
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
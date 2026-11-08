import csv
import os
import shutil
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk
from pathlib import Path

BASE = Path(__file__).resolve().parent                 
FOLDER = BASE / "data" / "real_dataset" / "images"  
FOLDER_CSV = BASE / "data" / "real_dataset"
CSV_PATH = FOLDER_CSV / "labels.csv"                        

PROGRESS_PATH = os.path.join(FOLDER, ".verify_progress")
ORIG_DIR = os.path.join(FOLDER, "_originals")

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

        # img
        self.img_scale = 1.0
        self.img_box = (0, 0, 0, 0)   # x0, y0, x1, y1 canvas
        self.img_size = (0, 0)        # file pixels
        self.sel_start = None
        self.sel = None               # (x0,y0,x1,y1) canvas
        self.sel_id = None

        # ---------- left ----------
        left = tk.Frame(root, bg=PANEL, padx=12, pady=12)
        left.pack(side="left", fill="y")
        self.info = tk.Label(left, text="", font=("Segoe UI", 12), bg=PANEL, fg=MUTED,
                             anchor="w")
        self.info.pack(fill="x")
        self.name = tk.Label(left, text="", font=("Consolas", 15, "bold"), bg=PANEL, fg=FG,
                             anchor="w")
        self.name.pack(fill="x", pady=(0, 8))
        self.text = tk.Text(left, width=40, height=8, wrap="word",
                            font=("Segoe UI", 20), undo=True, bg=INPUT_BG, fg=FG,
                            insertbackground=FG, selectbackground=ACCENT,
                            selectforeground="white", relief="flat", bd=0,
                            padx=10, pady=10, highlightthickness=1,
                            highlightbackground="#3c3c3c", highlightcolor=ACCENT)
        self.text.pack(fill="both", expand=True)

        btns = tk.Frame(left, bg=PANEL)
        btns.pack(fill="x", pady=(10, 0))
        make_button(btns, "Crop (Ctrl+R)", self.apply_crop).pack(side="left", padx=(0, 6))
        make_button(btns, "Original (Ctrl+B)", self.restore_original).pack(side="left", padx=(0, 6))
        make_button(btns, "Deselect (Esc)", self.clear_selection).pack(side="left")

        self.status = tk.Label(left, text="", font=("Segoe UI", 10), bg=PANEL, fg=MUTED,
                               anchor="w", justify="left", wraplength=480)
        self.status.pack(fill="x", pady=(8, 0))
        tk.Label(left, text="Enter - continue    Alt+←/→ - back/forward\n"
                            "PgUp/PgDn - ±10    Ctrl+F - search    Ctrl+S - save\n",
                 font=("Segoe UI", 9), bg=PANEL, fg=MUTED, anchor="w", justify="left"
                 ).pack(fill="x", pady=(6, 0))

        # ---------- right panel ----------
        right = tk.Frame(root, bg=CANVAS_BG)
        right.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(right, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.show_image())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # ---------- keys ----------
        self.text.bind("<Return>", self.on_enter)
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
        self.text.bind("<Control-z>", self.undo_line)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.load_current()

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
            self.status.config(text="No changes")
            return
        bak = CSV_PATH + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(CSV_PATH, bak)
        tmp = CSV_PATH + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            w.writerows(self.rows)
        os.replace(tmp, CSV_PATH)
        self.dirty = False
        changed = sum(1 for a, r in zip(self.original, self.rows) if a != r[1])
        self.status.config(text=f"Saved. Changed lines: {changed}")

    # ---------- show ----------
    def current_path(self):
        return os.path.join(FOLDER, self.rows[self.idx][0])

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
        self.clear_selection()
        self.show_image()
        self.save_progress()

    def show_image(self):
        self.canvas.delete("all")
        self.sel_id = None
        path = self.current_path()
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        if not os.path.exists(path):
            self.canvas.create_text(cw // 2, ch // 2, text=f"No file:\n{path}",
                                    font=("Segoe UI", 14), fill="#ff6b6b")
            self.img_box = (0, 0, 0, 0)
            return
        im = Image.open(path).convert("RGB")
        self.img_size = im.size
        scale = min((cw - 20) / im.width, (ch - 20) / im.height, 3.0)
        self.img_scale = scale
        w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        im = im.resize((w, h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        x0, y0 = (cw - w) // 2, (ch - h) // 2
        self.canvas.create_image(x0, y0, image=self.photo, anchor="nw")
        self.img_box = (x0, y0, x0 + w, y0 + h)
        self.draw_selection()


     # ---------- show ----------

    def clamp(self, x, y):
        x0, y0, x1, y1 = self.img_box
        return min(max(x, x0), x1), min(max(y, y0), y1)

    def on_press(self, e):
        if self.img_box[2] <= self.img_box[0]:
            return
        self.sel_start = self.clamp(e.x, e.y)
        self.sel = None
        self.draw_selection()

    def on_drag(self, e):
        if self.sel_start is None:
            return
        x, y = self.clamp(e.x, e.y)
        sx, sy = self.sel_start
        self.sel = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
        self.draw_selection()

    def on_release(self, e):
        self.sel_start = None
        if self.sel and (self.sel[2] - self.sel[0] < 4 or self.sel[3] - self.sel[1] < 4):
            self.sel = None
        self.draw_selection()
        if self.sel:
            self.status.config(text="Highlighted. Ctrl+R - cut, Esc - deselect")

    def draw_selection(self):
        if self.sel_id is not None:
            self.canvas.delete(self.sel_id)
            self.sel_id = None
        if self.sel:
            self.sel_id = self.canvas.create_rectangle(*self.sel, outline=SEL_COLOR,
                                                       width=2, dash=(5, 3))

    def clear_selection(self):
        self.sel = None
        self.sel_start = None
        self.draw_selection()

    def apply_crop(self):
        if not self.sel:
            self.status.config(text="First, select an area on the image using your mouse.")
            return
        path = self.current_path()
        if not os.path.exists(path):
            return
        bx, by, _, _ = self.img_box
        s = self.img_scale
        x0 = int(round((self.sel[0] - bx) / s))
        y0 = int(round((self.sel[1] - by) / s))
        x1 = int(round((self.sel[2] - bx) / s))
        y1 = int(round((self.sel[3] - by) / s))
        W, H = self.img_size
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            self.status.config(text="Area too small")
            return
        os.makedirs(ORIG_DIR, exist_ok=True)
        backup = os.path.join(ORIG_DIR, self.rows[self.idx][0])
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        im = Image.open(path)
        im.crop((x0, y0, x1, y1)).save(path)
        self.status.config(text=f"Cropped: {x1 - x0}x{y1 - y0} px (original in _originals)")
        self.clear_selection()
        self.name.config(text=self.rows[self.idx][0] + "   [cropped]")
        self.show_image()

    def restore_original(self):
        fn = self.rows[self.idx][0]
        backup = os.path.join(ORIG_DIR, fn)
        if not os.path.exists(backup):
            self.status.config(text="There is no saved original for this image")
            return
        shutil.copy2(backup, self.current_path())
        os.remove(backup)
        self.name.config(text=fn)
        self.clear_selection()
        self.show_image()
        self.status.config(text="Original restored")

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
            self.status.config(text="This is the last line")
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
                self.status.config(text="String not found")
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
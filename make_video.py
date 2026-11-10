#!/usr/bin/env python3
"""『変わらない男』— 6枚の写真から、ある男の一生を振り返る映像を生成する。

    python3 make_video.py            # 本編 (life_retrospective.mp4) を書き出す
    python3 make_video.py --stills   # レイアウト確認用の静止画を build/stills/ に書き出す

映像は Pillow + numpy で1フレームずつ描画して ffmpeg に流し込み、
音楽・効果音は numpy / scipy で合成している（外部素材は使っていない）。
"""
import math
import os
import subprocess
import sys
import wave
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import butter, fftconvolve, sosfilt

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
BUILD = os.path.join(ROOT, "build")
OUTPUT = os.path.join(ROOT, "life_retrospective.mp4")

W, H, FPS = 1920, 1080, 30
XF = 1.2          # シーン間クロスフェード秒数
SR = 44100        # 音声サンプルレート
GRAIN_AMT = float(os.environ.get("GRAIN_AMT", "0.006"))   # フィルムグレインの強さ（上げるとファイルが大きくなる）
CRF = os.environ.get("CRF", "24")

CREAM = (237, 230, 214)
GOLD = (201, 169, 110)
WHITE = (246, 243, 236)
VERMILION = (232, 112, 82)


# ---------------------------------------------------------------- fonts / utils
def _find_font(cands):
    for path, idx in cands:
        if os.path.exists(path):
            return path, idx
    raise SystemExit("日本語フォントが見つかりません (fonts-noto-cjk か fonts-ipafont-mincho を入れてください)")


SERIF = _find_font([("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
                    ("/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf", 0)])
SERIF_BOLD = _find_font([("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", 0),
                         ("/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf", 0)])


@lru_cache(maxsize=None)
def font(size, bold=False):
    path, idx = SERIF_BOLD if bold else SERIF
    return ImageFont.truetype(path, size, index=idx)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def smooth(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, t0, d):
    return smooth((t - t0) / d)


def lerp(a, b, p):
    return a + (b - a) * p


class Text:
    """影付きで描画済みのテキストレイヤー (float RGBA)。"""

    def __init__(self, s, size, color, bold=False, tracking=0, shadow=True):
        f = font(size, bold)
        widths = [f.getlength(c) for c in s] if tracking else None
        tw = (sum(widths) + tracking * (len(s) - 1)) if tracking else f.getlength(s)
        asc, desc = f.getmetrics()
        pad = 28
        w, h = int(math.ceil(tw)) + pad * 2, asc + desc + pad * 2

        def draw(img, fill):
            d = ImageDraw.Draw(img)
            if tracking:
                x = pad
                for c, cw in zip(s, widths):
                    d.text((x, pad), c, font=f, fill=fill)
                    x += cw + tracking
            else:
                d.text((pad, pad), s, font=f, fill=fill)

        txt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw(txt, color + (255,))
        if shadow:
            sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw(sh, (0, 0, 0, 210))
            sh = sh.filter(ImageFilter.GaussianBlur(7))
            txt = Image.alpha_composite(sh, txt)
        self.arr = np.asarray(txt, dtype=np.float32) / 255.0
        self.pad, self.w, self.size = pad, tw, size


def blit(frame, layer, x, y, opacity=1.0):
    if opacity <= 0.002:
        return
    h, w = layer.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    sub = layer[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sub[..., 3:4] * opacity
    reg = frame[y0:y1, x0:x1]
    reg *= 1.0 - a
    reg += sub[..., :3] * a


def draw_text(frame, t, x, y, opacity):
    blit(frame, t.arr, x - t.pad, y - t.pad, opacity)


def load(name, max_h=None):
    img = Image.open(os.path.join(ASSETS, name)).convert("RGB")
    if max_h and img.height > max_h:
        img = img.resize((round(img.width * max_h / img.height), max_h), Image.LANCZOS)
    return img


def cover(img, w, h):
    s = max(w / img.width, h / img.height)
    r = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


# ---------------------------------------------------------------- story
PHOTOS = {  # ファイル名, 顔の位置 (0..1)
    "student": ("01_student_1936.png", (0.50, 0.37)),
    "conscription": ("02_conscription_1938.jpg", (0.55, 0.36)),
    "battle": ("03_battlefield_1944.png", (0.46, 0.42)),
    "ueno": ("04_ueno_1946.png", (0.47, 0.49)),
    "alley": ("05_alley_1958.png", (0.43, 0.33)),
    "present": ("06_present_2026.jpg", (0.47, 0.21)),
}


class Scene:
    dur = 10.0
    grain = 1.0
    flicker = 0.0
    start = 0.0
    _ready = False

    def ensure(self):
        if not self._ready:
            self.prepare()
            self._ready = True

    def prepare(self):
        pass


class TitleCard(Scene):
    grain = 0.8

    def __init__(self, dur):
        self.dur = dur

    def prepare(self):
        self.intro = Text("これは、ある男の一生の記録である。", 40, CREAM)
        self.title = Text("変わらない男", 128, WHITE, bold=True, tracking=18)
        self.sub = Text("大正七年 ─ 令和八年", 32, GOLD, tracking=10)

    def render(self, u, t, fi):
        f = np.zeros((H, W, 3), np.float32)
        o = ramp(u, 0.8, 0.8) * (1 - ramp(u, 3.3, 0.7))
        draw_text(f, self.intro, (W - self.intro.w) / 2, 500, o)
        o = ramp(u, 4.1, 1.2)
        draw_text(f, self.title, (W - self.title.w) / 2, 400 + 10 * (1 - o), o)
        draw_text(f, self.sub, (W - self.sub.w) / 2, 600, ramp(u, 4.9, 1.0))
        return f


class Prologue(Scene):
    """現在の独房 → 「話は、九十年前にさかのぼる。」(シネスコの帯つき)"""
    grain = 0.7
    BAR = 130

    def __init__(self, dur, subs):
        self.dur, self.subs_src = dur, subs

    def prepare(self):
        self.src = load(PHOTOS["present"][0])
        self.label = Text("令和八年　東京", 28, GOLD, tracking=8)
        self.subs = [(Text(s, 40, CREAM), a, b) for s, a, b in self.subs_src]

    def render(self, u, t, fi):
        f = np.zeros((H, W, 3), np.float32)
        ah = H - 2 * self.BAR
        sw, sh = self.src.size
        p = 1 - (1 - clamp(u / self.dur, 0, 1)) ** 1.6       # ゆっくり引いていく
        bw = lerp(sw * 0.62, sw * 0.95, p)
        bh = bw * ah / W
        cx = lerp(sw * 0.47, sw * 0.50, p)
        cy = lerp(sh * 0.21, sh * 0.27, p)
        cx, cy = clamp(cx, bw / 2, sw - bw / 2), clamp(cy, bh / 2, sh - bh / 2)
        crop = self.src.resize((W, ah), Image.BICUBIC,
                               box=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        img = np.asarray(crop, np.float32) / 255.0
        img *= np.array([0.93, 0.98, 1.05], np.float32)     # 冷たい色味
        f[self.BAR:self.BAR + ah] = img
        draw_text(f, self.label, 70, 46, ramp(u, 0.8, 1.0))
        for txt, a, b in self.subs:
            o = ramp(u, a, 0.5) * (1 - ramp(u, b, 0.5))
            draw_text(f, txt, (W - txt.w) / 2, H - self.BAR + 30, o)
        return f


class Chapter(Scene):
    PH = 880
    PCX = 600         # 写真の中心 x
    TX = 1120         # テキスト左端

    def __init__(self, key, label, title, meta, lines, dur, zoom=(1.0, 1.16),
                 grain=1.0, flicker=0.012, line_start=3.0, line_gap=2.3, booms=None):
        self.key, self.label_s, self.title_s, self.meta_s = key, label, title, meta
        self.lines_src, self.dur, self.zoom = lines, dur, zoom
        self.grain, self.flicker = grain, flicker
        self.line_start, self.line_gap = line_start, line_gap
        self.booms = booms or []

    def prepare(self):
        fname, self.face = PHOTOS[self.key]
        full = load(fname)
        self.src = load(fname, max_h=int(self.PH * self.zoom[1] * 1.15))
        sw, sh = self.src.size
        self.ph = self.PH
        self.pw = round(self.PH * sw / sh)
        self.px, self.py = self.PCX - self.pw // 2, (H - self.ph) // 2

        # 背景: 写真をぼかして暗く敷き、写真プリントの影と縁を焼き込む
        bg = cover(full, W // 4, H // 4).filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BICUBIC)
        bg = np.asarray(bg, np.float32) / 255.0
        lum = bg.mean(axis=2, keepdims=True)
        bg = (bg * 0.6 + lum * 0.4) * 0.26
        ramp_x = np.clip((np.arange(W) - 1000) / 600, 0, 1)[None, :, None]
        bg *= 1 - 0.35 * ramp_x                                  # 文字側を少し暗く
        b = 14
        sh_img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(sh_img).rectangle((self.px - b + 14, self.py - b + 22,
                                          self.px + self.pw + b + 14, self.py + self.ph + b + 22), fill=255)
        sh_a = np.asarray(sh_img.filter(ImageFilter.GaussianBlur(26)), np.float32)[..., None] / 255.0
        bg *= 1 - 0.75 * sh_a
        bg[self.py - b:self.py + self.ph + b, self.px - b:self.px + self.pw + b] = np.array(CREAM, np.float32) / 255 * 0.93
        self.bg = bg

        # テキスト（ブロック全体を縦方向中央に）
        self.t_label = Text(self.label_s, 30, GOLD, tracking=10)
        self.t_title = Text(self.title_s, 92, WHITE, bold=True, tracking=6)
        self.t_meta = Text(self.meta_s, 28, GOLD, tracking=2)
        self.t_lines = []
        for i, item in enumerate(self.lines_src):
            s, style, at = item if len(item) == 3 else (item[0], item[1], None)
            if at is None:
                at = self.line_start + i * self.line_gap
            if style == "big":
                self.t_lines.append((Text(s, 68, VERMILION, bold=True, tracking=8), at, 126))
            else:
                self.t_lines.append((Text(s, 38, CREAM), at, 70))
        block = 52 + 124 + 58 + 40 + sum(step for _, _, step in self.t_lines)
        self.y0 = (H - block) // 2 - 6

    def crop_box(self, u, t):
        sw, sh = self.src.size
        p = 0.5 - 0.5 * math.cos(math.pi * clamp(u / self.dur, 0, 1))
        z = lerp(*self.zoom, p)
        cw, ch = sw / z, sh / z
        cx = lerp(sw / 2, self.face[0] * sw, p)
        cy = lerp(sh / 2, self.face[1] * sh, p)
        for tb in self.booms:                                   # 爆発の揺れ
            if t >= tb:
                a = 16 * math.exp(-(t - tb) * 4.5) * sw / 1000
                cx += a * math.sin(t * 57 + tb * 3)
                cy += a * math.cos(t * 63 + tb * 7)
        cx, cy = clamp(cx, cw / 2, sw - cw / 2), clamp(cy, ch / 2, sh - ch / 2)
        return (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)

    def render(self, u, t, fi):
        f = self.bg.copy()
        crop = self.src.resize((self.pw, self.ph), Image.BICUBIC, box=self.crop_box(u, t))
        f[self.py:self.py + self.ph, self.px:self.px + self.pw] = np.asarray(crop, np.float32) / 255.0

        out_t = self.dur - XF - 0.4

        def show(txt, y, t_in, dur_in=0.8):
            o = ramp(u, t_in, dur_in) * (1 - ramp(u, out_t, 0.6))
            rise = (1 - ramp(u, t_in, 1.0)) * 14
            draw_text(f, txt, self.TX, y + rise, o)

        y = self.y0
        show(self.t_label, y, 1.0)
        y += 52
        show(self.t_title, y, 1.3, 1.0)
        y += 124
        show(self.t_meta, y, 1.9)
        y += 58
        # 罫線
        ln = int(560 * smooth((u - 2.3) / 0.9))
        o = 1 - ramp(u, out_t, 0.6)
        if ln > 0 and o > 0:
            seg = f[y:y + 2, self.TX:self.TX + ln]
            seg *= 1 - 0.7 * o
            seg += np.array(GOLD, np.float32) / 255 * 0.7 * o
        y += 40
        for txt, at, step in self.t_lines:
            show(txt, y + (8 if step > 70 else 0), at, 0.9 if step > 70 else 0.8)
            y += step

        for tb in self.booms:                                   # 爆発の閃光
            if t >= tb - 0.02:
                k = math.exp(-max(t - tb, 0) * 6.5) * 0.30
                f += np.array([1.0, 0.62, 0.30], np.float32) * k
        return f


class Ending(Scene):
    grain = 0.9
    TH = 300

    def __init__(self, dur):
        self.dur = dur

    def prepare(self):
        order = ["student", "conscription", "battle", "ueno", "alley", "present"]
        years = ["1936", "1938", "1944", "1946", "1958", "2026"]
        angles = [-2.6, 1.9, -1.3, 2.3, -1.8, 1.6]
        b = 12
        cards, widths = [], []
        for key, ang in zip(order, angles):
            img = load(PHOTOS[key][0])
            tw = round(self.TH * img.width / img.height)
            card = Image.new("RGBA", (tw + 2 * b, self.TH + 2 * b), CREAM + (255,))
            card.paste(img.resize((tw, self.TH), Image.LANCZOS), (b, b))
            m = 40
            canvas = Image.new("RGBA", (card.width + 2 * m, card.height + 2 * m), (0, 0, 0, 0))
            shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            shadow.paste((0, 0, 0, 170), (m + 8, m + 12, m + 8 + card.width, m + 12 + card.height))
            canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(12)))
            canvas.alpha_composite(card, (m, m))
            canvas = canvas.rotate(ang, Image.BICUBIC, expand=True)
            cards.append(np.asarray(canvas, np.float32) / 255.0)
            widths.append(card.width)
        gap = 44
        x = (W - (sum(widths) + gap * (len(widths) - 1))) / 2
        self.cy = 520
        self.cards = []
        for arr, w, yr in zip(cards, widths, years):
            cx = x + w / 2
            self.cards.append((arr, cx, Text(yr, 26, GOLD, tracking=6)))
            x += w + gap
        self.title = Text("変わらない男", 96, WHITE, bold=True, tracking=14)
        self.tag = Text("大正七年生まれ。今日も、変わらず元気です。", 36, CREAM)
        self.fin = Text("完", 120, WHITE, bold=True)
        self.note = Text("この映像はフィクションです。", 26, GOLD, tracking=4)
        g = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        self.bg = (np.array([0.075, 0.062, 0.050], np.float32) * (1.15 - 0.4 * g)) * np.ones((1, W, 1), np.float32)

    def render(self, u, t, fi):
        f = self.bg.copy()
        out_o = 1 - ramp(u, 9.0, 0.9)
        for i, (arr, cx, yr) in enumerate(self.cards):
            a = 1.0 + i * 0.32
            o = ramp(u, a, 0.7) * out_o
            drop = (1 - ramp(u, a, 0.9)) * -34
            h, w = arr.shape[:2]
            blit(f, arr, cx - w / 2, self.cy - h / 2 + drop, o)
            draw_text(f, yr, cx - yr.w / 2, self.cy + self.TH / 2 + 34, o)
        draw_text(f, self.title, (W - self.title.w) / 2, 150, ramp(u, 3.4, 1.2) * out_o)
        draw_text(f, self.tag, (W - self.tag.w) / 2, 840, ramp(u, 4.8, 1.0) * out_o)
        o = ramp(u, 10.0, 1.0)
        draw_text(f, self.fin, (W - self.fin.w) / 2, 400, o)
        draw_text(f, self.note, (W - self.note.w) / 2, 620, ramp(u, 10.8, 1.0))
        return f


SCENES = [
    ("title", TitleCard(8.0)),
    ("prologue", Prologue(12.0, [
        ("男は、鉄格子の向こうからこちらを見ていた。", 1.4, 4.4),
        ("なぜ、彼はここにいるのか。", 4.8, 7.6),
        ("話は、九十年前にさかのぼる。", 8.0, 10.6),
    ])),
    ("ch1", Chapter("student", "第一章", "青雲", "昭和十一年（1936）・ 東京市 ・ 十八歳", [
        ("本を抱えて、市電の走る街を歩いた。", "n"),
        ("夢は、学者になること。", "n"),
        ("好物は、角の店の天ぷら。", "n"),
    ], 12.5, grain=1.1)),
    ("ch2", Chapter("conscription", "第二章", "検査", "昭和十三年（1938）・ 徴兵検査場 ・ 二十歳", [
        ("徴兵検査の日。", "n"),
        ("精一杯の力こぶを作ってみせた。", "n"),
        ("判定は、第二乙種。", "n"),
        ("それでも、時代は彼を必要とした。", "n"),
    ], 13.5, grain=1.25, flicker=0.02, line_gap=2.2)),
    ("ch3", Chapter("battle", "第三章", "戦火", "昭和十九年（1944）・ 戦地 ・ 二十六歳", [
        ("泥と炎の中を、ただ走った。", "n"),
        ("眼鏡だけは、決して手放さなかった。", "n"),
        ("生きて帰る。それだけを考えていた。", "n"),
    ], 12.5, zoom=(1.06, 1.2), grain=1.3, flicker=0.02)),
    ("ch4", Chapter("ueno", "第四章", "帰還", "昭和二十一年（1946）・ 上野駅 ・ 二十八歳", [
        ("『おかえりなさい』の文字が、", "n"),
        ("にじんで、よく見えなかった。", "n"),
        ("母は、何も言わずに泣いていた。", "n"),
    ], 12.5, grain=1.1)),
    ("ch5", Chapter("alley", "第五章", "灯り", "昭和三十三年（1958）・ 下町 ・ 四十歳", [
        ("焼け跡の街に、灯りが戻った。", "n"),
        ("冷やの一杯と、北斎の扇子。", "n"),
        ("彼は、よく笑う男になった。", "n"),
    ], 12.5, grain=0.9, flicker=0.008)),
    ("finale", Chapter("present", "終章", "そして、現在", "令和八年（2026）・ 東京", [
        ("職務質問で、男は身分証を差し出した。", "n", 3.0),
        ("生年月日 ── 大正七年。", "n", 5.6),
        ("百八歳。", "big", 7.6),
        ("確認のため、一晩泊まることになった。", "n", 10.0),
        ("本人いわく『ずっと、この顔です』", "n", 12.6),
    ], 17.0, zoom=(1.0, 1.22), grain=0.6, flicker=0.0)),
    ("ending", Ending(14.0)),
]

S = {}
_t = 0.0
for _name, _sc in SCENES:
    _sc.start = _t
    S[_name] = _t
    _t += _sc.dur - XF
TOTAL = SCENES[-1][1].start + SCENES[-1][1].dur
FLASH = S["ch1"] + XF * 0.5                                  # 過去へ戻る白い閃光
BOOMS = [S["ch3"] + d for d in (1.4, 3.7, 6.2, 7.9, 10.1)]
SCENES[4][1].booms = BOOMS


# ---------------------------------------------------------------- frame compositor
_G = {}


def _init_worker():
    rng = np.random.default_rng(1234)
    grains = []
    for _ in range(6):
        g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
        g = np.asarray(Image.fromarray(g, "F").resize((W, H), Image.BILINEAR), np.float32)
        grains.append(g[..., None])
    _G["grain"] = grains
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    _G["vig"] = (1 - 0.42 * r ** 2.2)[..., None].astype(np.float32)


def render_frame(fi):
    t = fi / FPS
    active = [sc for _, sc in SCENES if sc.start <= t < sc.start + sc.dur]
    if not active:
        active = [SCENES[-1][1]]
    frames, weights = [], []
    for sc in active:
        sc.ensure()
        frames.append(sc.render(t - sc.start, t, fi))
    if len(active) == 1:
        f, grain, flick = frames[0], active[0].grain, active[0].flicker
    else:
        a = smooth((t - active[1].start) / XF)
        f = frames[0] * (1 - a) + frames[1] * a
        grain = lerp(active[0].grain, active[1].grain, a)
        flick = lerp(active[0].flicker, active[1].flicker, a)

    if flick:
        rnd = np.random.default_rng(fi).random()
        f *= 1 + flick * (math.sin(fi * 1.7) * 0.5 + (rnd - 0.5))
    d = t - FLASH                                              # 白い閃光
    k = math.exp(-(d / 0.12) ** 2) if d < 0 else math.exp(-d / 0.35)
    if k > 0.002:
        f = f * (1 - 0.92 * k) + np.array([1.0, 0.97, 0.9], np.float32) * 0.92 * k
    f *= _G["vig"]
    f += _G["grain"][(fi // 2) % 6] * (GRAIN_AMT * grain)
    f *= ramp(t, 0.0, 0.6) * (1 - ramp(t, TOTAL - 1.2, 1.1))
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()


# ---------------------------------------------------------------- audio
BPM = 72
BEAT = 60 / BPM
EIGHTH = BEAT / 2
BAR = BEAT * 4
_ar = np.random.default_rng(7)

_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nfreq(name):
    acc = {"#": 1, "b": -1}.get(name[1], 0)
    octave = int(name[2:] if acc else name[1:])
    midi = 12 * (octave + 1) + _NOTE[name[0]] + acc
    return 440 * 2 ** ((midi - 69) / 12)


def _sos(kind, f, order=2):
    return butter(order, f, kind, fs=SR, output="sos")


def piano(name, vel, length):
    f = nfreq(name)
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, 13):
        fk = k * f * math.sqrt(1 + 0.00035 * k * k)
        if fk > 11000:
            break
        amp = k ** -1.45 * (0.45 + vel) ** ((k - 1) * 0.4)
        dec = 0.42 * (f / 262) ** 0.5 + 0.28 * (k - 1) ** 1.1
        env = np.exp(-t * dec)
        out += amp * env * (np.sin(2 * np.pi * fk * t + _ar.random() * 6.28)
                            + 0.6 * np.sin(2 * np.pi * fk * 1.0011 * t + _ar.random() * 6.28))
    out *= np.minimum(1, t / 0.004)
    rel = int(0.18 * SR)
    out[-rel:] *= np.linspace(1, 0, rel)
    out += _ar.standard_normal(n) * np.exp(-t * 400) * 0.05      # ハンマーの当たり
    return out * vel * 0.22


def pad(freqs, length, attack=2.0, release=2.0):
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        for det in (0.9985, 1.0015):
            ph = _ar.random() * 6.28
            out += (np.sin(2 * np.pi * f * det * t + ph) + 0.3 * np.sin(4 * np.pi * f * det * t + ph)
                    + 0.1 * np.sin(6 * np.pi * f * det * t)) * (1 + 0.15 * np.sin(2 * np.pi * 0.13 * t + ph))
    env = np.minimum(1, t / attack) * np.minimum(1, (length - t) / release)
    return out * env / (len(freqs) * 2)


def boom():
    n = int(3.2 * SR)
    t = np.arange(n) / SR
    low = sosfilt(_sos("low", 140, 4), _ar.standard_normal(n))
    low = low / np.abs(low).max() * np.exp(-t * 1.6)
    sweep = 2 * np.pi * np.cumsum(28 + 32 * np.exp(-t * 3)) / SR
    sub = np.sin(sweep) * np.exp(-t * 2.2)
    crack = sosfilt(_sos("high", 1500), _ar.standard_normal(n))
    crack = crack / np.abs(crack).max() * np.exp(-t * 11)
    debris = sosfilt(_sos("band", [2500, 7000]), _ar.standard_normal(n) * (_ar.random(n) > 0.997))
    debris = debris / (np.abs(debris).max() + 1e-9) * np.exp(-t * 2.5) * np.minimum(1, t / 0.3)
    s = low + 0.9 * sub + 0.3 * crack + 0.15 * debris
    s *= np.minimum(1, t / 0.003)
    return s / np.abs(s).max()


def noise_bed(length, lo, hi, mod_rate=None):
    n = int(length * SR)
    s = sosfilt(_sos("band", [lo, hi]), _ar.standard_normal(n))
    if mod_rate:
        m = sosfilt(_sos("low", mod_rate), _ar.standard_normal(n))
        m = 0.55 + 0.45 * m / np.abs(m).max()
        s *= m
    return s / np.abs(s).max()


def whistle(length=2.0):
    n = int(length * SR)
    t = np.arange(n) / SR
    glide = 0.965 + 0.035 * (1 - np.exp(-t * 6))
    s = np.zeros(n)
    for f in (311.1, 370.0, 466.2):
        ph = 2 * np.pi * np.cumsum(f * glide) / SR
        s += np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    s = s / 3 + 0.2 * noise_bed(length, 900, 3000)
    s *= np.minimum(1, t / 0.15) * np.minimum(1, (length - t) / 0.6)
    return sosfilt(_sos("low", 2800), s)


def clank():
    n = int(2.8 * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for i, r in enumerate((1, 2.41, 3.93, 5.47, 7.12, 9.8)):
        s += np.sin(2 * np.pi * 196 * r * t) * np.exp(-t * (2.0 + 1.6 * i)) / (i + 1) ** 0.7
    s += sosfilt(_sos("high", 2000), _ar.standard_normal(n)) * np.exp(-t * 60) * 0.8
    s += np.sin(2 * np.pi * 68 * t) * np.exp(-t * 12) * 0.9
    return s / np.abs(s).max()


def rin():
    """チーン"""
    n = int(4.5 * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for f, d, a in ((1184, 0.8, 1.0), (1187.5, 0.8, 0.8), (3205, 1.6, 0.45), (5640, 2.8, 0.2)):
        s += a * np.sin(2 * np.pi * f * t) * np.exp(-t * d)
    s *= np.minimum(1, t / 0.002)
    return s / np.abs(s).max()


def snare():
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    s = sosfilt(_sos("high", 1800), _ar.standard_normal(n)) * np.exp(-t * 16) * 0.7
    s += np.sin(2 * np.pi * 185 * t) * np.exp(-t * 28) * 0.5
    return sosfilt(_sos("low", 4000), s)


def riser(length):
    n = int(length * SR)
    t = np.arange(n) / SR
    s = sosfilt(_sos("high", 3000), _ar.standard_normal(n)) * 0.6
    s += np.sin(2 * np.pi * np.cumsum(200 + 1400 * (t / length) ** 2) / SR) * 0.4
    return s * (t / length) ** 3


CHORDS = {
    "F": ["F2", "C3", "F3", "A3", "C4", "A3", "F3", "C3"],
    "G": ["G2", "D3", "G3", "B3", "D4", "B3", "G3", "D3"],
    "Em": ["E2", "B2", "E3", "G3", "B3", "G3", "E3", "B2"],
    "Am": ["A2", "E3", "A3", "C4", "E4", "C4", "A3", "E3"],
    "C": ["C3", "G3", "C4", "E4", "G4", "E4", "C4", "G3"],
    "E": ["E2", "B2", "E3", "G#3", "B3", "G#3", "E3", "B2"],
}


def build_audio():
    n = int((TOTAL + 1) * SR)
    wet = np.zeros((2, n))    # 残響を通すバス
    dry = np.zeros((2, n))

    def add(bus, sig, t0, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= n:
            return
        seg = sig[: n - i0] * gain
        th = (pan + 1) * math.pi / 4
        bus[0, i0:i0 + len(seg)] += seg * math.cos(th)
        bus[1, i0:i0 + len(seg)] += seg * math.sin(th)

    def note(name, t0, vel, length, pan=None):
        t0 += _ar.normal(0, 0.006)
        vel *= 1 + _ar.normal(0, 0.07)
        if pan is None:
            pan = clamp((nfreq(name) - 400) / 900, -0.45, 0.45)
        add(wet, piano(name, vel, length), t0, pan=pan)

    def arpeggio(t0, chords, vel, step=EIGHTH, bass=False):
        for bi, ch in enumerate(chords):
            bar0 = t0 + bi * BAR
            pat = CHORDS[ch] if step == EIGHTH else CHORDS[ch][:4]
            for j, nm in enumerate(pat):
                tt = bar0 + j * step
                note(nm, tt, vel * (1.15 if j == 0 else 0.85), bar0 + BAR - tt + 0.35)
            if bass:
                low = CHORDS[ch][0][:-1] + str(int(CHORDS[ch][0][-1]) - 1)
                note(low, bar0, vel * 1.1, BAR + 0.4, pan=-0.2)

    def melody(t0, seq, vel):
        tt = t0
        for nm, beats in seq:
            if nm:
                note(nm, tt, vel, beats * BEAT + 0.5)
            tt += beats * BEAT

    # --- タイトル
    note("E5", 1.0, 0.22, 4.5)
    for i, nm in enumerate(["A2", "E3", "A3", "C4", "E4", "B4"]):
        note(nm, 4.15 + i * 0.07, 0.3, 4.0)

    # --- プロローグ: 冷たいドローン、扉の音、過去へ巻き戻る
    P = S["prologue"]
    add(wet, pad([55.0, 82.4, 110.0], FLASH - P + 0.4, attack=2.5, release=0.6), P, gain=0.32)
    add(wet, clank(), P + 0.9, gain=0.35, pan=0.3)
    for nm, d in (("E5", 2.6), ("D5", 5.1), ("C5", 7.6), ("B4", 9.6)):
        note(nm, P + d, 0.16, 3.5)
    add(wet, riser(1.6), FLASH - 1.6, gain=0.18)
    for i, nm in enumerate(["C5", "E5", "G5", "C6"]):
        note(nm, FLASH + i * 0.03, 0.22, 3.5)

    # --- 第一章〜第二章: メインテーマ（王道進行）
    T0 = FLASH + 0.1
    arpeggio(T0, ["F", "G", "Em", "Am"], 0.28)
    arpeggio(T0 + 4 * BAR, ["F", "G", "E"], 0.32, bass=True)
    melody(T0, [("A4", 1.5), ("G4", .5), ("A4", 1), ("C5", 1),
                ("D5", 1.5), ("C5", .5), ("B4", 1), ("G4", 1),
                ("B4", 1.5), ("A4", .5), ("G4", 1), ("E4", 1),
                ("A4", 2), ("E4", 1), ("G4", 1),
                ("A4", 1.5), ("C5", .5), ("F5", 1), ("E5", 1),
                ("D5", 1.5), ("E5", .5), ("D5", 1), ("B4", 1),
                ("G#4", 2), ("B4", 1), ("E5", 1)], 0.42)
    t = S["ch2"] + 1.2                                     # 遠くの小太鼓
    pattern = [(0, .9), (1, .5), (2, .7), (2.5, .45), (3, .6)]
    while t < S["ch3"] - 0.4:
        for b, v in pattern:
            if t + b * BEAT < S["ch3"] - 0.4:
                add(wet, snare(), t + b * BEAT, gain=0.07 * v, pan=0.35)
        t += BAR

    # --- 第三章: 戦火
    B = S["ch3"]
    add(wet, pad([73.4, 110.0, 146.8, 155.6], S["ch4"] + 0.8 - B, attack=1.0, release=1.5), B, gain=0.55)
    rumble = sosfilt(_sos("low", 110, 4), _ar.standard_normal(int((S["ch4"] + 1 - B) * SR)))
    rumble = rumble / np.abs(rumble).max()
    tr = np.arange(len(rumble)) / SR
    rumble *= np.minimum(1, tr / 1.0) * np.minimum(1, (tr[-1] - tr) / 1.5)
    add(dry, rumble, B, gain=0.35)
    for i, tb in enumerate(BOOMS):
        add(dry, boom(), tb, gain=0.75 + 0.1 * (i % 2), pan=(-0.4, 0.3, -0.1, 0.45, 0.0)[i])

    # --- 第四章: 上野駅
    U = S["ch4"]
    add(wet, whistle(2.2), U + 0.9, gain=0.10, pan=-0.3)
    crowd = noise_bed(S["ch5"] + 1.2 - U, 280, 1600, mod_rate=5) * 0.7 + noise_bed(S["ch5"] + 1.2 - U, 120, 500, mod_rate=3) * 0.3
    tc = np.arange(len(crowd)) / SR
    crowd *= np.minimum(1, tc / 1.5) * np.minimum(1, (tc[-1] - tc) / 1.5)
    add(dry, crowd, U, gain=0.05)
    T4 = U + 1.3
    arpeggio(T4, ["Am", "F"], 0.26, step=BEAT)
    arpeggio(T4 + 2 * BAR, ["C"], 0.24, step=EIGHTH)
    melody(T4, [("E5", 2), ("C5", 2), ("A4", 2), ("C5", 1), ("D5", 1), ("E5", 2), ("D5", 2)], 0.34)

    # --- 第五章: 下町の灯り
    A = S["ch5"] + 0.6
    arpeggio(A, ["F", "G", "C", "Am"], 0.31)
    melody(A, [("A4", 1.5), ("G4", .5), ("A4", 1), ("C5", 1),
               ("D5", 1.5), ("C5", .5), ("D5", 1), ("E5", 1),
               ("G5", 1.5), ("E5", .5), ("C5", 1), ("D5", 1),
               ("E5", 3), (None, 1)], 0.42)

    # --- 終章: 間 → チーン
    Fi = S["finale"]
    tt = Fi + 2.9
    k = 0
    while tt < Fi + 5.4:
        note(("E5", "C5")[k % 2], tt, 0.14, 0.3)
        tt += EIGHTH
        k += 1
    add(wet, rin(), Fi + 7.6, gain=0.32, pan=0.1)
    tt = Fi + 10.0
    for i in range(12):
        note(["C4", "E4", "G4", "E4"][i % 4], tt, 0.2, 0.45)
        tt += BEAT
    note("C3", Fi + 10.0, 0.22, 4.0)
    note("F2", Fi + 10.0 + 4 * BEAT, 0.22, 4.0)
    note("G2", Fi + 10.0 + 8 * BEAT, 0.22, 4.0)

    # --- エンディング
    E = S["ending"] + 0.6
    arpeggio(E, ["F", "G"], 0.3)
    melody(E, [("A4", 1.5), ("G4", .5), ("A4", 1), ("C5", 1),
               ("D5", 1.5), ("E5", .5), ("D5", 1), ("B4", 1)], 0.42)
    for i, nm in enumerate(["C2", "G2", "C3", "G3", "C4", "E4", "G4", "C5"]):
        note(nm, E + 2 * BAR + i * 0.045, 0.3, 7.0)
    note("C6", S["ending"] + 10.0, 0.16, 4.0)
    note("G5", S["ending"] + 10.05, 0.12, 4.0)

    # --- ミックス
    wet = np.stack([sosfilt(_sos("low", 7000), ch) for ch in wet])
    irs = []
    tir = np.arange(int(2.6 * SR)) / SR
    for _ in range(2):
        ir = sosfilt(_sos("low", 4500), _ar.standard_normal(len(tir))) * np.exp(-tir * 2.6)
        ir[: int(0.015 * SR)] = 0
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    rev = np.stack([fftconvolve(wet[c], irs[c])[:n] for c in range(2)])
    music = wet + 0.32 * rev
    music *= 0.5 / np.abs(music).max()
    mix = music + dry * 0.8
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    tm = np.arange(n) / SR
    mix *= np.minimum(1, tm / 0.5) * np.clip((TOTAL - tm) / 1.5, 0, 1)
    mix *= 0.9 / np.abs(mix).max()
    return mix


def write_wav(path, stereo):
    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- main
def stills():
    out = os.path.join(BUILD, "stills")
    os.makedirs(out, exist_ok=True)
    _init_worker()
    times = {"title": 6.0, "prologue": S["prologue"] + 6.0, "ch1": S["ch1"] + 9.5,
             "ch2": S["ch2"] + 10.5, "ch3": S["ch3"] + 9.6, "ch4": S["ch4"] + 9.5,
             "ch5": S["ch5"] + 9.5, "finale": S["finale"] + 14.0, "ending": S["ending"] + 7.0,
             "fin": S["ending"] + 12.0, "flash": FLASH + 0.25}
    for name, t in times.items():
        buf = render_frame(int(t * FPS))
        Image.frombuffer("RGB", (W, H), buf).save(os.path.join(out, f"{name}.jpg"), quality=88)
        print("still", name, f"{t:.1f}s")


def main():
    os.makedirs(BUILD, exist_ok=True)
    if "--stills" in sys.argv:
        stills()
        return
    print(f"total {TOTAL:.1f}s / {int(TOTAL * FPS)} frames")
    wav = os.path.join(BUILD, "soundtrack.wav")
    write_wav(wav, build_audio())
    print("audio done")
    nframes = int(TOTAL * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-c:v", "libx264", "-preset", "slow", "-crf", CRF, "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
           "-metadata", "title=変わらない男", OUTPUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init_worker) as pool:
        for i, buf in enumerate(pool.imap(render_frame, range(nframes), chunksize=6)):
            proc.stdin.write(buf)
            if i % 300 == 0:
                print(f"frame {i}/{nframes}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""映画『変わらない男』予告編を生成する（make_video.py の部品を流用）。

    python3 make_trailer.py            # trailer.mp4 を書き出す
    python3 make_trailer.py --stills   # 確認用の静止画を build/trailer_stills/ に書き出す

素材は assets/ の6枚の写真だけ。カット割り（寄り・パン・手持ちの揺れ）、
火の粉・埃・グリッチ・光の筋・時計・照合端末の画面などはすべてコードで描き、
音楽と効果音（ブラーム、太鼓、時計の秒針、電子音など）も合成している。
"""
import bisect
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import fftconvolve, sosfilt

from make_video import (BAR, BEAT, CHORDS, CREAM, CRF, FPS, GOLD, GRAIN_AMT, H, PHOTOS, SR, VERMILION,
                        WHITE, W, _ar, _sos, blit, boom, clamp, font, lerp, load, nfreq, noise_bed, pad,
                        piano, ramp, rin, smooth, whistle, write_wav)

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
OUTPUT = os.path.join(ROOT, "trailer.mp4")

PY0, PY1 = 138, 942          # シネスコ (2.39:1) の画面領域
PH = PY1 - PY0
ASPECT = W / PH
CYAN = (110, 220, 235)


# ---------------------------------------------------------------- text
@lru_cache(maxsize=None)
def mono(size):
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(path):
        for i in range(12):
            try:
                f = ImageFont.truetype(path, size, index=i)
            except OSError:
                break
            if f.getname()[0] == "Noto Sans Mono CJK JP":
                return f
    return font(size)


class Label:
    """描画済みテキスト。拡大縮小した版もキャッシュする。"""

    def __init__(self, s=None, fnt=None, color=WHITE, tracking=0, shadow=True, image=None):
        if image is None:
            widths = [fnt.getlength(c) for c in s]
            tw = sum(widths) + tracking * (len(s) - 1)
            asc, desc = fnt.getmetrics()
            p = 30
            size = (int(math.ceil(tw)) + 2 * p, asc + desc + 2 * p)

            def draw(img, fill):
                d = ImageDraw.Draw(img)
                x = p
                for c, cw in zip(s, widths):
                    d.text((x, p), c, font=fnt, fill=fill)
                    x += cw + tracking

            image = Image.new("RGBA", size, (0, 0, 0, 0))
            draw(image, tuple(color) + (255,))
            if shadow:
                sh = Image.new("RGBA", size, (0, 0, 0, 0))
                draw(sh, (0, 0, 0, 220))
                image = Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)), image)
            self.pad, self.w = p, tw
        else:
            self.pad, self.w = 0, image.width
        self.img = image
        self.arr = np.asarray(image, np.float32) / 255.0
        self._sc = {}

    def scaled(self, k):
        k = round(k, 3)
        if k == 1.0:
            return self.arr
        a = self._sc.get(k)
        if a is None:
            im = self.img.convert("RGBa").resize(
                (max(1, round(self.img.width * k)), max(1, round(self.img.height * k))), Image.BICUBIC)
            a = self._sc[k] = np.asarray(im.convert("RGBA"), np.float32) / 255.0
        return a


def put(f, lab, cx, cy, o=1.0, k=1.0):
    a = lab.scaled(k)
    blit(f, a, cx - a.shape[1] / 2, cy - a.shape[0] / 2, o)


def put_left(f, lab, x, y, o=1.0, frac=1.0):
    a = lab.arr
    if frac < 1.0:
        a = a[:, : lab.pad + max(0, int(lab.w * frac))]
    blit(f, a, x - lab.pad, y - lab.pad, o)


# ---------------------------------------------------------------- image fx
GRADES = {
    "cold": dict(mult=(0.86, 0.97, 1.08), sat=0.75, contrast=1.08, lift=(0.0, 0.01, 0.025)),
    "sepia": dict(mult=(1.07, 1.0, 0.86), sat=0.8, contrast=1.06, lift=(0.015, 0.008, 0.0)),
    "mono": dict(mult=(1.04, 1.0, 0.92), sat=0.0, contrast=1.12, lift=(0.01, 0.005, 0.0)),
    "war": dict(mult=(1.1, 0.97, 0.86), sat=1.1, contrast=1.18, lift=(0.0, 0.0, 0.0)),
    "faded": dict(mult=(1.03, 1.0, 0.94), sat=0.78, contrast=1.0, lift=(0.03, 0.03, 0.035)),
    "showa": dict(mult=(1.06, 1.0, 0.9), sat=1.08, contrast=1.06, lift=(0.0, 0.0, 0.0)),
}
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def grade(a, mult, sat, contrast, lift):
    if sat != 1.0:
        lum = (a @ _LUMA)[..., None]
        a = lum + (a - lum) * sat
    if contrast != 1.0:
        a = (a - 0.5) * contrast + 0.5
    return a * np.array(mult, np.float32) + np.array(lift, np.float32)


def chroma(f, px):
    d = int(round(px))
    if d >= 1:
        f[..., 0] = np.roll(f[..., 0], d, axis=1)
        f[..., 2] = np.roll(f[..., 2], -d, axis=1)


def glitch(f, seed, amt):
    if amt <= 0.02:
        return
    r = np.random.default_rng(seed)
    for _ in range(int(8 * amt) + 1):
        y0 = int(r.integers(PY0, PY1 - 12))
        h = int(r.integers(4, 70))
        f[y0:y0 + h] = np.roll(f[y0:y0 + h], int(r.normal(0, 90 * amt)), axis=1)
    chroma(f, 12 * amt)
    f[PY0:PY1:3] *= 1 - 0.25 * amt


_PART = {}


def particles(f, t, kind, seed, strength=1.0):
    key = (kind, seed)
    if key not in _PART:
        r = np.random.default_rng(seed)
        n = 90 if kind == "embers" else 70
        _PART[key] = dict(
            x0=r.random(n) * W, y0=r.random(n), ph=r.random(n) * 6.28, fr=r.uniform(3, 9, n),
            v=r.uniform(70, 230, n) if kind == "embers" else r.uniform(-7, 7, n),
            vx=r.uniform(-25, 45, n) if kind == "embers" else r.uniform(4, 18, n),
            size=r.uniform(1.2, 3.4, n) if kind == "embers" else r.uniform(1.0, 2.4, n),
            br=r.uniform(0.5, 1.0, n) if kind == "embers" else r.uniform(0.12, 0.35, n))
    P = _PART[key]
    if kind == "embers":
        span = PH + 100
        y = PY1 + 50 - ((P["y0"] * span + P["v"] * t) % span)
        x = (P["x0"] + P["vx"] * t + 18 * np.sin(t * 1.3 + P["ph"])) % W
        b = P["br"] * (0.55 + 0.45 * np.sin(t * P["fr"] + P["ph"]))
        color = (1.0, 0.5, 0.15)
    else:
        y = PY0 + ((P["y0"] * PH + P["v"] * t) % PH)
        x = (P["x0"] + P["vx"] * t) % W
        b = P["br"] * (0.7 + 0.3 * np.sin(t * 0.7 * P["fr"] + P["ph"]))
        color = (0.8, 0.9, 1.0)
    canvas = Image.new("L", (W // 2, H // 2), 0)
    d = ImageDraw.Draw(canvas)
    for xi, yi, si, bi in zip(x / 2, y / 2, P["size"] / 2, b):
        d.ellipse((xi - si, yi - si, xi + si, yi + si), fill=int(255 * clamp(bi, 0, 1)))
    layer = (np.asarray(canvas.filter(ImageFilter.GaussianBlur(0.8)), np.float32)
             + 2.0 * np.asarray(canvas.filter(ImageFilter.GaussianBlur(4)), np.float32)) / 255.0
    layer = np.asarray(Image.fromarray(layer, "F").resize((W, H), Image.BILINEAR), np.float32)
    f += layer[..., None] * (np.array(color, np.float32) * strength)


_SRC = {}


def src(key):
    if key not in _SRC:
        _SRC[key] = load(PHOTOS[key][0])
    return _SRC[key]


def frame():
    return np.zeros((H, W, 3), np.float32)


# ---------------------------------------------------------------- clips
BOOMS = []          # 戦場の爆発（タイムライン構築時に埋まる）
FLASHES = []        # (時刻, 強さ, 減衰秒, 色)
CHROMA = []         # (時刻, ずれ px)


class Clip:
    start = 0.0
    dur = 1.0
    _ready = False

    def ensure(self):
        if not self._ready:
            self.prepare()
            self._ready = True

    def prepare(self):
        pass

    def render(self, u, t):
        return frame()


class Black(Clip):
    pass


class Shot(Clip):
    """写真の一部をシネスコで切り出し、a → b へ動かす1カット。a, b = (中心x, 中心y, 幅の割合)"""

    def __init__(self, key, a, b, look, fx=None, shake=0.0, year=None, ease=True, fade_in=0.0, fade_out=0.0):
        self.key, self.a, self.b, self.look = key, a, b, look
        self.fx, self.shake, self.year, self.ease = fx, shake, year, ease
        self.fade_in, self.fade_out = fade_in, fade_out

    def prepare(self):
        self.src = src(self.key)
        if self.year:
            self.y_lab = Label(self.year[0], font(66, True), WHITE, tracking=6)
            self.p_lab = Label(self.year[1], font(30, True), (236, 208, 150), tracking=8)

    def render(self, u, t):
        p = clamp(u / self.dur, 0, 1)
        e = 0.5 - 0.5 * math.cos(math.pi * p) if self.ease else p
        cx, cy, wf = (lerp(self.a[i], self.b[i], e) for i in range(3))
        sw, sh = self.src.size
        bw = wf * sw
        bh = bw / ASPECT
        if bh > sh:
            bh, bw = sh, sh * ASPECT
        x, y = cx * sw, cy * sh
        if self.shake:
            amp = self.shake
            for tb in BOOMS:
                if t >= tb:
                    amp += 22 * math.exp(-(t - tb) * 5)
            amp *= bw / W
            x += amp * (math.sin(t * 13.1) + 0.6 * math.sin(t * 31.7 + 1.3))
            y += amp * (math.cos(t * 11.3) + 0.6 * math.sin(t * 27.1 + 0.4))
        x, y = clamp(x, bw / 2, sw - bw / 2), clamp(y, bh / 2, sh - bh / 2)
        crop = self.src.resize((W, PH), Image.BICUBIC, box=(x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2))
        crop = crop.filter(ImageFilter.UnsharpMask(2, 60, 2))
        f = frame()
        f[PY0:PY1] = grade(np.asarray(crop, np.float32) / 255.0, **GRADES[self.look])
        if self.fx:
            particles(f, t, self.fx, seed=sum(map(ord, self.key)), strength=0.9 if self.fx == "embers" else 1.0)
        if self.year:
            o = ramp(u, 0.15, 0.35) * (1 - ramp(u, self.dur - 0.25, 0.25))
            dx = (1 - ramp(u, 0.15, 0.5)) * -24
            put_left(f, self.y_lab, 130 + dx, PY1 - 140, o)
            put_left(f, self.p_lab, 130 + dx + self.y_lab.w + 22, PY1 - 108, o)
        if self.fade_in or self.fade_out:
            f *= (ramp(u, 0, self.fade_in) if self.fade_in else 1.0) * \
                 (1 - ramp(u, self.dur - self.fade_out, self.fade_out) if self.fade_out else 1.0)
        return f


class Card(Clip):
    """黒地の文字カード（ゆっくり寄る）"""

    def __init__(self, text, size=64, color=WHITE):
        self.text, self.size, self.color = text, size, color

    def prepare(self):
        self.lab = Label(self.text, font(self.size), self.color, tracking=10, shadow=False)

    def render(self, u, t):
        f = frame()
        o = ramp(u, 0.0, 0.22) * (1 - ramp(u, self.dur - 0.3, 0.3))
        put(f, self.lab, W / 2, H / 2, o, 1.06 - 0.06 * u / self.dur)
        return f


class Logo(Clip):
    """架空の配給会社ロゴ：地平線から昇る太陽"""

    def prepare(self):
        n = 900
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = np.hypot(xx - n / 2, yy - n / 2)
        core = np.clip((112 - d) / 2, 0, 1)
        k = np.clip(d / 112, 0, 1)[..., None]
        core_c = np.array([1.0, 0.86, 0.58]) * (1 - k) + np.array([0.98, 0.5, 0.2]) * k
        glow = np.exp(-(d / 190) ** 2) * 0.65 * np.clip((n / 2 - d) / 120, 0, 1)
        alpha = np.maximum(core, glow)
        rgb = (core_c * core[..., None] + np.array([1.0, 0.55, 0.25]) * (glow * (1 - core))[..., None]) \
            / np.maximum(alpha, 1e-4)[..., None]
        self.sun = np.dstack([rgb, alpha]).astype(np.float32)
        self.name = Label("百年映画", font(72), WHITE, tracking=34, shadow=False)
        self.en = Label("HYAKUNEN  PICTURES", font(20), GOLD, tracking=12, shadow=False)

    def render(self, u, t):
        f = frame()
        n = self.sun.shape[0]
        hz = 590
        sy = lerp(hz + 200, hz - 30, 1 - (1 - clamp((u - 0.5) / 2.6, 0, 1)) ** 3)
        blit(f, self.sun, W / 2 - n / 2, sy - n / 2)
        f[hz + 1:PY1] = 0
        ln = int(1000 * smooth((u - 0.2) / 1.2))
        if ln:
            f[hz - 1:hz + 1, W // 2 - ln // 2:W // 2 + ln // 2] = np.array(CREAM, np.float32) / 255 * 0.85
        put(f, self.name, W / 2, 690, ramp(u, 1.8, 0.9))
        put(f, self.en, W / 2, 785, ramp(u, 2.3, 0.9))
        return f * (1 - ramp(u, self.dur - 0.7, 0.7))


class Rewind(Clip):
    """時計が逆回りし、2026 → 1936 へ年が巻き戻る"""
    R = 200
    CY = 430
    SPIN = 2.3

    def prepare(self):
        S, R = 2, self.R
        size = 2 * R + 40
        img = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c = size * S / 2
        d.ellipse((c - R * S, c - R * S, c + R * S, c + R * S), outline=CREAM + (220,), width=3 * S)
        for i in range(60):
            a = math.radians(i * 6)
            r0 = R - (24 if i % 5 == 0 else 11)
            d.line([(c + r0 * S * math.sin(a), c - r0 * S * math.cos(a)),
                    (c + (R - 6) * S * math.sin(a), c - (R - 6) * S * math.cos(a))],
                   fill=CREAM + (200,), width=(4 if i % 5 == 0 else 2) * S)
        self.face = np.asarray(img.resize((size, size), Image.LANCZOS), np.float32) / 255.0
        self.size = size
        self.years = {}
        self.era = Label("昭和十一年", font(32), GOLD, tracking=14, shadow=False)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.exp(-(((xx - W / 2) / 700) ** 2 + ((yy - self.CY) / 420) ** 2)) * 0.07
        self.bg = (g[..., None] * np.array([1.0, 0.8, 0.55], np.float32)).astype(np.float32)

    def angle(self, u):
        p = clamp(u / self.SPIN, 0, 1)
        return -2 * math.pi * 14 * (1 - (1 - p) ** 3)

    def render(self, u, t):
        f = self.bg.copy()
        S = 2
        n = self.size
        hands = Image.new("RGBA", (n * S, n * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(hands)
        c = n * S / 2
        for j, alpha in enumerate((255, 110, 60, 30)):
            th = self.angle(u - j * 0.018)
            for ang, ln, wd in ((th, self.R * 0.82, 4), (th / 12, self.R * 0.55, 7)):
                d.line([(c, c), (c + ln * S * math.sin(ang), c - ln * S * math.cos(ang))],
                       fill=WHITE + (alpha,), width=wd * S)
        d.ellipse((c - 9 * S, c - 9 * S, c + 9 * S, c + 9 * S), fill=GOLD + (255,))
        hands = np.asarray(hands.resize((n, n), Image.LANCZOS), np.float32) / 255.0
        o = ramp(u, 0, 0.25)
        blit(f, self.face, W / 2 - n / 2, self.CY - n / 2, o)
        blit(f, hands, W / 2 - n / 2, self.CY - n / 2, o)
        p = clamp(u / self.SPIN, 0, 1)
        year = int(round(2026 - 90 * (1 - (1 - p) ** 3)))
        if year not in self.years:
            self.years[year] = Label(str(year), font(110, True), WHITE, tracking=10, shadow=False)
        put(f, self.years[year], W / 2, 735, o)
        put(f, self.era, W / 2, 840, ramp(u, self.SPIN - 0.1, 0.25))
        f *= 1 + 0.05 * math.sin(u * 37)
        return f


class Quote(Clip):
    def prepare(self):
        self.stars = Label("★★★★★", font(40), GOLD, tracking=14, shadow=False)
        self.q = Label("「百年に一度の、変わらなさ。」", font(56), WHITE, tracking=4, shadow=False)
        self.src_l = Label("―― 月刊 空想シネマ", font(26), GOLD, tracking=6, shadow=False)

    def render(self, u, t):
        f = frame()
        o = ramp(u, 0, 0.25) * (1 - ramp(u, self.dur - 0.3, 0.3))
        k = 1.04 - 0.04 * u / self.dur
        put(f, self.stars, W / 2, 420, o, k)
        put(f, self.q, W / 2, 530, o, k)
        put(f, self.src_l, W / 2, 640, o * ramp(u, 0.3, 0.3), k)
        return f


class YearSlam(Clip):
    def __init__(self, year, era):
        self.year, self.era = year, era

    def prepare(self):
        self.big = Label(self.year, font(230, True), WHITE, tracking=20, shadow=False)
        self.small = Label(self.era, font(34), GOLD, tracking=16, shadow=False)

    def render(self, u, t):
        f = frame()
        k = lerp(1.25, 1.0, 1 - (1 - clamp(u / 0.18, 0, 1)) ** 2)
        put(f, self.big, W / 2, 500, 1.0, k)
        put(f, self.small, W / 2, 680, ramp(u, 0.15, 0.2))
        glitch(f, int(t * FPS) * 13 + 5, 1.0 - ramp(u, 0.05, 0.45) * 0.85)
        return f


class HUD(Clip):
    """架空の身元照合端末"""
    FIELDS = [("氏名", "■■ ■■", "w"), ("生年月日", "大正07年 04月01日", "w"), ("戸籍上の年齢", "108歳", "w"),
              ("外見からの推定", "20代前半", "w"), ("顔照合", "一致 ― 1946年 復員名簿", "w"),
              ("判定", "不一致 ／ 要確認", "r")]
    X0, Y0, STEP = 800, 300, 74
    FX, FY, FW, FH = 300, 245, 360, 450

    @classmethod
    def type_times(cls, start):
        """各フィールドの (開始時刻, 文字数)"""
        return [(start + 0.35 + i * 0.55 + 0.15, len(v)) for i, (_, v, _) in enumerate(cls.FIELDS)]

    def prepare(self):
        bg = frame()
        bg[PY0:PY1] = (0.012, 0.04, 0.065)
        grid = np.array([0.0, 0.045, 0.06], np.float32)
        for x in range(0, W, 48):
            bg[PY0:PY1, x] += grid
        for y in range(PY0, PY1, 48):
            bg[y, :] += grid
        im = src("present")
        sw, sh = im.size
        bw = 0.22 * sw
        bh = bw * self.FH / self.FW
        cx, cy = 0.475 * sw, 0.215 * sh
        face = im.resize((self.FW, self.FH), Image.LANCZOS, box=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        lum = np.asarray(face.convert("L"), np.float32)[..., None] / 255.0
        bg[self.FY:self.FY + self.FH, self.FX:self.FX + self.FW] = lum * np.array([0.45, 0.85, 0.95]) * 0.95
        cyan = np.array(CYAN, np.float32) / 255
        for (x, y, sx, sy) in ((self.FX - 12, self.FY - 12, 1, 1), (self.FX + self.FW + 12, self.FY - 12, -1, 1),
                               (self.FX - 12, self.FY + self.FH + 12, 1, -1),
                               (self.FX + self.FW + 12, self.FY + self.FH + 12, -1, -1)):
            xs = sorted((x, x + sx * 36))
            ys = sorted((y, y + sy * 36))
            bg[y - 1:y + 2, xs[0]:xs[1]] = cyan
            bg[ys[0]:ys[1], x - 1:x + 2] = cyan
        put_left(bg, Label("身元照合システム", mono(30), CYAN, tracking=6, shadow=False), self.X0, 190)
        put_left(bg, Label("CASE 2026-1111-0401", mono(20), (90, 150, 165), tracking=3, shadow=False), 1500, 200)
        bg[248:250, self.X0:1800] = cyan * 0.6
        put_left(bg, Label("対象者", mono(22), (90, 150, 165), tracking=6, shadow=False), self.FX, self.FY + self.FH + 30)
        self.bg = bg
        self.labels = [Label(k, mono(24), (95, 170, 185), shadow=False) for k, _, _ in self.FIELDS]
        self.values = [Label(v, mono(32), VERMILION if c == "r" else WHITE, tracking=2, shadow=False)
                       for _, v, c in self.FIELDS]

    def render(self, u, t):
        f = self.bg.copy()
        cyan = np.array(CYAN, np.float32) / 255
        sy = self.FY + int(self.FH * ((u * 0.9) % 1.0))
        for dy, a in ((0, 0.9), (-2, 0.45), (-5, 0.2), (-9, 0.08)):
            yy = sy + dy
            if self.FY <= yy < self.FY + self.FH:
                f[yy, self.FX:self.FX + self.FW] = f[yy, self.FX:self.FX + self.FW] * (1 - a) + cyan * a
        for i, (lab, val) in enumerate(zip(self.labels, self.values)):
            ts = 0.35 + i * 0.55
            if u < ts:
                break
            y = self.Y0 + i * self.STEP
            put_left(f, lab, self.X0, y + 4)
            n_chars = len(self.FIELDS[i][1])
            typed = clamp((u - ts - 0.15) * 30 / n_chars, 0, 1)
            o = 1.0
            if typed >= 1 and self.FIELDS[i][2] == "r":
                o = 1.0 if int(u * 5) % 2 == 0 else 0.3
            put_left(f, val, self.X0 + 250, y, o, typed)
            if 0 < typed < 1:
                xc = int(self.X0 + 250 + val.w * typed) + 4
                f[y + 8:y + 40, xc:xc + 14] = cyan
        glitch(f, int(t * FPS) * 7 + 1, 1 - ramp(u, 0, 0.3))
        return f


class Title(Clip):
    def prepare(self):
        base = Label("変わらない男", font(150, True), WHITE, tracking=30, shadow=False)
        a = base.arr
        h = a.shape[0]
        g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        rgb = np.array([1.0, 0.95, 0.86], np.float32) * (1 - g) + np.array([0.80, 0.62, 0.38], np.float32) * g
        rgb = np.broadcast_to(rgb, a[..., :3].shape)
        self.title = Label(image=Image.fromarray((np.dstack([rgb, a[..., 3:]]) * 255).astype(np.uint8), "RGBA"))
        glow = Image.fromarray((a[..., 3] * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(24))
        ga = np.asarray(glow, np.float32) / 255.0 * 0.7
        self.glow = Label(image=Image.fromarray(
            (np.dstack([np.broadcast_to(np.array([1.0, 0.62, 0.3], np.float32), a[..., :3].shape), ga]) * 255)
            .astype(np.uint8), "RGBA"))
        self.alpha = a[..., 3]
        self.en = Label("THE  UNCHANGING  MAN", font(26), GOLD, tracking=16, shadow=False)
        self.tag = Label("時代は変わる。男は、変わらない。", font(40), CREAM, tracking=8, shadow=False)

    def render(self, u, t):
        f = frame()
        particles(f, t, "embers", seed=91, strength=0.55)
        o = ramp(u, 0, 0.1) * (1 - ramp(u, self.dur - 0.6, 0.6))
        k = lerp(1.18, 1.0, 1 - (1 - clamp(u / 0.5, 0, 1)) ** 3)
        cy = 480
        put(f, self.glow, W / 2, cy, o * (0.7 + 0.3 * math.sin(u * 2)), k)
        put(f, self.title, W / 2, cy, o, k)
        if 0.7 < u < 2.2 and k == 1.0:                       # 光の筋
            h, w = self.alpha.shape
            x0, y0 = int(W / 2 - w / 2), int(cy - h / 2)
            pos = lerp(-300, w + 300, (u - 0.7) / 1.5)
            xs = np.arange(w, dtype=np.float32)[None, :]
            ys = np.arange(h, dtype=np.float32)[:, None]
            band = np.exp(-((xs - pos + 0.4 * ys) / 55) ** 2) * self.alpha * 0.85 * o
            f[y0:y0 + h, x0:x0 + w] += band[..., None]
        put(f, self.en, W / 2, 640, o * ramp(u, 1.0, 0.7))
        put(f, self.tag, W / 2, 760, o * ramp(u, 2.2, 0.9))
        return f


class Release(Clip):
    def prepare(self):
        self.big = Label("2027年 春", font(96, True), WHITE, tracking=12, shadow=False)
        self.sub = Label("全国ロードショー", font(40), CREAM, tracking=22, shadow=False)
        self.note = Label("この予告編はフィクションです。", font(22), GOLD, tracking=4, shadow=False)

    def render(self, u, t):
        f = frame()
        particles(f, t, "embers", seed=91, strength=0.35)
        o = ramp(u, 0.1, 0.5) * (1 - ramp(u, self.dur - 0.5, 0.5))
        put(f, self.big, W / 2, 470, o)
        put(f, self.sub, W / 2, 590, o * ramp(u, 0.4, 0.5))
        put(f, self.note, W / 2, 870, o * ramp(u, 0.8, 0.5))
        return f


class EndTag(Clip):
    def prepare(self):
        self.t = Label("変わらない男", font(56, True), WHITE, tracking=16, shadow=False)
        self.s = Label("2027年 春　全国ロードショー", font(26), GOLD, tracking=8, shadow=False)

    def render(self, u, t):
        f = frame()
        o = ramp(u, 0.1, 0.4) * (1 - ramp(u, self.dur - 0.8, 0.8))
        put(f, self.t, W / 2, 500, o)
        put(f, self.s, W / 2, 590, o)
        return f


# ---------------------------------------------------------------- timeline
class Timeline:
    def __init__(self):
        self.clips, self.subs, self.M, self.t = [], [], {}, 0.0

    def add(self, clip, dur, name=None, subs=()):
        dur = round(dur * FPS) / FPS
        clip.start, clip.dur = self.t, dur
        if name:
            self.M[name] = self.t
        for s, a, b in subs:
            self.subs.append((s, self.t + a, self.t + b))
        self.clips.append(clip)
        self.t += dur
        return clip


def build_timeline():
    tl = Timeline()
    add = tl.add
    # --- 配給ロゴ
    add(Logo(), 5.0, "logo")
    # --- 第一幕: 取調べ
    add(Black(), 0.6)
    add(Shot("present", (0.475, 0.215, 0.40), (0.475, 0.215, 0.30), "cold", fx="dust"), 3.4, "p1",
        subs=[("名前は？", 0.3, 1.5), ("生年月日は？", 1.8, 3.3)])
    add(Black(), 1.9, "answer", subs=[("大正七年、四月一日です。", 0.2, 1.8)])
    add(Shot("present", (0.5, 0.30, 1.0), (0.49, 0.29, 0.86), "cold", fx="dust"), 2.4, "p2",
        subs=[("……ふざけてるのか？", 0.3, 2.3)])
    add(Card("その男は、"), 1.6, "c1")
    add(Shot("student", (0.49, 0.37, 0.45), (0.49, 0.37, 0.41), "sepia", ease=False), 0.45, "f1")
    add(Card("百八年間、"), 1.4, "c2")
    add(Shot("conscription", (0.55, 0.365, 0.34), (0.55, 0.365, 0.31), "mono", ease=False), 0.45, "f2")
    add(Card("一度も、歳をとらなかった。"), 2.1, "c3")
    add(Rewind(), 2.8, "rewind")
    # --- 第二幕: 百年の記憶
    add(Shot("student", (0.5, 0.42, 1.0), (0.5, 0.37, 0.62), "sepia", year=("1936", "東京")), 2.8, "s1",
        subs=[("学者になるんだ。百年かかっても。", 0.4, 2.7)])
    add(Shot("student", (0.54, 0.08, 0.55), (0.54, 0.20, 0.5), "sepia"), 1.8, "s2")
    add(Shot("conscription", (0.5, 0.47, 1.0), (0.53, 0.40, 0.8), "mono", year=("1938", "徴兵検査")), 2.5, "s3")
    add(Shot("conscription", (0.55, 0.38, 0.42), (0.55, 0.365, 0.32), "mono"), 2.4, "s4",
        subs=[("長生き、するんだよ。", 0.3, 2.3)])
    s5 = add(Shot("battle", (0.5, 0.5, 1.0), (0.47, 0.46, 0.78), "war", fx="embers", shake=5,
                  year=("1944", "戦地")), 2.6, "s5", subs=[("走れ！　振り向くな！", 0.5, 2.5)])
    add(Shot("battle", (0.45, 0.42, 0.46), (0.45, 0.42, 0.38), "war", fx="embers", shake=6), 1.3, "s6")
    s6b = add(Shot("battle", (0.74, 0.33, 0.5), (0.74, 0.31, 0.42), "war", fx="embers", shake=8), 1.0, "s6b")
    add(Card("時代は、変わる。"), 1.8, "c4")
    add(Shot("ueno", (0.38, 0.12, 0.8), (0.47, 0.47, 0.6), "faded", year=("1946", "上野")), 3.0, "s7",
        subs=[("ただいま、母さん。", 1.0, 2.9)])
    add(Shot("ueno", (0.81, 0.6, 0.38), (0.81, 0.6, 0.32), "faded"), 2.6, "s8",
        subs=[("あんた……ちっとも変わらないねえ。", 0.2, 2.5)])
    add(Shot("alley", (0.45, 0.40, 0.95), (0.44, 0.34, 0.56), "showa", year=("1958", "下町")), 2.8, "s9",
        subs=[("みんな、先に歳をとっていった。", 0.4, 2.7)])
    add(Shot("alley", (0.25, 0.2, 0.55), (0.5, 0.2, 0.55), "showa"), 1.6, "s10")
    add(Card("男は、変わらない。"), 1.8, "c5")
    add(Quote(), 2.2, "quote")
    # --- 第三幕: 現在
    add(YearSlam("2026", "令和八年"), 0.8, "y2026")
    add(HUD(), 4.2, "hud")
    add(Shot("present", (0.475, 0.212, 0.30), (0.475, 0.212, 0.25), "cold", fx="dust"), 1.8, "s12",
        subs=[("お前は……何者だ。", 0.2, 1.7)])
    add(Card("百八年分の、記憶。"), 1.4, "c6")
    montage = [
        ("student", (0.49, 0.37, 0.42), "sepia"), ("conscription", (0.88, 0.38, 0.24), "mono"),
        ("battle", (0.75, 0.38, 0.42), "war"), ("ueno", (0.38, 0.13, 0.55), "faded"),
        ("alley", (0.63, 0.53, 0.36), "showa"), ("present", (0.55, 0.14, 0.4), "cold"),
        ("conscription", (0.87, 0.6, 0.26), "mono"), ("ueno", (0.76, 0.22, 0.3), "faded"),
        ("alley", (0.2, 0.22, 0.4), "showa"), ("conscription", (0.53, 0.35, 0.5), "mono"),
        ("student", (0.69, 0.48, 0.42), "sepia"), ("ueno", (0.81, 0.6, 0.34), "faded"),
        ("battle", (0.45, 0.42, 0.4), "war"), ("alley", (0.43, 0.33, 0.4), "showa"),
        ("present", (0.475, 0.215, 0.32), "cold"),
    ]
    durs = [0.5, 0.46, 0.42, 0.39, 0.36, 0.33, 0.31, 0.29, 0.27, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2]
    cuts = []
    for i, ((key, box, look), d) in enumerate(zip(montage, durs)):
        c = add(Shot(key, box, (box[0], box[1], box[2] * 0.94), look, ease=False,
                     fx="embers" if key == "battle" else None, shake=4 if key == "battle" else 0), d, f"m{i}")
        cuts.append(c.start)
        if i % 2 == 0:
            FLASHES.append((c.start, 0.65, 0.05, (1.0, 1.0, 1.0)))
        CHROMA.append((c.start, 9))
    add(Card("そして、たったひとつの約束。"), 2.0, "c7")
    add(Black(), 2.8, "whisper", subs=[("母さん。……長生き、しすぎたかな。", 0.4, 2.6)])
    add(Title(), 6.0, "title")
    add(Release(), 4.0, "release")
    add(Black(), 0.7)
    add(Shot("present", (0.5, 0.30, 0.95), (0.5, 0.29, 0.88), "cold", fx="dust", fade_in=0.3), 5.0, "stinger",
        subs=[("……で、若さの秘訣は？", 0.4, 2.3), ("早寝、早起きです。", 2.7, 4.9)])
    add(EndTag(), 3.0, "endtag")
    tl.end = tl.t
    tl.cuts = cuts

    M = tl.M
    BOOMS.extend([s5.start + 0.15, s5.start + 1.6, s6b.start + 0.05])
    for tb in BOOMS:
        FLASHES.append((tb, 0.45, 0.12, (1.0, 0.62, 0.3)))
        CHROMA.append((tb, 8))
    for name in ("f1", "f2", "y2026"):
        FLASHES.append((M[name], 0.6, 0.06, (1.0, 1.0, 1.0)))
        CHROMA.append((M[name], 10))
    FLASHES.append((M["s1"], 0.85, 0.15, (1.0, 0.95, 0.85)))
    FLASHES.append((M["title"], 0.9, 0.25, (1.0, 0.92, 0.8)))
    CHROMA.append((M["title"], 14))
    return tl


TL = build_timeline()
STARTS = [c.start for c in TL.clips]


# ---------------------------------------------------------------- compositor
_G = {}


def _init_worker():
    rng = np.random.default_rng(4321)
    _G["grain"] = []
    for _ in range(6):
        g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
        _G["grain"].append(np.asarray(Image.fromarray(g, "F").resize((W, H), Image.BILINEAR), np.float32)[..., None])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (PH / 2 + 60)) ** 2) / math.sqrt(2)
    _G["vig"] = np.clip(1 - 0.45 * r ** 2.2, 0, 1)[..., None].astype(np.float32)
    _G["subs"] = [(Label(s, font(36), CREAM), a, b) for s, a, b in TL.subs]


def render_frame(fi):
    t = fi / FPS
    i = max(0, bisect.bisect_right(STARTS, t + 1e-6) - 1)
    clip = TL.clips[i]
    clip.ensure()
    f = clip.render(t - clip.start, t)
    for tf, peak, dec, col in FLASHES:
        d = t - tf
        if 0 <= d < dec * 6:
            k = peak * math.exp(-d / dec)
            f = f * (1 - k) + np.array(col, np.float32) * k
    for tc, px in CHROMA:
        d = t - tc
        if 0 <= d < 0.4:
            chroma(f, px * math.exp(-d * 10))
    f *= _G["vig"]
    f += _G["grain"][(fi // 2) % 6] * GRAIN_AMT
    f[:PY0] = 0
    f[PY1:] = 0
    for lab, a, b in _G["subs"]:
        if a <= t < b:
            put(f, lab, W / 2, 1010, ramp(t, a, 0.12) * (1 - ramp(t, b - 0.15, 0.15)))
    f *= 1 - ramp(t, TL.end - 0.5, 0.5)
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()


# ---------------------------------------------------------------- audio
def _t(sec):
    return np.arange(int(sec * SR)) / SR


def braam(length=3.2, root=55.0):
    t = _t(length)
    s = np.zeros_like(t)
    for mult, amp in ((1, 1.0), (1.5, 0.55), (2, 0.7), (3, 0.3), (4, 0.18)):
        for det in (0.996, 1.0, 1.004):
            fr = root * mult * det
            s += amp * (2 * ((fr * t + _ar.random()) % 1) - 1)
    dark = sosfilt(_sos("low", 260), s)
    bright = sosfilt(_sos("low", 1700), s)
    eb = np.exp(-t * 2.2) * np.minimum(1, t / 0.06)
    s = dark * (1 - eb) + bright * eb
    s = np.tanh(s / np.abs(s).max() * 2.5)
    env = np.minimum(1, t / 0.035) * np.exp(-t * 0.65) * np.clip((length - t) / 0.8, 0, 1)
    out = s * env + 0.7 * np.sin(2 * np.pi * root / 2 * t) * np.exp(-t * 1.1) * np.minimum(1, t / 0.02)
    return out / np.abs(out).max()


def hit(length=2.5):
    t = _t(length)
    sub = np.sin(2 * np.pi * np.cumsum(34 + 70 * np.exp(-t * 14)) / SR) * np.exp(-t * 2.0)
    body = sosfilt(_sos("low", 1500), _ar.standard_normal(len(t)))
    body = body / np.abs(body).max() * np.exp(-t * 9)
    crack = sosfilt(_sos("high", 3000), _ar.standard_normal(len(t)))
    crack = crack / np.abs(crack).max() * np.exp(-t * 35)
    s = (sub + 0.5 * body + 0.25 * crack) * np.minimum(1, t / 0.001)
    return s / np.abs(s).max()


def crash(length=3.0):
    t = _t(length)
    s = sosfilt(_sos("high", 4500), _ar.standard_normal(len(t))) * np.exp(-t * 1.4)
    return s / np.abs(s).max()


def whoosh(length=0.6, up=True):
    t = _t(length)
    n = _ar.standard_normal(len(t))
    lo, hi = sosfilt(_sos("low", 700), n), sosfilt(_sos("band", [1500, 6000]), n)
    lo, hi = lo / np.abs(lo).max(), hi / np.abs(hi).max()
    x = t / length if up else 1 - t / length
    s = lo * (1 - x) + hi * x
    return s * np.sin(np.pi * np.clip(t / length, 0, 1)) ** 2 * (t / length if up else 1)


def tick(high=True):
    t = _t(0.06)
    s = sosfilt(_sos("band", [2500, 7000]), _ar.standard_normal(len(t))) * np.exp(-t * 300)
    s += np.sin(2 * np.pi * (3300 if high else 2500) * t) * np.exp(-t * 160) * 0.5
    return s / np.abs(s).max()


def taiko(vel=1.0):
    t = _t(1.0)
    body = np.sin(2 * np.pi * np.cumsum(58 + 75 * np.exp(-t * 20)) / SR) * np.exp(-t * 4.5)
    skin = sosfilt(_sos("low", 900), _ar.standard_normal(len(t)))
    skin = skin / np.abs(skin).max() * np.exp(-t * 40)
    s = (body + 0.35 * skin) * np.minimum(1, t / 0.002)
    return s / np.abs(s).max() * vel


def beep(freq):
    t = _t(0.035)
    return np.sin(2 * np.pi * freq * t) * np.sin(np.pi * t / 0.035)


def alarm():
    out = []
    for i in range(6):
        t = _t(0.16)
        fr = 880 if i % 2 == 0 else 660
        out.append((np.sin(2 * np.pi * fr * t) + 0.3 * np.sin(6 * np.pi * fr * t)) * np.minimum(1, (0.16 - t) / 0.02))
    return np.concatenate(out)


def glitch_sfx(length=0.6):
    t = _t(length)
    s = np.zeros_like(t)
    seg = int(0.03 * SR)
    for i in range(0, len(t), seg):
        kind = _ar.integers(0, 3)
        tt = t[i:i + seg]
        if kind == 0:
            s[i:i + seg] = np.sign(np.sin(2 * np.pi * _ar.uniform(200, 2500) * tt))
        elif kind == 1:
            s[i:i + seg] = np.round(_ar.standard_normal(len(tt)) * 2) / 2
    return sosfilt(_sos("low", 6000), s) * np.exp(-t * 3)


def shimmer(length=4.5):
    t = _t(length)
    s = np.zeros_like(t)
    for nm in ("C6", "E6", "G6", "B6", "D7"):
        fr = nfreq(nm)
        s += np.sin(2 * np.pi * fr * t) * (0.6 + 0.4 * np.sin(2 * np.pi * _ar.uniform(3, 7) * t + _ar.random() * 6))
    env = np.minimum(1, t / 1.2) * np.clip((length - t) / 1.5, 0, 1)
    return s * env / 5


STR = {"F": ["F3", "A3", "C4"], "G": ["G3", "B3", "D4"], "Em": ["E3", "G3", "B3"], "Am": ["A3", "C4", "E4"],
       "C": ["C4", "E4", "G4"]}


def build_audio(tl):
    M = tl.M
    n = int((tl.end + 1.5) * SR)
    wet = np.zeros((2, n))
    dry = np.zeros((2, n))

    def add(bus, sig, t0, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= n or i0 < 0:
            return
        seg = sig[: n - i0] * gain
        th = (pan + 1) * math.pi / 4
        bus[0, i0:i0 + len(seg)] += seg * math.cos(th)
        bus[1, i0:i0 + len(seg)] += seg * math.sin(th)

    def note(name, t0, vel, length, pan=None):
        t0 += _ar.normal(0, 0.005)
        vel *= 1 + _ar.normal(0, 0.06)
        if pan is None:
            pan = clamp((nfreq(name) - 400) / 900, -0.45, 0.45)
        add(wet, piano(name, vel, length), t0, pan=pan)

    def theme(t0, chords, mel, vel=0.3, strings=0.22):
        for bi, ch in enumerate(chords):
            b0 = t0 + bi * BAR
            for j, nm in enumerate(CHORDS[ch]):
                tt = b0 + j * BEAT / 2
                note(nm, tt, vel * (1.15 if j == 0 else 0.85), b0 + BAR - tt + 0.35)
            if strings:
                add(wet, pad([nfreq(x) for x in STR[ch]], BAR + 1.2, attack=0.9, release=1.0), b0, gain=strings)
        tt = t0
        for nm, beats in mel:
            if nm:
                note(nm, tt, 0.42, beats * BEAT + 0.5)
            tt += beats * BEAT

    def big_hit(t0, g=1.0, root=55.0, blen=3.2):
        add(dry, braam(blen, root), t0, gain=0.8 * g)
        add(dry, hit(), t0, gain=0.55 * g)
        add(wet, hit(), t0, gain=0.25 * g)

    # 配給ロゴ
    add(wet, shimmer(4.6), 0.2, gain=0.35)
    add(dry, hit(), 0.9, gain=0.12)
    for i, nm in enumerate(("C3", "G3", "E4")):
        note(nm, 2.0 + i * 0.08, 0.2, 3.0)

    # 第一幕: 時計の音、ドローン、ブラーム
    add(wet, pad([55.0, 82.4, 110.0], M["c1"] - M["p1"] + 1.5, attack=2.0, release=1.2), M["p1"] - 0.6, gain=0.45)
    tt, k = M["p1"], 0
    while tt < M["c1"] - 0.05:
        add(wet, tick(k % 2 == 0), tt, gain=0.14)
        tt += 0.5
        k += 1
    big_hit(M["c1"], 0.85)
    for name in ("f1", "f2"):
        add(dry, whoosh(0.45, up=True), M[name] - 0.42, gain=0.35)
        add(dry, hit(), M[name], gain=0.45)
    add(dry, hit(), M["c2"], gain=0.25)
    big_hit(M["c3"], 1.0, root=49.0, blen=3.0)
    # 巻き戻し: 秒針が猛烈に逆回りして止まる
    R = M["rewind"]
    add(dry, whoosh(Rewind.SPIN, up=False), R, gain=0.3)
    for q in range(1, 14 * 4 + 1):
        e = q / 56
        p = 1 - (1 - e) ** (1 / 3)
        add(wet, tick(q % 2 == 0), R + p * Rewind.SPIN, gain=0.10)
    add(dry, hit(), M["s1"], gain=0.5)
    add(dry, crash(), M["s1"], gain=0.12)

    # 第二幕: テーマ
    theme(M["s1"], ["F", "G", "Em"], [("A4", 1.5), ("G4", .5), ("A4", 1), ("C5", 1),
                                      ("D5", 1.5), ("C5", .5), ("B4", 1), ("G4", 1),
                                      ("B4", 1.5), ("A4", .5), ("G4", 1), ("E4", 1)])
    B = M["s5"]
    add(wet, pad([73.4, 110.0, 146.8, 155.6], M["c4"] - B + 1.5, attack=0.8, release=1.2), B, gain=0.5)
    rum = sosfilt(_sos("low", 110, 4), _ar.standard_normal(int((M["c4"] - B + 0.5) * SR)))
    rum = rum / np.abs(rum).max() * np.minimum(1, _t((M["c4"] - B + 0.5)) / 0.4)
    add(dry, rum, B, gain=0.3)
    for i, tb in enumerate(BOOMS):
        add(dry, boom(), tb, gain=0.75, pan=(-0.3, 0.35, 0.0)[i])
    for name in ("s5", "s6", "s6b"):
        add(dry, taiko(), M[name], gain=0.45)
    add(dry, hit(), M["c4"], gain=0.45)
    note("A1", M["c4"], 0.35, 3.0)
    add(wet, whistle(2.2), M["s7"] + 0.3, gain=0.09, pan=-0.3)
    crowd = noise_bed(M["c5"] - M["s7"], 280, 1600, mod_rate=5)
    tc = _t(M["c5"] - M["s7"])
    crowd *= np.minimum(1, tc / 1.0) * np.clip((tc[-1] - tc) / 1.0, 0, 1)
    add(dry, crowd, M["s7"], gain=0.045)
    theme(M["s7"], ["Am", "F", "C", "G"], [("E5", 2), ("C5", 2), ("A4", 2), ("C5", 1), ("D5", 1),
                                           ("E5", 1.5), ("D5", .5), ("C5", 1), ("G4", 1), ("D5", 3), (None, 1)],
          vel=0.3, strings=0.28)
    add(dry, hit(), M["c5"], gain=0.45)
    note("C6", M["quote"] + 0.1, 0.22, 2.5)

    # 第三幕
    big_hit(M["y2026"], 0.6, root=58.3, blen=2.0)
    add(dry, glitch_sfx(0.7), M["y2026"], gain=0.35)
    add(wet, pad([58.3, 82.4, 116.5], M["c7"] - M["y2026"] + 0.5, attack=3.0, release=0.4),
        M["y2026"], gain=0.5)
    for ts, nch in HUD.type_times(M["hud"]):
        for c in range(nch):
            add(dry, beep(_ar.uniform(1300, 1900)), ts + c / 30, gain=0.05)
    last_ts, last_n = HUD.type_times(M["hud"])[-1]
    add(dry, alarm(), last_ts + last_n / 30, gain=0.09)
    add(dry, hit(), M["s12"], gain=0.4)
    big_hit(M["c6"], 0.7)
    mont_len = M["c7"] - M["m0"]
    rt = _t(mont_len)
    rise = sosfilt(_sos("high", 2500), _ar.standard_normal(len(rt))) * (rt / mont_len) ** 2.5
    rise += np.sin(2 * np.pi * np.cumsum(150 + 900 * (rt / mont_len) ** 2) / SR) * (rt / mont_len) ** 2 * 0.5
    add(dry, rise / np.abs(rise).max(), M["m0"], gain=0.3)
    for i, c in enumerate(tl.cuts):
        add(dry, taiko(1.0 if i % 2 == 0 else 0.75), c, gain=0.4 + 0.3 * i / len(tl.cuts),
            pan=(-0.15 if i % 2 else 0.15))
    big_hit(M["c7"], 1.1, root=41.2, blen=4.0)
    add(dry, crash(), M["c7"], gain=0.2)
    note("A4", M["whisper"] + 0.5, 0.12, 4.0)

    # タイトル
    big_hit(M["title"], 1.25, root=55.0, blen=4.5)
    add(dry, boom(), M["title"], gain=0.5)
    add(dry, crash(), M["title"], gain=0.3)
    T = M["title"] + 1.3
    theme(T, ["F", "G"], [("A4", 1.5), ("G4", .5), ("A4", 1), ("C5", 1),
                          ("D5", 1.5), ("E5", .5), ("D5", 1), ("B4", 1), ("C5", 4)], vel=0.28, strings=0.25)
    for i, nm in enumerate(["C2", "G2", "C3", "G3", "C4", "E4", "G4"]):
        note(nm, T + 2 * BAR + i * 0.045, 0.3, 6.0)
    add(wet, pad([nfreq(x) for x in ("C4", "E4", "G4")], 6.0, attack=0.6, release=2.5), T + 2 * BAR, gain=0.25)
    # おまけ
    add(wet, rin(), M["stinger"] + 3.0, gain=0.3, pan=0.1)
    for i, nm in enumerate(("C5", "E5", "G5")):
        note(nm, M["endtag"] + 0.2 + i * 0.06, 0.13, 3.0)

    # ミックス
    wet = np.stack([sosfilt(_sos("low", 7500), ch) for ch in wet])
    tir = _t(2.4)
    rev = []
    for c in range(2):
        ir = sosfilt(_sos("low", 4500), _ar.standard_normal(len(tir))) * np.exp(-tir * 2.8)
        ir[: int(0.015 * SR)] = 0
        rev.append(fftconvolve(wet[c], ir / np.sqrt(np.sum(ir ** 2)))[:n])
    music = wet + 0.3 * np.stack(rev)
    music *= 0.55 / np.abs(music).max()
    mix = np.tanh((music + dry * 0.85) * 1.3) / np.tanh(1.3)
    tm = np.arange(n) / SR
    mix *= np.clip((tl.end - tm) / 0.6, 0, 1)
    return mix * 0.95 / np.abs(mix).max()


# ---------------------------------------------------------------- main
def stills():
    out = os.path.join(BUILD, "trailer_stills")
    os.makedirs(out, exist_ok=True)
    _init_worker()
    M = TL.M
    picks = {"logo": M["logo"] + 3.5, "p1": M["p1"] + 2.5, "c3": M["c3"] + 1.0, "rewind": M["rewind"] + 1.2,
             "rewind_end": M["rewind"] + 2.6, "s1": M["s1"] + 1.6, "s2": M["s2"] + 1.0, "s4": M["s4"] + 1.5,
             "s5": M["s5"] + 0.6, "s7": M["s7"] + 2.4, "s8": M["s8"] + 1.5, "s9": M["s9"] + 1.8, "s10": M["s10"] + 0.8,
             "quote": M["quote"] + 1.2, "y2026": M["y2026"] + 0.5, "hud": M["hud"] + 4.0, "m4": M["m4"] + 0.2,
             "m6": M["m6"] + 0.2, "c7": M["c7"] + 1.0, "title": M["title"] + 3.5, "release": M["release"] + 2.0,
             "stinger": M["stinger"] + 3.5}
    for name, t in picks.items():
        Image.frombuffer("RGB", (W, H), render_frame(int(round(t * FPS)))).save(os.path.join(out, f"{name}.jpg"), quality=88)
    print("stills ->", out, f"(total {TL.end:.1f}s)")


def main():
    os.makedirs(BUILD, exist_ok=True)
    if "--stills" in sys.argv:
        stills()
        return
    nframes = int(round(TL.end * FPS))
    print(f"total {TL.end:.1f}s / {nframes} frames")
    wav = os.path.join(BUILD, "trailer_audio.wav")
    write_wav(wav, build_audio(TL))
    print("audio done")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "slow", "-crf", CRF, "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
           "-metadata", "title=変わらない男 予告編", OUTPUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init_worker) as pool:
        for i, buf in enumerate(pool.imap(render_frame, range(nframes), chunksize=4)):
            proc.stdin.write(buf)
            if i % 300 == 0:
                print(f"frame {i}/{nframes}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()

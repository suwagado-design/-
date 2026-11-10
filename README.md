# 変わらない男

6枚の写真から、ある男の一生を振り返る約1分45秒の映像です。

▶ **[life_retrospective.mp4](life_retrospective.mp4)**（1920×1080 / 30fps / 音声つき）

## 構成

| 時代 | 章 | 写真 |
| --- | --- | --- |
| 令和八年（2026） | プロローグ — 鉄格子の向こうの男 | `assets/06_present_2026.jpg` |
| 昭和十一年（1936）・十八歳 | 第一章「青雲」— 東京市の学生 | `assets/01_student_1936.png` |
| 昭和十三年（1938）・二十歳 | 第二章「検査」— 徴兵検査 | `assets/02_conscription_1938.jpg` |
| 昭和十九年（1944）・二十六歳 | 第三章「戦火」— 戦地 | `assets/03_battlefield_1944.png` |
| 昭和二十一年（1946）・二十八歳 | 第四章「帰還」— 上野駅 | `assets/04_ueno_1946.png` |
| 昭和三十三年（1958）・四十歳 | 第五章「灯り」— 下町の路地 | `assets/05_alley_1958.png` |
| 令和八年（2026）・百八歳 | 終章「そして、現在」 | `assets/06_present_2026.jpg` |

どの写真でも顔が変わらないので、物語もそれを中心に組み立てています。
映像はフィクションです。

## 作り直す

```sh
# 必要: python3, numpy, scipy, Pillow, ffmpeg (libx264), 日本語明朝フォント (Noto Serif CJK JP か IPA明朝)
python3 make_video.py            # life_retrospective.mp4 を書き出す
python3 make_video.py --stills   # レイアウト確認用の静止画を build/stills/ に書き出す
```

- 映像は Pillow + numpy で1フレームずつ描画しています（ゆっくりズームするケン・バーンズ効果、フィルムグレイン、ちらつき、クロスフェード、爆発の閃光と揺れ）。
- 音楽と効果音（ピアノ、ドローン、小太鼓、爆発音、汽笛、雑踏、鉄扉、おりん）はすべてコードで合成しており、外部の音源は使っていません。
- セリフや年代、章の長さは `make_video.py` の `SCENES` で変えられます。

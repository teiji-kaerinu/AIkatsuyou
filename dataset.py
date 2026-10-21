"""
dataset.py ― 卒研の「dataset」フォルダの代わりになる合成数字データ

卒研のデータ: 0〜9 の 2値(白黒) BMP 画像、各クラス100枚（計1000枚）、
ファイル名は「5-001.bmp」形式。imageDatastore(..., 'LabelSource','foldernames')
で読み込んで 28x28 にリサイズしていた（samplev43.m 10, 48行目）。

ここでは同じ形（28x28・値は 0/1・各クラス100枚）のデータを
5x7 ドットフォント 2 書体 + ランダム変形（拡大・斜体・太さ・かすれ）で作る。
※卒研の BMP は白背景(1)に黒字(0)だったと思われる（reformation28 の色反転処理から推測）。
  ここでは CNN で扱いやすい黒背景(0)に白字(1)にしている。
"""
import numpy as np

# 1 文字 = 7 行 × 5 列。"1" が数字の線
FONT_A = {
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
FONT_B = {
    0: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    1: ["00100", "01100", "10100", "00100", "00100", "00100", "11111"],
    2: ["01110", "10001", "00010", "00100", "01000", "10000", "11111"],
    3: ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
    4: ["10010", "10010", "10010", "11111", "00010", "00010", "00010"],
    5: ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    6: ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "10001", "00010", "00100", "00100", "00100", "00100"],
    8: ["01110", "10001", "01010", "00100", "01010", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
}


def _dilate(a):
    """十字型の膨張（線を 1 画素太くする）"""
    p = np.pad(a, 1)
    return np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])


def _render(glyph, rng, size):
    g = np.array([[int(c) for c in row] for row in glyph], dtype=float)
    # 拡大（縦横の倍率をばらつかせる）
    h, w = rng.integers(14, 22), rng.integers(9, 16)
    img = g[np.arange(h) * 7 // h][:, np.arange(w) * 5 // w]
    # 斜体（行ごとに横へずらす）
    shear = rng.uniform(-0.35, 0.35)
    out = np.zeros((h, w + 8))
    for r in range(h):
        s = 4 + int(round(shear * (r - h / 2)))
        out[r, s:s + w] = img[r]
    # 太さ
    if rng.random() < 0.5:
        out = _dilate(out)
    # 線の一部がかすれる
    if rng.random() < 0.3:
        r0, c0 = rng.integers(0, h - 3), rng.integers(0, w + 5)
        out[r0:r0 + 3, c0:c0 + 3] = 0
    # 28x28 のキャンバスの中央付近（±3 画素ずらす）に置く
    canvas = np.zeros((size, size))
    oh, ow = out.shape
    top = np.clip((size - oh) // 2 + rng.integers(-3, 4), 0, size - oh)
    left = np.clip((size - ow) // 2 + rng.integers(-3, 4), 0, size - ow)
    canvas[top:top + oh, left:left + ow] = out
    return canvas


def make_digits(n_per_class=100, size=28, seed=0):
    """卒研と同じ構成（10クラス × n_per_class 枚）のデータを作る。
    戻り値: X (N, size, size) 値は 0/1,  y (N,) ラベル 0〜9"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for d in range(10):
        for _ in range(n_per_class):
            font = FONT_A if rng.random() < 0.5 else FONT_B
            X.append(_render(font[d], rng, size))
            y.append(d)
    return np.array(X), np.array(y)


def salt_and_pepper(X, level, rng):
    """imnoise(I, 'salt & pepper', level) 相当: 割合 level の画素をランダムに 0 か 1 にする"""
    X = X.copy()
    mask = rng.random(X.shape) < level
    X[mask] = rng.integers(0, 2, mask.sum())
    return X


def median_filter(X, k=3):
    """medfilt2(I, [k k]) 相当（端はゼロ埋め）。X: (N, H, W)"""
    p = k // 2
    P = np.pad(X, ((0, 0), (p, p), (p, p)))
    H, W = X.shape[1:]
    win = np.stack([P[:, i:i + H, j:j + W] for i in range(k) for j in range(k)], axis=-1)
    return np.median(win, axis=-1)


def show_pair(a, b, title=""):
    """2 枚の画像を横に並べて文字で表示（matplotlib が無くても確認できるように）"""
    print(title)
    for ra, rb in zip(a, b):
        print("".join("#" if v > 0.5 else "." for v in ra) + "    " + "".join("#" if v > 0.5 else "." for v in rb))


if __name__ == "__main__":
    X, y = make_digits()
    print(X.shape, y.shape, "クラスごとの枚数:", np.bincount(y))
    rng = np.random.default_rng(1)
    for i in [30, 530]:
        show_pair(X[i], median_filter(salt_and_pepper(X[i:i + 1], 0.05, rng))[0],
                  f"ラベル {y[i]}  （左: 元画像 / 右: ノイズ5% → メディアンフィルタ）")

"""
step2_cnn.py ― CNN を NumPy だけで組んで学習し、ノイズに対する強さを調べる

卒研の layers（samplev43.m 167〜187行目）:
    imageInputLayer([28 28 1])
    convolution2dLayer(5, 17, 'Padding','same')  batchNormalizationLayer  reluLayer
    maxPooling2dLayer(2, 'Stride',[1 1])
    convolution2dLayer(5, 17, 'Padding','same')  batchNormalizationLayer  reluLayer
    maxPooling2dLayer(2, 'Stride',[1 1])
    convolution2dLayer(5, 25, 'Padding','same')  batchNormalizationLayer  reluLayer
    fullyConnectedLayer(10)  softmaxLayer  classificationLayer
（17 と 25 は fix(2*(Groups+1)/2)*2+1, fix(3*(Groups+1)/2)*2+1 に Groups=7 を入れた値）

NumPy では重いので、同じ並びのままフィルタ数を減らし、
プーリングを Stride 2（縦横半分）にして小さくしている。
"""
import time
import numpy as np
from dataset import make_digits, salt_and_pepper, median_filter, show_pair
from layers import Conv2D, BatchNorm, ReLU, MaxPool2D, Affine, Sequential, SGDMomentum, train, gradient_check


def build_cnn(seed=0):
    rng = np.random.default_rng(seed)
    return Sequential([
        Conv2D(1, 8, 5, rng), BatchNorm(8), ReLU(),
        MaxPool2D(2, stride=2),                       # 28x28 → 14x14
        Conv2D(8, 16, 5, rng), BatchNorm(16), ReLU(),
        MaxPool2D(2, stride=2),                       # 14x14 → 7x7
        Affine(16 * 7 * 7, 10, rng),                  # fullyConnectedLayer(10)
    ])


if __name__ == "__main__":
    X, y = make_digits()
    X = X[:, None]  # (N, 1, 28, 28)

    print("■ 勾配チェック（畳み込み・BatchNorm・プーリングの逆伝播が正しいか）")
    worst = gradient_check(build_cnn(1), X[::125], y[::125])
    print(f"  最大のずれ {worst:.1e}  （1e-5 程度以下なら正しい）\n")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    test, train_idx = idx[:200], idx[200:800]

    print("■ 学習 A: 元画像のまま学習（卒研と同じ。sgdm, MiniBatchSize 50, 学習データ 600 枚）")
    net = build_cnn()
    t0 = time.time()
    train(net, X[train_idx], y[train_idx], epochs=10, batch=50, opt=SGDMomentum(lr=0.01))
    print(f"  学習時間 {time.time() - t0:.1f} 秒\n")

    print("■ 学習 B: 学習データにも同じメディアンフィルタをかけてから学習")
    net_b = build_cnn()
    train(net_b, median_filter(X[train_idx, 0])[:, None], y[train_idx], epochs=10, batch=50,
          opt=SGDMomentum(lr=0.01), verbose=False)
    print("  （途中経過は省略）\n")

    print("■ テスト（salt & pepper ノイズ → 3x3 メディアンフィルタ。samplev43.m 50〜64行目と同じ処理）")
    print("  ノイズ率   A:フィルタなし   A:メディアンあり   B:メディアンあり")
    Xt, yt = X[test, 0], y[test]
    acc = lambda n, Z: (n.proba(Z[:, None]).argmax(1) == yt).mean() * 100
    for level in [0, 0.001, 0.02, 0.05, 0.1]:
        noisy = salt_and_pepper(Xt, level, rng)
        med = median_filter(noisy)
        note = "  ← 卒研の NoiseLevel" if level == 0.001 else ""
        print(f"  {level:6.3f}     {acc(net, noisy):6.1f}%         {acc(net, med):6.1f}%"
              f"           {acc(net_b, med):6.1f}%{note}")
    print("  ※ ノイズが少ない時（卒研の 0.001 など）は、メディアンフィルタが線の角を削る分だけ下がる。")
    print("    ノイズが多い（10%）と逆にフィルタが効く。テストは 200 枚なので 1 枚 = 0.5%、数 % の差はばらつきの範囲。")

    print("\n■ 間違えた例（ノイズ 5% + メディアンフィルタ）")
    noisy = median_filter(salt_and_pepper(Xt, 0.05, rng))
    P = net.proba(noisy[:, None])
    wrong = np.flatnonzero(P.argmax(1) != yt)
    for i in wrong[:2]:
        show_pair(Xt[i], noisy[i], f"正解 {yt[i]} → 判定 {P[i].argmax()}（確率 {P[i].max():.2f}）"
                                   "   左: 元画像 / 右: 入力した画像")
    if len(wrong) == 0:
        print("  なし")

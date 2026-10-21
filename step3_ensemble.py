"""
step3_ensemble.py ― samplev43.m の「アイデア」を縮小して再現する

卒研コードの 1 ループの流れ:
  [1] テスト 200 枚を抜き、ノイズ + メディアンフィルタをかける                  （46〜64行目）
  [2] CNN を 5 個。CNN ごとに残り 800 枚から学習 600 / 検証 200 を選び直す      （127〜207行目）
  [3] 手作り特徴 59 次元 + 浅いネット patternnet(59) を 3 個                    （86〜163行目）
  [4] 検証データで「間違えた」「当たったが自信が低い」サンプルを集め、
      それを重点的に学ぶ「誤り学習 CNN」を作る。1 個作るごとに学習データ全体を
      見直し、新しく間違えたサンプルを追加していく                              （280〜532行目）
  [5] 全モデルの出力（10 クラスの確率）を 1 本に並べ、それを入力にした
      「融合 BP ネット」patternnet(36) で最終判定する                           （576〜649行目）
  これを LOOPs = 12 回くり返して平均精度を出していた。

ここでは NumPy で回る大きさに縮めている（CNN 3 個・浅いネット 2 個・誤り学習 CNN 2 個）。
また卒研の誤り学習 CNN は「回帰」で "どのクラスと間違えやすいか" まで教えていたが、
ここでは「難しいサンプルを 3 倍に増やして学ぶ分類 CNN」に単純化している。

実行: python step3_ensemble.py        （1 ループ、2 分ほど）
      python step3_ensemble.py 3      （3 ループの平均）
"""
import sys
import time
import numpy as np
from dataset import make_digits, salt_and_pepper, median_filter
from layers import Sequential, Affine, Tanh, Adam, train
from step2_cnn import build_cnn

N_CNN, N_SHALLOW, N_ERR = 3, 2, 2   # 卒研: numNetworks=5, nummanualNetworks=3, numNetworksErr=12〜19
TRAIN, VAL, TEST = 600, 200, 200
NOISE = 0.05                        # 卒研は NoiseLevel = 1e-3（ほぼノイズなし）。差を見るため強めにした
EPOCHS = 10


def hand_features(X):
    """getFeature59_28 の代わり（元の関数は samplev43.m に入っていないので中身は不明）。
    行ごと・列ごとの線の量 (28+28) + 線の割合 + 縦横比 + 重心の高さ = 59 次元"""
    rows, cols = X.mean(axis=2), X.mean(axis=1)
    ink = X.mean(axis=(1, 2))
    h = np.maximum(X.any(axis=2).sum(axis=1), 1)
    w = X.any(axis=1).sum(axis=1)
    cy = (X.sum(axis=2) * np.arange(X.shape[1])).sum(axis=1) / np.maximum(X.sum(axis=(1, 2)), 1) / X.shape[1]
    return np.column_stack([rows, cols, ink, w / h, cy])


def collect_hard(models, X, y, pool, thr):
    """どれか 1 つのモデルでも「間違えた」(Level1) か「当たったが確率 < thr」(Level2) の
    サンプル番号を集める（卒研の ErrFiles1 にあたる。重複は 1 回だけ数える）"""
    hard = set()
    for f in models:
        P = f(X[pool])
        bad = (P.argmax(axis=1) != y[pool]) | (P.max(axis=1) < thr)
        hard.update(pool[bad].tolist())
    return np.array(sorted(hard), dtype=int)


def fusion_net(Z, y, seed):
    """融合 BP: patternnet(36) 相当。並べた出力 Z から最終判定を学ぶ"""
    rng = np.random.default_rng(seed)
    net = Sequential([Affine(Z.shape[1], 36, rng), Tanh(), Affine(36, 10, rng)])
    return train(net, Z, y, epochs=100, opt=Adam(lr=0.003), seed=seed, verbose=False)


def run_once(seed):
    rng = np.random.default_rng(seed)
    X, y = make_digits()
    idx = rng.permutation(len(X))
    test, rest = idx[:TEST], idx[TEST:]
    Xt, yt = median_filter(salt_and_pepper(X[test], NOISE, rng)), y[test]
    as_fn = lambda net: (lambda Z: net.proba(Z[:, None]))  # ネット → 「画像を入れると確率が返る関数」

    # [2] CNN
    cnns, trs, vals = [], [], []
    for k in range(N_CNN):
        perm = rng.permutation(rest)
        tr, va = perm[:TRAIN], perm[TRAIN:TRAIN + VAL]
        net = train(build_cnn(seed * 10 + k), X[tr, None], y[tr], epochs=EPOCHS, seed=k, verbose=False)
        cnns.append(as_fn(net))
        trs.append(tr)
        vals.append(va)
        print(f"  CNN {k + 1}/{N_CNN} 学習完了")

    # [3] 浅いネット（CNN と同じ学習データを使う。卒研 162行目）
    F = hand_features(X[rest])
    lo, hi = F.min(axis=0), F.max(axis=0)
    feat = lambda Z: (hand_features(Z) - lo) / np.maximum(hi - lo, 1e-9)  # mapminmax(F, 0, 1)
    shallows = []
    for m in range(N_SHALLOW):
        r = np.random.default_rng(seed * 10 + 5 + m)
        net = Sequential([Affine(59, 59, r), Tanh(), Affine(59, 10, r)])
        train(net, feat(X[trs[m]]), y[trs[m]], epochs=100, opt=Adam(lr=0.003), seed=m, verbose=False)
        shallows.append(lambda Z, net=net: net.proba(feat(Z)))

    # [4] 誤り学習 CNN
    hard = collect_hard(cnns, X, y, np.unique(np.concatenate(vals)), thr=0.625)
    print(f"  検証データで見つかった難しいサンプル: {len(hard)} 枚")
    errs = []
    for e in range(N_ERR):
        tr = np.concatenate([rng.permutation(rest)[:TRAIN], np.repeat(hard, 3)])  # 難しいサンプルを 3 倍に
        net = train(build_cnn(seed * 10 + 8 + e), X[tr, None], y[tr], epochs=EPOCHS, seed=e, verbose=False)
        errs.append(as_fn(net))
        hard = np.union1d(hard, collect_hard([errs[-1]], X, y, rest, thr=0.75))  # 全体を見直して追加
        print(f"  誤り学習 CNN {e + 1}/{N_ERR} 学習完了（難しいサンプルは累計 {len(hard)} 枚）")

    # [5] 融合 BP
    # 学習データは「検証セット k に全モデルの出力を並べたもの」を k 個ぶん（卒研 224〜251行目）。
    # ※検証セット k の一部は別の CNN の学習データでもある。そこでは出力が自信過剰になるので、
    #   融合 BP は「本番より当たりやすい出力」を見て学ぶことになる（後述の改善ポイント）。
    Xv = np.concatenate([X[va] for va in vals])
    yv = np.concatenate([y[va] for va in vals])
    stack = lambda fns, Z: np.hstack([f(Z) for f in fns])  # allCnnOutputs0 を作る処理
    base = cnns + shallows
    fus1 = fusion_net(stack(base, Xv), yv, seed)
    fus2 = fusion_net(stack(base + errs, Xv), yv, seed)

    acc = lambda P: (P.argmax(axis=1) == yt).mean() * 100
    return {
        "従来法（CNN 単体の平均）": np.mean([acc(f(Xt)) for f in cnns]),
        "浅いネット単体の平均（参考）": np.mean([acc(f(Xt)) for f in shallows]),
        "CNN の出力を平均しただけ（単純アンサンブル）": acc(np.mean([f(Xt) for f in cnns], axis=0)),
        "融合 BP（誤り学習なし）": acc(fus1.proba(stack(base, Xt))),
        "融合 BP（誤り学習あり）": acc(fus2.proba(stack(base + errs, Xt))),
    }


if __name__ == "__main__":
    loops = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t0 = time.time()
    results = []
    for loop in range(loops):
        print(f"---------- {loop + 1} ループ目 ----------")
        results.append(run_once(loop))
        for name, v in results[-1].items():
            print(f"  {v:5.1f}%  {name}")
    print(f"\n========== {loops} ループの平均（テスト: ノイズ {NOISE * 100:.0f}% → メディアンフィルタ）==========")
    for name in results[0]:
        print(f"  {np.mean([r[name] for r in results]):5.1f}%  {name}")
    print(f"経過時間 {time.time() - t0:.0f} 秒")

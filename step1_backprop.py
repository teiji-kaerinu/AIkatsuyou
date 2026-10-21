"""
step1_backprop.py ― バックプロパゲーションを「数式そのまま」で書く

卒研の patternnet(59)（隠れ層 59 個の浅いネット、samplev43.m 160行目）と同じ形:

    入力 x ─[W1,b1]→ u1 ─tanh→ z1 ─[W2,b2]→ u2 ─softmax→ y
    損失 E = −Σ t·log(y)   （t は正解のワンホット、交差エントロピー）

部品(クラス)を使わず、行列の掛け算だけで
    1. 順伝播   2. 逆伝播（連鎖律）   3. 数値微分で答え合わせ   4. 学習
を全部見えるように書いてある。step2 以降はこれを部品化したもの（layers.py）を使う。
"""
import numpy as np
from dataset import make_digits

rng = np.random.default_rng(0)

# ---- データ（卒研と同じく テスト200 / 学習600）----
X, y = make_digits()
X = X.reshape(len(X), -1)            # 28x28 の画像 → 784 次元のベクトル
T = np.eye(10)[y]                    # ind2vec 相当: ラベル 3 → [0,0,0,1,0,0,0,0,0,0]
idx = rng.permutation(len(X))
test, train = idx[:200], idx[200:800]

# ---- パラメータ（重み）----
n_in, n_hid, n_out = 784, 59, 10
W1 = rng.normal(0, np.sqrt(1 / n_in), (n_in, n_hid))
b1 = np.zeros(n_hid)
W2 = rng.normal(0, np.sqrt(1 / n_hid), (n_hid, n_out))
b2 = np.zeros(n_out)


def forward(x):
    u1 = x @ W1 + b1
    z1 = np.tanh(u1)                 # patternnet の隠れ層の活性化関数は tansig (= tanh)
    u2 = z1 @ W2 + b2
    e = np.exp(u2 - u2.max(axis=1, keepdims=True))
    y = e / e.sum(axis=1, keepdims=True)   # softmax: 出力を「合計 1 の確率」にする
    return z1, y


def loss(x, t):
    _, y = forward(x)
    return -np.sum(t * np.log(y + 1e-12)) / len(x)


def backward(x, t):
    """連鎖律を出力側から 1 段ずつさかのぼる"""
    z1, y = forward(x)
    d2 = (y - t) / len(x)            # ∂E/∂u2: softmax + 交差エントロピーは「出力 − 正解」になる
    dW2 = z1.T @ d2                  # ∂E/∂W2 = z1ᵀ · δ2
    db2 = d2.sum(axis=0)
    dz1 = d2 @ W2.T                  # 誤差を 1 つ前の層へ「逆向きに伝える」← ここが back propagation
    d1 = dz1 * (1 - z1 ** 2)         # tanh の微分 (1 − tanh²) を掛ける
    dW1 = x.T @ d1                   # ∂E/∂W1 = xᵀ · δ1
    db1 = d1.sum(axis=0)
    return dW1, db1, dW2, db2


def accuracy(ids):
    return (forward(X[ids])[1].argmax(axis=1) == y[ids]).mean() * 100


# ---- 1〜3. 勾配チェック ----
print("■ 勾配チェック: 数値微分 (E(w+e) - E(w-e)) / 2e と、逆伝播で出した勾配を比べる")
xb, tb = X[train[:20]], T[train[:20]]
grads = backward(xb, tb)
eps = 1e-6
for name, P, G in [("W1", W1, grads[0]), ("b1", b1, grads[1]), ("W2", W2, grads[2]), ("b2", b2, grads[3])]:
    j = np.unravel_index(np.abs(G).argmax(), G.shape)  # 勾配が一番大きい要素で比べる
    old = P[j]
    P[j] = old + eps
    lp = loss(xb, tb)
    P[j] = old - eps
    lm = loss(xb, tb)
    P[j] = old
    print(f"  {name}{list(map(int, j))}: 数値微分 {(lp - lm) / (2 * eps):+.8f}   逆伝播 {G[j]:+.8f}")
print("  → ほぼ同じ値なら backward() は正しい。数値微分は遅いので学習には逆伝播を使う。\n")

# ---- 4. 学習（ミニバッチ確率的勾配降下法）----
print("■ 学習: w ← w − η·∂E/∂w をミニバッチごとに繰り返す")
print(f"  学習前  テスト正解率 {accuracy(test):5.1f}%  (でたらめなら約10%)")
lr, epochs, batch = 0.1, 30, 50
for ep in range(1, epochs + 1):
    perm = rng.permutation(train)
    for s in range(0, len(perm), batch):
        b = perm[s:s + batch]
        dW1, db1, dW2, db2 = backward(X[b], T[b])
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    if ep % 5 == 0:
        print(f"  epoch {ep:2d}  損失 {loss(X[train], T[train]):.4f}  "
              f"学習データ {accuracy(train):5.1f}%  テスト {accuracy(test):5.1f}%")

print("\n  学習データとテストの正解率の差 ＝ 過学習の度合い。")
print("  全結合ネットは「画素の位置」をそのまま覚えるので、数字の位置がずれると弱い。")
print("  → それを解決するのが step2 の CNN（畳み込み）。")

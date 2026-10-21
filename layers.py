"""
layers.py ― NumPy だけで作るニューラルネットの部品

MATLAB（Deep Learning Toolbox）の層と 1 対 1 で対応させてある。

    MATLAB                                      このファイル
    ------------------------------------------  ------------------------------
    convolution2dLayer(5, 17, 'Padding','same')  Conv2D(c_in, 17, 5, rng)
    batchNormalizationLayer                     BatchNorm(チャネル数)
    reluLayer / tanhLayer                       ReLU() / Tanh()
    maxPooling2dLayer(2, 'Stride',[1 1])        MaxPool2D(2, stride=1)
    fullyConnectedLayer(10)                     Affine(n_in, 10, rng)
    softmaxLayer + classificationLayer          SoftmaxCrossEntropy（Sequential に内蔵）
    trainingOptions('sgdm') / ('adam')          SGDMomentum() / Adam()
    trainNetwork(X, Y, layers, options)         train(net, X, y, ...)
    predict(net, X)                             net.proba(X)

どの層も次の 2 つだけを持つ。
    forward(x)    入力 → 出力（順伝播）
    backward(d)   「損失 E の出力についての勾配 ∂E/∂out」を受け取り、
                  「入力についての勾配 ∂E/∂in」を返す（逆伝播）。
                  ついでにパラメータの勾配 ∂E/∂W を self.grads に置く。
全部の層の backward を後ろから順に呼べば、連鎖律で全パラメータの勾配がそろう。
これがバックプロパゲーション（誤差逆伝播法）。

画像データの形は (N, C, H, W) = (枚数, チャネル, 縦, 横)。
MATLAB の (H, W, C, N)（testData_Noise(:,:,1,l) の並び）とは順番が逆なので注意。
"""
import numpy as np


class Layer:
    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x, train=False):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


# ---------------------------------------------------------------- 全結合・活性化
class Affine(Layer):
    """fullyConnectedLayer: out = x·W + b"""

    def __init__(self, n_in, n_out, rng):
        super().__init__()
        self.params["W"] = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))  # He の初期値
        self.params["b"] = np.zeros(n_out)

    def forward(self, x, train=False):
        self.x_shape = x.shape
        self.x = x.reshape(len(x), -1)  # 画像なら平らにのばす
        return self.x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.x.T @ dout           # ∂E/∂W = xᵀ·δ
        self.grads["b"] = dout.sum(axis=0)
        return (dout @ self.params["W"].T).reshape(self.x_shape)  # ∂E/∂x = δ·Wᵀ


class ReLU(Layer):
    def forward(self, x, train=False):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask  # 正だった所だけ勾配を通す


class Tanh(Layer):
    def forward(self, x, train=False):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dout):
        return dout * (1 - self.y ** 2)  # tanh'(u) = 1 - tanh(u)^2


# ---------------------------------------------------------------- 畳み込み・プーリング
def im2col(x, k, stride, pad):
    """画像から k×k の窓を全部切り出し、1 窓 = 1 行の行列にする。
    こうすると畳み込みが「行列の掛け算 1 回」で済む（for 文で画素を回すより桁違いに速い）。"""
    N, C, H, W = x.shape
    OH, OW = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    img = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    col = np.empty((N, C, k, k, OH, OW))
    for i in range(k):
        for j in range(k):
            col[:, :, i, j] = img[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1), OH, OW


def col2im(col, x_shape, k, stride, pad, OH, OW):
    """im2col の逆。重なった窓の勾配は足し合わせて元の画像の形に戻す"""
    N, C, H, W = x_shape
    col = col.reshape(N, OH, OW, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for i in range(k):
        for j in range(k):
            img[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride] += col[:, :, i, j]
    return img[:, :, pad:H + pad, pad:W + pad]


class Conv2D(Layer):
    """convolution2dLayer(k, c_out, 'Padding','same')"""

    def __init__(self, c_in, c_out, k, rng, same=True):
        super().__init__()
        self.k, self.pad = k, (k // 2 if same else 0)
        self.params["W"] = rng.normal(0, np.sqrt(2 / (c_in * k * k)), (c_out, c_in, k, k))
        self.params["b"] = np.zeros(c_out)

    def forward(self, x, train=False):
        self.x_shape = x.shape
        self.col, self.OH, self.OW = im2col(x, self.k, 1, self.pad)   # (N*OH*OW, C*k*k)
        W = self.params["W"].reshape(len(self.params["b"]), -1)       # (c_out, C*k*k)
        out = self.col @ W.T + self.params["b"]                       # (N*OH*OW, c_out)
        return out.reshape(len(x), self.OH, self.OW, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        W = self.params["W"]
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, len(W))
        self.grads["W"] = (dout.T @ self.col).reshape(W.shape)  # 全結合と同じ形の式
        self.grads["b"] = dout.sum(axis=0)
        dcol = dout @ W.reshape(len(W), -1)
        return col2im(dcol, self.x_shape, self.k, 1, self.pad, self.OH, self.OW)


class MaxPool2D(Layer):
    """maxPooling2dLayer(size, 'Stride', stride)"""

    def __init__(self, size=2, stride=2):
        super().__init__()
        self.size, self.stride = size, stride

    def forward(self, x, train=False):
        N, C, H, W = self.x_shape = x.shape
        col, self.OH, self.OW = im2col(x.reshape(N * C, 1, H, W), self.size, self.stride, 0)
        self.arg = col.argmax(axis=1)  # どの画素が最大だったかを覚えておく
        return col.max(axis=1).reshape(N, C, self.OH, self.OW)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        dcol = np.zeros((dout.size, self.size * self.size))
        dcol[np.arange(dout.size), self.arg] = dout.ravel()  # 最大だった画素にだけ勾配を返す
        dx = col2im(dcol, (N * C, 1, H, W), self.size, self.stride, 0, self.OH, self.OW)
        return dx.reshape(self.x_shape)


class BatchNorm(Layer):
    """batchNormalizationLayer: ミニバッチ内でチャネルごとに平均0・分散1にそろえてから γ倍してβ足す"""

    def __init__(self, n, momentum=0.9, eps=1e-5):
        super().__init__()
        self.params["gamma"], self.params["beta"] = np.ones(n), np.zeros(n)
        self.mean, self.var = np.zeros(n), np.ones(n)  # 推論用（学習中の移動平均）
        self.momentum, self.eps = momentum, eps

    @staticmethod
    def _flat(x):  # (N,C,H,W) → (N*H*W, C)
        return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1]) if x.ndim == 4 else x

    def _unflat(self, x2):
        if len(self.shape) == 2:
            return x2
        N, C, H, W = self.shape
        return x2.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    def forward(self, x, train=False):
        self.shape = x.shape
        x2 = self._flat(x)
        if train:
            mu, var = x2.mean(axis=0), x2.var(axis=0)
            self.mean = self.momentum * self.mean + (1 - self.momentum) * mu
            self.var = self.momentum * self.var + (1 - self.momentum) * var
        else:
            mu, var = self.mean, self.var
        self.std = np.sqrt(var + self.eps)
        self.xn = (x2 - mu) / self.std
        return self._unflat(self.params["gamma"] * self.xn + self.params["beta"])

    def backward(self, dout):
        d2 = self._flat(dout)
        self.grads["gamma"] = (d2 * self.xn).sum(axis=0)
        self.grads["beta"] = d2.sum(axis=0)
        dxn = d2 * self.params["gamma"]
        # 平均と分散もバッチから計算しているので、その分の勾配も含めた式になる
        dx = (dxn - dxn.mean(axis=0) - self.xn * (dxn * self.xn).mean(axis=0)) / self.std
        return self._unflat(dx)


# ---------------------------------------------------------------- 出力層とネット全体
def softmax(u):
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """softmaxLayer + classificationLayer。
    softmax と交差エントロピーをまとめて微分すると ∂E/∂u = (y - t) / N ととても簡単になる。"""

    def forward(self, u, t):
        self.y, self.t = softmax(u), t
        return -np.log(self.y[np.arange(len(t)), t] + 1e-12).mean()

    def backward(self):
        d = self.y.copy()
        d[np.arange(len(self.t)), self.t] -= 1
        return d / len(self.t)


class Sequential:
    """MATLAB の layers = [ ... ] 配列にあたるもの"""

    def __init__(self, layers):
        self.layers = layers
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, x, train=False):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def loss(self, x, t):
        return self.loss_layer.forward(self.forward(x, train=True), t)

    def backward(self):
        d = self.loss_layer.backward()
        for layer in reversed(self.layers):  # 後ろから順に ＝ 誤差を「逆」に「伝播」
            d = layer.backward(d)

    def proba(self, x, batch=200):
        """predict(net, X) 相当: 各クラスの確率 (N, クラス数)"""
        return np.concatenate([softmax(self.forward(x[i:i + batch])) for i in range(0, len(x), batch)])


# ---------------------------------------------------------------- 学習アルゴリズム
class SGDMomentum:
    """trainingOptions('sgdm') 相当（MATLAB の既定値: 学習率 0.01, モーメンタム 0.9）
        v ← 0.9·v − η·∂E/∂w,   w ← w + v"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum, self.v = lr, momentum, {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            for k, p in layer.params.items():
                v = self.momentum * self.v.get((i, k), 0) - self.lr * layer.grads[k]
                self.v[(i, k)] = v
                p += v


class Adam:
    """trainingOptions('adam') 相当（既定値: 学習率 0.001, β1=0.9, β2=0.999）"""

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            for k, p in layer.params.items():
                g = layer.grads[k]
                m = self.b1 * self.m.get((i, k), 0) + (1 - self.b1) * g
                v = self.b2 * self.v.get((i, k), 0) + (1 - self.b2) * g * g
                self.m[(i, k)], self.v[(i, k)] = m, v
                mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(net, X, y, epochs=10, batch=50, opt=None, seed=0, verbose=True):
    """trainNetwork 相当。'Shuffle','every-epoch' と同じく毎エポック順番を入れ替える"""
    rng = np.random.default_rng(seed)
    opt = opt or SGDMomentum()
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        total = 0.0
        for s in range(0, len(X), batch):
            b = idx[s:s + batch]
            total += net.loss(X[b], y[b]) * len(b)  # 1) 順伝播して損失を出す
            net.backward()                          # 2) 逆伝播で全パラメータの勾配を出す
            opt.step(net.layers)                    # 3) 勾配の逆向きに少しだけ動かす
        if verbose:
            acc = (net.proba(X).argmax(axis=1) == y).mean() * 100
            print(f"  epoch {ep:2d}  loss {total / len(X):.4f}  学習データ正解率 {acc:5.1f}%")
    return net


def gradient_check(net, x, t, n=3, eps=1e-5, seed=0):
    """数値微分 (E(w+ε) − E(w−ε)) / 2ε と、バックプロパゲーションの勾配を比べる。
    一致していれば backward の実装は正しい。"""
    rng = np.random.default_rng(seed)
    net.loss(x, t)
    net.backward()
    grads = [{k: g.copy() for k, g in layer.grads.items()} for layer in net.layers]
    worst = 0.0
    for i, layer in enumerate(net.layers):
        for k, p in layer.params.items():
            G = grads[i][k]
            cand = np.flatnonzero(np.abs(G) > 1e-7)  # 勾配が 0 の所を比べても意味がないので除く
            if len(cand) == 0:
                # 例: BatchNorm の直前の層のバイアス b。BN が平均を引くので b を動かしても損失は変わらない
                print(f"  層{i:2d} {type(layer).__name__:9s} {k:5s}  勾配がすべてほぼ 0（損失に効かないパラメータ）")
                continue
            errs = []
            for f in rng.choice(cand, size=min(n, len(cand)), replace=False):
                j = np.unravel_index(f, G.shape)
                old = p[j]
                p[j] = old + eps
                lp = net.loss(x, t)
                p[j] = old - eps
                lm = net.loss(x, t)
                p[j] = old
                num = (lp - lm) / (2 * eps)
                errs.append(abs(num - G[j]) / max(abs(num) + abs(G[j]), 1e-12))
            worst = max(worst, max(errs))
            print(f"  層{i:2d} {type(layer).__name__:9s} {k:5s}  数値微分とのずれ(相対) {max(errs):.1e}")
    return worst

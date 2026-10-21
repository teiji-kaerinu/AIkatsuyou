"""
check.py ― exercises.py の答え合わせ

    python check.py             ○ / × とヒントを表示（- はまだ答えていない問題）
    python check.py --answers   まだ合っていない問題の正解も表示（まずは自力で）

レベル C の傾きは、数値微分（本当に少し動かして測る）で出した値と比べている。
"""
import math
import sys

try:
    import exercises as ex
except Exception as e:  # 書きかけで文法エラーがある時など
    print(f"exercises.py を読み込めませんでした: {e!r}")
    sys.exit(1)

SHOW = "--answers" in sys.argv
score = [0, 0]


def close(got, exp, tol=1e-3):
    if isinstance(exp, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(exp)
                and all(close(g, e, tol) for g, e in zip(got, exp)))
    try:
        return abs(float(got) - exp) <= tol * max(1.0, abs(exp))
    except (TypeError, ValueError):
        return False


def blank(got):
    return got is None or (isinstance(got, (list, tuple)) and all(g is None for g in got))


def fmt(v):
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    return f"{v + 0.0:.5g}" if isinstance(v, float) else str(v)  # + 0.0 で -0 を 0 と表示


def report(name, got, exp, hint):
    score[1] += 1
    if isinstance(got, Exception):
        mark, msg = "×", f"エラー: {got!r}"
    elif blank(got):
        mark, msg = "-", "まだ"
    elif close(got, exp):
        mark, msg = "○", ""
        score[0] += 1
    else:
        mark, msg = "×", f"あなたの答え {fmt(got)}   ヒント: {hint}"
    if SHOW and mark != "○":
        msg += f"   正解: {fmt(exp)}"
    print(f"  {mark} {name}  {msg}")


def call(f, *args):
    try:
        return f(*args)
    except Exception as e:
        return e


def num_grad(loss, params, eps=1e-6):
    """数値微分: パラメータを 1 個ずつ ±eps 動かして、損失の変化から傾きを出す"""
    g = []
    for i in range(len(params)):
        p = list(params)
        p[i] += eps
        up = loss(*p)
        p[i] -= 2 * eps
        g.append((up - loss(*p)) / (2 * eps))
    return g


def gd_step(w, lr, x=2.0, t=6.0):
    return w - lr * 2 * (w * x - t) * x


def relu_net(x, t, w1, w2):
    """lesson_basics.py の 2 番と同じネット。(y, E, ∂E/∂w1, ∂E/∂w2) を返す"""
    u1 = w1 * x
    z1 = max(0.0, u1)
    y = w2 * z1
    dy = y - t
    return y, 0.5 * dy ** 2, dy * w2 * (1.0 if u1 > 0 else 0.0) * x, dy * z1


print("レベル A：数字を変えて予想する")
report("A1 最初の傾き", ex.A1_grad, 2 * (5 * 2 - 6) * 2.0, "傾き = 2 × (w×x − t) × x に w = 5 を入れる")
report("A1 更新後の w", ex.A1_w_next, gd_step(5.0, 0.05), "w − 0.05 × 傾き。傾きがプラスなら w は減る")
report("A2 学習率 0.125 で 1 回後", ex.A2_w_after1_lr0125, gd_step(1.0, 0.125),
       "最初の傾きは −16。w は −0.125 × (−16) だけ動く")
report("A2 学習率 0.25 で 2 回後", ex.A2_w_after2_lr025, gd_step(gd_step(1.0, 0.25), 0.25),
       "1 回目の w を出したら、その場所で傾きを計算し直す")
y, E, dw1, dw2 = relu_net(-2.0, 3.0, 0.5, 1.5)
report("A3 出力 y", ex.A3_y, y, "u1 = 0.5 × (−2) は負。ReLU を通すと z1 はいくつ？")
report("A3 ∂E/∂w1", ex.A3_dw1, dw1, "ReLU の負の所は傾き 0。逆伝播はそこで止まる")
report("A3 ∂E/∂w2", ex.A3_dw2, dw2, "∂E/∂w2 = ∂E/∂y × z1。z1 はいくつ？")

print("\nレベル B：手で計算してから確かめる")
y, E, dw1, dw2 = relu_net(2.0, 1.0, 0.5, 1.5)
report("B1 損失 E", ex.B1_E, E, "y = 1.5 は変わらない。½(y − t)² の t を 1 にする")
report("B1 ∂E/∂w1", ex.B1_dw1, dw1, "∂E/∂y = y − t = +0.5 から、t = 3 の時と同じ順番で戻る")
report("B1 ∂E/∂w2", ex.B1_dw2, dw2, "∂E/∂w2 = ∂E/∂y × z1")
_, _, g1, g2 = relu_net(2.0, 3.0, 0.95, 1.65)
report("B2 2 回目の後の w1", ex.B2_w1, 0.95 - 0.1 * g1, "まず y = 1.65 × 1.9。∂E/∂y = y − 3 から逆伝播して w ← w − 0.1 × 傾き")
report("B2 2 回目の後の w2", ex.B2_w2, 1.65 - 0.1 * g2, "∂E/∂w2 = ∂E/∂y × z1（z1 = 1.9）")

print("\nレベル C：関数を書く")


def ref_fit(data, w, b, lr, steps):
    for _ in range(steps):
        gw = sum(2 * (w * x + b - t) * x for x, t in data) / len(data)
        gb = sum(2 * (w * x + b - t) for x, t in data) / len(data)
        w, b = w - lr * gw, b - lr * gb
    return w, b


data = [(1.0, 5.0), (3.0, 9.0)]
report("C1 1 回だけ更新", call(ex.fit_line, data, 0.0, 0.0, 0.05, 1), ref_fit(data, 0.0, 0.0, 0.05, 1),
       "2 つのデータで傾きを出して平均し、w と b を同時に更新する")
report("C1 2000 回更新", call(ex.fit_line, data, 0.0, 0.0, 0.05, 2000), ref_fit(data, 0.0, 0.0, 0.05, 2000),
       "1 回目が合っていれば、steps 回 for 文でくり返すだけ。2 点を通る直線に近づくはず")

for i, (x, t, w1, w2) in enumerate([(2.0, 3.0, 0.5, 1.5), (-1.0, 0.5, 0.3, -2.0)], 1):
    exp = num_grad(lambda a, b: 0.5 * (b * math.tanh(a * x) - t) ** 2, [w1, w2])
    report(f"C2 tanh の例{i}", call(ex.grads_tanh, x, t, w1, w2), exp,
           "z1 = tanh(u1)。ReLU の傾き (1 か 0) の代わりに 1 − z1² を掛ける")

for i, (x, t, a1, a2, v1, v2) in enumerate([(2.0, 3.0, 0.5, -0.4, 1.0, 0.7), (1.5, -1.0, 0.8, 0.6, -0.5, 1.2)], 1):
    loss = lambda a1, a2, v1, v2: 0.5 * (v1 * max(0.0, a1 * x) + v2 * max(0.0, a2 * x) - t) ** 2
    report(f"C3 2 ニューロンの例{i}", call(ex.grads_two_hidden, x, t, a1, a2, v1, v2), num_grad(loss, [a1, a2, v1, v2]),
           "∂E/∂v1 = ∂E/∂y × z1、∂E/∂z1 = ∂E/∂y × v1 → ReLU の傾き → × x。2 本目も同じ")

u = [1.0, 2.0, 3.0]
e = [math.exp(v) for v in u]
report("C4 softmax", call(ex.softmax, u), [v / sum(e) for v in e], "各 exp(u_i) を、全部の exp の合計で割る")
for c in [0, 2]:
    exp = num_grad(lambda *uu: -math.log(math.exp(uu[c]) / sum(math.exp(v) for v in uu)), u)
    report(f"C4 δ（正解 = {c}）", call(ex.softmax_ce_grad, u, c), exp, "softmax(u) を計算して、正解の番号の所だけ 1 を引く")

report("C5 学習したフィルタ", call(ex.learn_filter, [0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, -1, 0], 0.1, 500), (-1.0, 1.0),
       "毎回、出力を計算 → ∂E/∂a, ∂E/∂b を足し合わせて出す → a ← a − lr × ∂E/∂a（b も）")

report("C6 プーリングの例1", call(ex.maxpool_backward, [1, 5, 2, 8], [0.3, -0.7]), [0, 0.3, 0, -0.7],
       "2 個ずつ組にして、大きい方の場所にだけ δ を入れる")
report("C6 プーリングの例2", call(ex.maxpool_backward, [4, 1, 0, 3, 9, 2], [1, 2, 3]), [1, 0, 0, 2, 3, 0],
       "組の数は len(inp) // 2。i 番目の組は inp[2i] と inp[2i+1]")

print(f"\n{score[0]} / {score[1]} 正解")
if score[0] == score[1]:
    print("全問正解！ 勾配降下法・バックプロパゲーション・CNN の基本は、もう自分の手で書けている")

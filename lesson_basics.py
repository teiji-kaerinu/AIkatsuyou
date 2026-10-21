"""
lesson_basics.py ― 解説に出てくる数字を、自分の手で確かめるためのスクリプト

  1. 重み 1 個の勾配降下法（y = w·x で、x = 2 のとき 6 を出したい）
  2. 2 層ネットのバックプロパゲーションを、手計算と同じ順番で
  3. 畳み込みの勾配は「使った場所ぶんの合計」になる（1 次元の例）

ライブラリは何も使わない。数字を変えて何度も動かしてみるのがおすすめ。
"""

print("■ 1. 勾配降下法:  y = w·x,  x = 2,  正解 t = 6,  損失 E = (w·x − t)²")
x, t = 2.0, 6.0
for lr in [0.05, 0.3]:
    print(f"  学習率 {lr}")
    w = 1.0
    for step in range(6):
        y = w * x
        E = (y - t) ** 2
        grad = 2 * (y - t) * x      # dE/dw ＝ 坂の傾き
        print(f"    {step}回目  w = {w:7.3f}  予測 {y:7.3f}  損失 {E:8.3f}  傾き {grad:+8.3f}")
        w = w - lr * grad           # 傾きと逆向きに、学習率の分だけ動く

print("\n■ 2. バックプロパゲーション:  x →[w1]→ u1 →ReLU→ z1 →[w2]→ y,  損失 E = ½(y − t)²")
x, t, w1, w2, lr = 2.0, 3.0, 0.5, 1.5, 0.1


def forward(w1, w2):
    u1 = w1 * x
    z1 = max(0.0, u1)               # ReLU
    y = w2 * z1
    return u1, z1, y, 0.5 * (y - t) ** 2


u1, z1, y, E = forward(w1, w2)
print(f"  順伝播:  u1 = {u1}   z1 = {z1}   y = {y}   E = {E}")
dy = y - t                          # ∂E/∂y
dw2 = dy * z1                       # ∂E/∂w2 = ∂E/∂y × (w2 の入口の値 z1)
dz1 = dy * w2                       # ∂E/∂z1 = ∂E/∂y × w2   ← 1 つ前へ誤差を渡す
du1 = dz1 * (1.0 if u1 > 0 else 0.0)  # ReLU の傾き（正なら 1、負なら 0）
dw1 = du1 * x                       # ∂E/∂w1 = ∂E/∂u1 × (w1 の入口の値 x)
print(f"  逆伝播:  ∂E/∂y = {dy}   ∂E/∂w2 = {dw2}   ∂E/∂z1 = {dz1}   ∂E/∂u1 = {du1}   ∂E/∂w1 = {dw1}")

eps = 1e-6                          # 答え合わせ: 本当に少し動かして測る（数値微分）
num1 = (forward(w1 + eps, w2)[3] - forward(w1 - eps, w2)[3]) / (2 * eps)
num2 = (forward(w1, w2 + eps)[3] - forward(w1, w2 - eps)[3]) / (2 * eps)
print(f"  数値微分: ∂E/∂w1 = {num1:.6f}   ∂E/∂w2 = {num2:.6f}   （逆伝播と一致すれば OK）")

w1, w2 = w1 - lr * dw1, w2 - lr * dw2
u1, z1, y, E2 = forward(w1, w2)
print(f"  更新後:  w1 = {w1:.3f}   w2 = {w2:.3f}   y = {y:.3f}   損失 {E:.3f} → {E2:.4f}")

print("\n■ 3. 畳み込みの勾配:  入力 [1, 2, 3, 4] に、フィルタ [a, b] を 1 マスずつ滑らせる")
inp, a, b = [1.0, 2.0, 3.0, 4.0], 0.5, -1.0


def conv(a, b):
    return [a * inp[i] + b * inp[i + 1] for i in range(3)]


def conv_loss(a, b):                # 目標は全部 0、損失 E = ½Σ(出力)²
    return 0.5 * sum(o ** 2 for o in conv(a, b))


out = conv(a, b)
delta = out                         # ∂E/∂(出力 i) = 出力 i − 0
da = sum(delta[i] * inp[i] for i in range(3))      # a は 3 か所で使ったので 3 か所ぶん足す
db = sum(delta[i] * inp[i + 1] for i in range(3))
print(f"  出力 {out}   誤差 δ = {delta}")
print(f"  ∂E/∂a = {delta[0]}×{inp[0]} + {delta[1]}×{inp[1]} + {delta[2]}×{inp[2]} = {da}")
print(f"  ∂E/∂b = {delta[0]}×{inp[1]} + {delta[1]}×{inp[2]} + {delta[2]}×{inp[3]} = {db}")
print(f"  数値微分: ∂E/∂a = {(conv_loss(a + eps, b) - conv_loss(a - eps, b)) / (2 * eps):.6f}"
      f"   ∂E/∂b = {(conv_loss(a, b + eps) - conv_loss(a, b - eps)) / (2 * eps):.6f}")

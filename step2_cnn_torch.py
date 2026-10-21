"""
step2_cnn_torch.py ― step2 の CNN を PyTorch で書き直し、MNIST で動かす

step2_cnn.py（NumPy）で自作した部品は、PyTorch ではそれぞれ 1 行で済む。
    Conv2D → nn.Conv2d      BatchNorm → nn.BatchNorm2d      ReLU → nn.ReLU
    MaxPool2D → nn.MaxPool2d    Affine → nn.Linear    SoftmaxCrossEntropy → nn.CrossEntropyLoss
    全部の層に自分で書いた backward() → loss.backward() の 1 行（自動微分）

実行:
    python step2_cnn_torch.py                   # 卒研と同じ構成（samplev43.m 167行目）
    python step2_cnn_torch.py --model small     # step2_cnn.py と同じ小さい構成
    python step2_cnn_torch.py --epochs 5
初回は MNIST（手書き数字 学習 60000 枚 / テスト 10000 枚、約 11MB）を ./data にダウンロードする。
"""
import argparse
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def thesis_cnn():
    """samplev43.m 167〜187行目の layers をそのまま書き写したもの"""
    return nn.Sequential(
        # imageInputLayer([28 28 1]) にあたる行は不要（入力の形は最初のデータで決まる）
        nn.Conv2d(1, 17, 5, padding="same"), nn.BatchNorm2d(17), nn.ReLU(),   # 28x28x17
        nn.MaxPool2d(2, stride=1),                                             # 27x27x17
        nn.Conv2d(17, 17, 5, padding="same"), nn.BatchNorm2d(17), nn.ReLU(),  # 27x27x17
        nn.MaxPool2d(2, stride=1),                                             # 26x26x17
        nn.Conv2d(17, 25, 5, padding="same"), nn.BatchNorm2d(25), nn.ReLU(),  # 26x26x25
        nn.Flatten(),                                                          # 16900 次元に平らにする
        nn.Linear(25 * 26 * 26, 10),                                           # fullyConnectedLayer(10)
        # softmaxLayer + classificationLayer は、損失関数 nn.CrossEntropyLoss の中に入っている
    )


def small_cnn():
    """step2_cnn.py の build_cnn() と同じ構成"""
    return nn.Sequential(
        nn.Conv2d(1, 8, 5, padding="same"), nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2, stride=2),                                             # 14x14x8
        nn.Conv2d(8, 16, 5, padding="same"), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2, stride=2),                                             # 7x7x16
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 10),
    )


def salt_and_pepper(x, level):
    """imnoise(I, 'salt & pepper', level) 相当"""
    mask = torch.rand_like(x) < level
    return torch.where(mask, torch.randint_like(x, 0, 2), x)


def median_filter(x, k=3):
    """medfilt2(I, [k k]) 相当（端はゼロ埋め）。x: (N, 1, H, W)"""
    p = k // 2
    win = nn.functional.pad(x, (p, p, p, p)).unfold(2, k, 1).unfold(3, k, 1)  # (N, 1, H, W, k, k)
    return win.reshape(*win.shape[:4], -1).median(dim=-1).values


@torch.no_grad()  # テスト中は勾配を記録しない（速くなる）
def evaluate(model, loader, noise=0.0, median=False):
    model.eval()  # BatchNorm を推論モードに（学習中にためた平均・分散を使う）
    correct = 0
    for x, t in loader:
        if noise:
            x = salt_and_pepper(x, noise)
        if median:
            x = median_filter(x)
        correct += (model(x).argmax(dim=1) == t).sum().item()
    return correct / len(loader.dataset) * 100


def compare_with_numpy():
    """答え合わせ: 自作の NumPy 版（layers.py）と PyTorch の自動微分に、
    同じ重み・同じ入力を与えて、勾配が一致するか確かめる"""
    from dataset import make_digits
    from step2_cnn import build_cnn

    X, y = make_digits()
    xb, yb = X[::100, None], y[::100]                     # 10 枚
    np_net = build_cnn(0)
    t_net = small_cnn().double()
    np_layers = [l for l in np_net.layers if l.params]    # Conv2D, BatchNorm, Affine
    t_layers = [m for m in t_net if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))]
    with torch.no_grad():                                 # NumPy 版の重みを PyTorch 側にコピー
        for a, b in zip(np_layers, t_layers):
            w = a.params.get("W", a.params.get("gamma"))
            b.weight.copy_(torch.from_numpy(w.T if isinstance(b, nn.Linear) else w))  # Linear は (出力, 入力) の向き
            b.bias.copy_(torch.from_numpy(a.params.get("b", a.params.get("beta"))))

    np_net.loss(xb, yb)
    np_net.backward()                                     # 自作の逆伝播
    t_net.train()
    nn.functional.cross_entropy(t_net(torch.from_numpy(xb)), torch.from_numpy(yb).long()).backward()  # 自動微分

    print("■ 答え合わせ: 自作 NumPy 版の backward と PyTorch の自動微分で、重みの勾配を比べる")
    for a, b in zip(np_layers, t_layers):
        g_np = a.grads.get("W", a.grads.get("gamma"))
        g_t = b.weight.grad.numpy()
        g_t = g_t.T if isinstance(b, nn.Linear) else g_t
        print(f"  {type(a).__name__:9s} 勾配の大きさ {abs(g_np).max():.2e}   差の最大 {abs(g_np - g_t).max():.1e}")
    print("  → 差が 1e-12 程度なら、自作の逆伝播は PyTorch と同じ計算をしている\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["thesis", "small"], default="thesis")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(0)

    compare_with_numpy()

    tf = transforms.ToTensor()  # 0〜255 の画像 → 0〜1 の (1, 28, 28) テンソル
    train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=50, shuffle=True)  # MiniBatchSize 50, 'Shuffle','every-epoch'
    test_loader = DataLoader(test_set, batch_size=500)

    model = thesis_cnn() if args.model == "thesis" else small_cnn()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"■ 学習（{args.model} 構成, パラメータ {n_params:,} 個, 学習 {len(train_set)} 枚）")
    loss_fn = nn.CrossEntropyLoss()                                        # softmaxLayer + classificationLayer
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)       # trainingOptions('sgdm')

    for ep in range(1, args.epochs + 1):
        model.train()  # BatchNorm を学習モードに（ミニバッチの平均・分散を使う）
        t0, total = time.time(), 0.0
        for i, (x, t) in enumerate(train_loader, 1):
            loss = loss_fn(model(x), t)       # 1) 順伝播して損失を出す
            opt.zero_grad()                   #    前回の勾配を消す（PyTorch は勾配を足し込んでいく仕様）
            loss.backward()                   # 2) 逆伝播 ― NumPy 版で全層に書いた backward がこの 1 行
            for p in model.parameters():      #    'GradientThreshold', 1 : 重みごとに勾配の長さを 1 までに抑える
                nn.utils.clip_grad_norm_(p, 1.0)
            opt.step()                        # 3) 重みを更新
            total += loss.item()
            if i % 300 == 0:
                print(f"    {i * len(x):5d} 枚目まで  直近の損失 {total / i:.4f}", flush=True)
        print(f"  epoch {ep}  損失 {total / len(train_loader):.4f}  テスト正解率 {evaluate(model, test_loader):.2f}%"
              f"  （{time.time() - t0:.0f} 秒）", flush=True)

    print("\n■ ノイズに対する強さ（テスト 10000 枚）")
    print("  ノイズ率   フィルタなし   メディアンあり")
    for level in [0, 0.001, 0.02, 0.05, 0.1]:
        note = "  ← 卒研の NoiseLevel" if level == 0.001 else ""
        print(f"  {level:6.3f}     {evaluate(model, test_loader, level):6.2f}%"
              f"       {evaluate(model, test_loader, level, median=True):6.2f}%{note}", flush=True)

    torch.save(model.state_dict(), f"mnist_{args.model}.pt")  # 学習済みの重みを保存（次回は読み込むだけで使える）
    print(f"\n  学習済みの重みを mnist_{args.model}.pt に保存しました")


if __name__ == "__main__":
    main()

# 卒研 samplev43.m を Python で復習する

NumPy だけ（PyTorch などは使わない）で、卒研の MATLAB コードの中身を一から組み直したもの。
「ライブラリが中でやっていること」を自分の手で書いて、CNN とバックプロパゲーションを思い出すのが目的。

## 実行順

```bash
python dataset.py          # 合成の数字データを見る（卒研の dataset フォルダの代わり）
python step1_backprop.py   # BP を数式そのままで書く（patternnet の中身）
python step2_cnn.py        # CNN を部品から組む（convolution2dLayer などの中身）
python step3_ensemble.py   # 卒研の「CNN 複数 → 誤り学習 → 融合 BP」を縮小再現
python step2_cnn_torch.py  # step2 を PyTorch で書き直し、MNIST で学習（卒研と同じ構成）
```

step1〜3 は NumPy だけで動く。`layers.py` が部品集（全部の層の順伝播・逆伝播）。
`step2_cnn_torch.py` だけは PyTorch が必要（`python -m pip install torch torchvision`）。
初回に MNIST（約 11MB）を `data/` にダウンロードし、学習した重みを `mnist_thesis.pt` に保存する。
`--model small` で step2 と同じ小さい構成、`--epochs 5` でエポック数を変えられる。

## 練習問題（勾配降下法 → BP → CNN を自分の手で）

```bash
python lesson_basics.py      # 解説に出てくる数字を確かめる（数字を変えて何度も動かす）
python check.py              # exercises.py の答え合わせ
python check.py --answers    # どうしても分からない時だけ正解を表示
```

`exercises.py` の `None` の所を自分の答えに書き換えていく。全 11 問（採点項目 24 個）。

| レベル | 問題 | 内容 |
|---|---|---|
| A | A1〜A3 | 数字を変えて結果を予想する（学習率・初期値・ReLU が止まる場合） |
| B | B1〜B2 | 逆伝播を手で計算する（正解を変える、2 回目の更新） |
| C | C1〜C6 | 関数を書く（バイアス、tanh、ニューロン 2 個、softmax、フィルタの学習、プーリングの逆伝播） |

## samplev43.m は何をしているか（1 ループ分）

| 段 | 行 | 内容 |
|---|---|---|
| 1 | 46〜64 | 1000 枚からテスト 200 枚を抜き、salt & pepper ノイズ → 3×3 メディアンフィルタ |
| 2 | 127〜207 | CNN を 5 個。CNN ごとに残り 800 枚から学習 600 / 検証 200 を選び直して学習 |
| 3 | 86〜163 | 手作り特徴 59 次元（`getFeature59_28`）+ 浅いネット `patternnet(59)` を 3 個 |
| 4 | 280〜532 | 検証で「間違えた（Level1）」「当たったが確率 < 0.625（Level2）」等のサンプルを集め、それ専用の「誤り学習 CNN」（回帰・56×56 の 4 分割画像入力）を 12〜19 個、間違いを足しながら順に学習 |
| 5 | 576〜649 | 全モデルの出力（10 クラス確率）を縦に並べ、`patternnet(36)` で最終判定（誤り学習なし） |
| 6 | 679〜1004 | 上位 5 候補だけ抜き出して各種統計量を 2 次元に並べ、小さな CNN（融合 CNN）で「5 候補のどれが正解か」を判定（誤り学習あり） |
| 7 | 1003〜1125 | 精度比較と、誤り学習で直った/悪化した例の表示。これを `LOOPs = 12` 回 |

一言でいうと **「複数モデルのアンサンブル + 苦手サンプルの重点学習 + 出力をもう一段のネットで統合（スタッキング）」で、ノイズに強い数字認識を作る研究**。

## MATLAB → Python 対応表

| MATLAB | このフォルダ（NumPy） | PyTorch（次の一歩） |
|---|---|---|
| `convolution2dLayer(5,17,'Padding','same')` | `Conv2D(c_in, 17, 5, rng)` | `nn.Conv2d(c_in, 17, 5, padding='same')` |
| `batchNormalizationLayer` | `BatchNorm(c)` | `nn.BatchNorm2d(c)` |
| `reluLayer` / `tanhLayer` | `ReLU()` / `Tanh()` | `nn.ReLU()` / `nn.Tanh()` |
| `maxPooling2dLayer(2,'Stride',[1 1])` | `MaxPool2D(2, stride=1)` | `nn.MaxPool2d(2, stride=1)` |
| `fullyConnectedLayer(10)` | `Affine(n, 10, rng)` | `nn.Linear(n, 10)` |
| `softmaxLayer` + `classificationLayer` | `SoftmaxCrossEntropy` | `nn.CrossEntropyLoss()` |
| `regressionLayer` | ― | `nn.MSELoss()` |
| `trainingOptions('sgdm')` / `('adam')` | `SGDMomentum()` / `Adam()` | `torch.optim.SGD(..., momentum=0.9)` / `torch.optim.Adam` |
| `trainNetwork` | `train(net, X, y)` | for 文で自分で書く |
| `predict(net, X)` | `net.proba(X)` | `model.eval(); softmax(model(x))` |
| `patternnet(59)` | `Affine → Tanh → Affine` | `nn.Sequential(nn.Linear(59,59), nn.Tanh(), nn.Linear(59,10))` |
| `imnoise(I,'salt & pepper',d)` | `salt_and_pepper(X, d, rng)` | ― |
| `medfilt2(I,[3 3])` | `median_filter(X, 3)` | `scipy.ndimage.median_filter` |
| `mapminmax(F,0,1)` | `(F - lo) / (hi - lo)` | `sklearn.preprocessing.MinMaxScaler` |
| `datasample(A, n, 'Replace', false)` | `rng.permutation(A)[:n]` | ― |
| `setdiff(A, B)` | `np.setdiff1d(A, B)` | ― |
| `ind2vec(labels)` | `np.eye(10)[y]` | `F.one_hot(y, 10)` |
| `A(:,:,1,l)`（縦,横,ch,枚数） | `X[l, 0]`（枚数,ch,縦,横） | 同左 |
| 添字は 1 から | 添字は 0 から | 同左 |

## 読み返して気づいた点（改善ポイント）

1. **619 行目の「ソヘ」**: 打ち間違いと思われる。MATLAB の名前に全角文字は使えないので、このままだとエラーになるはず。
2. **`getFeature59_28` が入っていない**: 別ファイル。手作り特徴の中身はこのファイルからは分からない。
3. **融合 BP の学習データの「漏れ」**: 検証セットを CNN ごとに独立に選んでいるので、CNN k の検証セットは別の CNN の学習データと重なる。そこでは出力が当たりやすく、融合 BP は「本番より甘い出力」で学ぶことになる（398 行目の補正式はこれを意識した跡に見える）。定石は K 分割して「そのモデルが学習に使っていないデータでの出力」だけで融合器を学ぶ（out-of-fold スタッキング）。
4. **テストにだけメディアンフィルタ**: ノイズ 0.1% だと、フィルタが線の角を削る害の方が大きくなりうる（step2 で確認できる）。
5. **`mapminmax` をテスト画像込みで計算**（95 行目）: 厳密にはテストの情報が学習側に混ざる。学習データだけで min/max を決めるのが正しい。
6. **マジックナンバーが多い**（0.625, 0.81, 2.5, 3.6, 1.5 など）: 試行錯誤の跡。検証データだけで決めた値なら問題ないが、テスト結果を見て決めたなら過大評価になる。
7. **誤り学習の効果の切り分け**: 誤り学習ありの融合は、モデルの数そのものが増えている（卒研では 12〜19 個）。「苦手サンプルを重点学習した効果」なのか「単にモデルが増えた効果」なのかは、普通の CNN を同じ数だけ足した場合と比べないと言えない（対照実験）。

## 実行結果の例（step3、3 ループ平均、テストはノイズ 5% → メディアンフィルタ）

| 方法 | 正解率 |
|---|---|
| 従来法（CNN 単体の平均） | 78.0% |
| 浅いネット単体の平均（参考） | 53.4% |
| CNN 3 個の出力を平均しただけ | 82.7% |
| 融合 BP（誤り学習なし） | 85.2% |
| 融合 BP（誤り学習あり） | 87.3% |

卒研の狙いどおり「単体 < 平均 < 融合 < 誤り学習つき融合」の順になった。
ただしテストは 200 枚（1 枚 = 0.5%）で 3 ループだけなので、最後の 2% の差はばらつきの範囲でもある。

## 実行結果の例（step2_cnn_torch.py、MNIST、3 エポック、CPU）

最初に、自作 NumPy 版と PyTorch の自動微分の勾配を比べている。差は 1e-16 で、計算が完全に一致した。

| 構成 | パラメータ数 | 1 エポック | テスト正解率 |
|---|---|---|---|
| small（step2 と同じ） | 11,322 | 約 15 秒 | 98.76% |
| thesis（卒研と同じ） | 187,462 | 約 44 秒 | 99.04% |

thesis 構成でのノイズへの強さ（テスト 10000 枚）:

| ノイズ率 | フィルタなし | メディアンあり |
|---|---|---|
| 0（なし） | 99.04% | 98.27% |
| 0.001（卒研の値） | 99.04% | 98.27% |
| 0.05 | 97.84% | 97.99% |
| 0.10 | 95.01% | 97.80% |

ノイズが少ないとメディアンフィルタはかえって下げ、ノイズ 5% を超えると効いてくる。合成データ（step2）と同じ傾向。

## 次の一歩

1. Python を新しくする（手元は 3.9.6。3.9 は 2025 年 10 月にサポート終了済み）。
2. ~~step2 の CNN を PyTorch で書き直す~~ → `step2_cnn_torch.py`（済み）。次は自分で層の数やフィルタ数を変えて、精度と時間がどう変わるか試す。
3. 上の「改善ポイント 3」（out-of-fold スタッキング）を自分で実装してみる。
4. GitHub に「MATLAB の卒研を Python に移植して、問題点を見つけて直した」リポジトリとして置く → ポートフォリオになる。

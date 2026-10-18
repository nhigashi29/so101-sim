# 2カメラACTをColabで学習する

PPOの成功軌跡を教師データとして、LeRobotの公式ACTを学習する。
PPOの重みの読み替えではなく、画像方策への蒸留。
PPOの配置・解放・退避・帰還は手元で追加学習する。
その教師によるデータ収集とACTの本学習をColabで実行する。ACT学習にはGPUを使う。

## 実行

1. `notebooks/act_colab.ipynb`をColabの「ファイル → ノートブックをアップロード」で開く。
2. 「ランタイム → ランタイムのタイプを変更」でGPUを選ぶ。
3. セルを上から実行し、要求されたら`outputs/act_colab_bundle.zip`をアップロードする。
4. Google Driveを接続する。収集した軌跡・モデル・optimizerを自分のDriveに保存する。
5. 最後に未知の配置で評価し、`act_checkpoint.zip`をダウンロードする。

初期値は成功軌跡200エピソード、画像256×256、batch 8、30,000更新。
`EPISODES`、`TOTAL_STEPS`、`BATCH_SIZE`はノートブック内で変更できる。
GPUにデータと学習の接続を確認する小さな1更新のセルもある。
その小さなモデルは、本学習の初期値に使わない。

中断後は同じZIP・同じ実験名でセルを再実行する。
収集は成功済みのエピソードから、学習は最新の保存済み更新数から続ける。
カメラ設定や収集数を変えてデータを作り直す場合は、EXPERIMENT名を変更する。
データセットが変わったまま古い正規化・optimizerで再開することは拒否する。

## 入出力と同期

| 項目 | 内容 |
| --- | --- |
| `observation.images.wrist` | 手元RGB画像 |
| `observation.images.overhead` | 固定した斜め上からの俯瞰RGB画像 |
| `observation.state` | 6関節の現在角度、rad |
| `action` | 6モーターの位置目標、rad |
| 制御周期 | 25 Hz |
| 予測チャンク | 25ステップ |
| 実行するステップ | 毎回1ステップ、次の画像で再計画 |

2画像と関節角度は、同じ物理状態・操作前の時刻に記録する。
PPOのTCP速度4次元を、モーターに渡す直前の6関節目標として記録する。
最終step中に離して指令が凍結される場合も、補間前の目標を保存する。

ACTは、PPOが使うキューブ座標・目標座標・工程番号・接触判定を入力に含めない。
目標位置は画像内の緑のマーカーから判断する。
位置変更後も、画像と自分の関節角度から開き直して追いかける。
PPOは教師データの収集時に使い、ACTの評価時には使わない。

## データと評価

教師の元モデルは`outputs/pick_place_ws1`（5,109,840ステップ）。
[帰還までの追加学習](return-home.md)と評価を手元で行い、選んだ方策をバンドルに入れる。
作業範囲`workspace_scale=1.0`と、配置・解放・退避・帰還の成功条件も引き継ぐ。
帰還未学習の旧モデルを入れた場合、ノートブックは収集前に停止する。
初期設定は`SCENARIOS = ["standard"]`で、掴み直しあり・途中移動なし・物理ランダム化なし・外力なし。
3条件を混ぜる場合は`SCENARIOS = ["standard", "recovery", "strong"]`に変更する。
教師のハッシュをDriveの実験名に含め、別方策のデータやチェックポイントとの混在を避ける。
成功した試行の全軌跡を保存し、掴み直す前の失敗と回復動作も残す。
失敗した試行の種・理由・物理パラメーターは`attempts.jsonl`に記録する。
試行上限までに成功データが揃わなければ停止し、`--resume`で続けられる。

予測できない位置変更・外力の開始をまたぐ未来ラベルはマスクする。
移動前の画像から、移動後のランダムな場所への操作を予測させない。
ただし移動後のフレームからは、新しい場所への回復動作を学ぶ。
同じエピソードのフレームを訓練と検証に分散させず、シナリオで層別化して分割する。
画像はImageNetのmean/std、関節と行動は訓練エピソードだけのmean/stdを使う。

検証lossで保存候補を選んだ後、別シード90000系列でシミュレーターを閉ループ評価する。
lossの低下だけで、配置や掴み直しの成功を判断しない。

データはエピソードごとのNPZ。Driveには圧縮して保存し、学習時はColabのローカルディスクに一度だけ展開してメモリマップで読む。
ランダムな1フレームの取得で軌跡全体を何度も解凍しない。推論は`src/act.py`の前処理を使用する。
モデル本体はLeRobotの`ACTPolicy.save_pretrained`で保存し、正規化と入出力の定義を同梱する。
チェックポイントの書き込み完了後に`latest.json`を更新するため、中断時は前の完了世代から再開できる。

## 実行環境

Python 3.12、LeRobot 0.6.1、PyTorch 2.8.0、torchvision 0.23.0、MuJoCo 3.16.0、SB3 2.9.0。
ColabではCUDA 12.6のPyTorchを使う。
本プロジェクトとLeRobotのNumPy/Pillowの制約が異なるため、
`requirements/act-colab.txt`を使った独立環境を作る。
プロジェクト全体をその環境へ`pip install -e .`する手順にはしない。

## ローカルのviewerで学習済みACTを見る

`act_checkpoint.zip`を`outputs/`に展開すると、`outputs/act_checkpoint/`ができる。
ACT用の独立環境を初めて作る場合:

```bash
uv venv .venv-act --python 3.12
uv pip install --python .venv-act/bin/python -r requirements/act-colab.txt
```

```bash
.venv-act/bin/python src/view.py --act evaluate \
  --run outputs/act_checkpoint --scenario standard --episodes 10 --watch
```

通常条件は`--scenario standard`、強い条件は`--scenario strong`。
入力カメラは固定なので、viewerの視点を回してもACTに渡す画像は変わらない。

## 教師を更新してバンドルを作り直す

```bash
uv run python src/act.py bundle \
  --run outputs/pick_place_return_home/selected \
  --output outputs/act_colab_bundle.zip
```

バンドルにはスクリプト・モデルのアセット・教師の3ファイルを含める。
`.venv`、他の実験、学習履歴全体は含めない。
各ファイルのSHA-256を記録し、Colabで展開したときに確認する。
ノートブックを再生成する開発用コマンドは`uv run python src/make_act_notebook.py`。

## 実機への移行

まずMuJoCo内で画像方策を評価する構成。
実機では、手元・俯瞰カメラの取り付け位置、内部パラメーター、画像の向きを合わせる。
関節のrad/degree/正規化値・ゼロ点・グリッパ変換も揃え、実画像のデータで追加学習する。
LeRobot標準CLIへ接続する場合も、この入出力変換と正規化を適用する。

## 確認済みの範囲

固定した依存環境で、2カメラの成功軌跡6本の収集、小さなACTの学習・保存・再開、
保存モデルを使ったMuJoCo評価の接続を確認した。
バンドルの全ファイルのチェックサムと、展開先での2カメラ描画も確認済み。
これは接続確認であり、ACTのタスク成功率を保証する結果ではない。
帰還まで追加学習した教師でも、Colabと同じ依存環境で成功軌跡2本の収集と小さなACTの1更新を確認した。
本収集とACTの本学習はColabで行う。
Colab上のGPU本学習は、このノートブックから実行する必要がある。

## 参照

- [LeRobot ACT・複数カメラ・Colab](https://huggingface.co/docs/lerobot/act)
- [固定したACT設定の実装（v0.6.1）](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py)
- [PyTorch 2.8.0 / torchvision 0.23.0 / CUDA 12.6](https://pytorch.org/get-started/previous-versions/)

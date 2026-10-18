# SO-101 Simulation

SO-101の公式MuJoCoモデルを使う最小シミュレーション。
`uv`でPythonと依存関係を管理する。Python 3.12、macOS arm64で動作確認済み。

## 起動

```bash
git clone https://github.com/nhigashi29/so101-sim.git
cd so101-sim
uv sync --locked
uv run python src/view.py
```

マウスで視点変更・拡大縮小ができる。右のControlパネルで各関節の目標角度を変更できる。
ウィンドウを閉じると終了する。

標準で手首カメラとマウント付きのモデルを使用する。
手首カメラの視点で表示する場合:

```bash
uv run python src/view.py --camera wrist
```

旋回とグリッパ開閉のデモ:

```bash
uv run python src/view.py --demo
```

macOSでは`src/view.py`が`mjpython`を使用し、uvのPythonに必要な共有ライブラリの
探索パスを設定する。単に`uv run mjpython src/main.py`とすると環境によって
`libpython3.12.dylib`が見つからないため、このランチャーを使う。

## GUIなしで物理計算・画像保存

```bash
uv run python src/main.py --headless --demo --seconds 5 --snapshot outputs/so101.png
```

6関節・6アクチュエータの読み込み、有限な関節状態、MuJoCo警告の有無を確認し、
640×480のRGB画像を保存する。画像保存にはOpenGL環境が必要。
`--snapshot`を省略すればレンダリングせず物理計算のみ実行できる。

手首カメラから640×480のRGB画像を保存する場合:

```bash
uv run python src/main.py --headless --camera wrist --snapshot outputs/wrist.png
```

モデル内のカメラ名は`wrist`。手首に追従し、レンズの前方（カメラ本体の+Z）を撮影する。
垂直画角は仮設定の60度で、実機の内部パラメータによる校正は未実施。
`--camera free`（既定）で全体を見渡す視点にできる。

### 手首カメラの設定

`assets/so101/wrist_camera.xml` を編集し、アプリを再起動すると反映される。
ロボットのXMLから`include`で読み込む。

| 属性 | 用途 | 現在の設定 |
| --- | --- | --- |
| `pos` | カメラ本体の座標系での撮影位置（m） | `0 0 0.0264` |
| `xyaxes` | カメラ座標系のX軸・Y軸の向き | `0 1 0 1 0 0` |
| `fovy` | 垂直画角（度）。大きくすると広角 | `60` |
| `mode` | 親ボディへの固定・追従方式 | `fixed`（手首と一緒に動く） |

向きは`xyaxes`の代わりに`quat`（w, x, y, z）や`euler`（このモデルではrad）でも指定できる。
向きの表現はどれか1つを使う。
実機の内部パラメーターを合わせる場合は、`sensorsize`と`resolution`、
`focalpixel`（fx, fy）、`principalpixel`（画像中心からの主点オフセット）なども指定できる。
`sensorsize`を指定すると`fovy`ではなく焦点距離とセンサーサイズから画角が決まる。
現在の保存画像サイズ640×480は`src/main.py`の`mujoco.Renderer`で指定している。
XMLの`resolution`だけでは保存サイズは変わらない。

詳細: [MuJoCo公式カメラ仕様](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)。

## ファイル

- `src/main.py`: モデル読み込み、位置制御、物理計算、表示、画像保存。
- `src/view.py`: macOS対応のGUIランチャー。
- `src/task_logic.py`: タスクの段階と成功・失敗条件。
- `src/pick_place.py`: Gymnasium環境、接触形状、手先制御、観測・報酬。
- `src/task.py`: PPO学習・評価、固定動作デモ。
- `assets/so101/`: 公式のMJCF・URDF・メッシュ・ライセンス・取得元。
- `uv.lock`: 依存関係の固定。

公式モデル: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
取得したコミットは`assets/so101/SOURCE.md`を参照。
関節・グリッパ指令はrad。LeRobotのグリッパ0–100表現との変換は未実装。

## Lint・フォーマット・型チェック

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

自動修正とフォーマット:

```bash
uv run ruff check --fix .
uv run ruff format .
```

RuffはPython 3.12・行幅88文字を基準とし、基本エラー・未使用import・
import順序・バグを招きやすい書き方・Python構文の更新をチェックする。
配布元のモデルを含む`assets`、取得用の`third_party`、生成物の`outputs`は対象外。

開発依存の非公式`mujoco-stubs`でMuJoCo 3.16系の型情報を補完する。
Pyrightは`pyproject.toml`の設定で`.venv`を参照する。
エディタに古い診断が残る場合は、PyrightのLanguage Serverを再起動する。
MuJoCoを更新するときは、stubも対応するバージョンに更新する。

## Pick and place の強化学習（v5）

このワークスペースの固定評価モデルは `outputs/final/pick_place_v5_2003440/`。
未使用シードで標準配置20/20、ランダム配置45/50。
[評価結果と再現コマンド](docs/pick-place-validation.md) を参照。モデルはGit管理外の生成物。

単一のPPO方策が、手先のxyz速度と指の開閉速度を選ぶ。
机上の3cm・25gの物体を掴み、持ち上げ、目標へ運び、机に下ろし、離し、物体の静止を確認して終了する。
空中で落とす・解放後に押して位置を直す動作は失敗になる。
状態観測を使用し、教師データや固定動作による事前学習は使用しない。

学習は `release → place → transport → lift → grasp → pick → random` の順に初期状態を難しくする。
まず机上での解放、次に下ろして配置、さらに運搬、持ち上げ、把持へ開始範囲を広げる。
机に支持された解放を確認すると方策の操作を終え、最後のモーター指令を保持して0.4秒の物体静止を検査する。
全段階の成功条件は共通。`lift`も持ち上げるだけでなく、運搬・配置まで完了して成功になる。`place`の成功は全タスクの成功とは区別する。
進級には別シードでの決定的・サンプリング評価と直前の段階の維持を要求する。
詳細な契約・報酬・観測は [設計書](docs/pick-place-design.md) を参照。

```bash
# 新規学習。8環境で1つの方策を学習する
uv run python src/task.py train --steps 2000000 --output outputs/pick_place_v5

# 続きを学習する。stepsは追加するステップ数
uv run python src/task.py train --resume outputs/pick_place_v5 --steps 500000 --output outputs/pick_place_v5

# 最初から全タスクだけを学習する場合
uv run python src/task.py train --stage random --fixed-stage --steps 2000000 --output outputs/pick_place_full
```

v5は観測69次元・行動4次元。以前のモデルは互換性がないため、新規学習が必要。
旧モデルの保存先はそのまま保持する。Ctrl+Cで方策・正規化統計・設定を保存する。
50,000ステップごとに再開可能な `checkpoints/step_*/`、初期段階ごとの最良方策を `best/<stage>/` に保存する。
進級評価は `validation.jsonl`、学習の各試行は `monitor/` に記録する。

学習した方策を通常のMuJoCo viewerで見る:

```bash
uv run python src/view.py --pick-place evaluate --run outputs/pick_place_v5 --stage random --episodes 5 --watch

# 最初の解放段階を学習中なら、その初期状態から見る
uv run python src/view.py --pick-place evaluate --run outputs/pick_place_v5 --stage release --episodes 5 --watch
```

GUIなしで評価する。進級判定とは別のシード20000以降を使い、推論方式別に記録する:

```bash
uv run python src/task.py evaluate --run outputs/pick_place_v5 --stage random --episodes 50
uv run python src/task.py evaluate --run outputs/pick_place_v5 --stage random --episodes 50 --stochastic
```

モーターの発熱への代理ペナルティとして、全6軸の上限で正規化した出力トルク二乗の時間積分を小さく引く。
実際の温度を計算する熱モデルは未実装。

物理検証用の固定動作を見る場合（学習済み方策とは別）:

```bash
uv run python src/view.py --pick-place demo --stage pick
uv run python src/task.py demo --headless --stage random --episodes 5
```

回帰テスト:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -q
```

## 開始位置を徐々にランダム化する

物体の位置と手先の初期位置を6段階で広げる追加学習:

```bash
uv run python src/task.py train \
  --resume outputs/final/pick_place_v5_2003440 \
  --stage random --expand-starts --steps 1000000 --envs 8 \
  --eval-every 50000 --eval-episodes 25 --output outputs/pick_place_expanding
```

現在・以前の範囲の成功率を2回連続で確認してから次の範囲へ進む。
再開時は`--resume outputs/pick_place_expanding`で範囲と昇格状況も引き継ぐ。
範囲・検証条件・viewerのコマンドは[開始位置カリキュラム](docs/start-position-curriculum.md)を参照。

## キューブへの外乱とdomain randomization

横押しと、質量・摩擦・関節粘性・モーターゲインのばらつきを加えて追加学習する:

```bash
uv run python src/task.py train \
  --resume outputs/pick_place_expanding --domain-randomization --disturbances \
  --steps 200000 --envs 8 --learning-rate 0.00005 \
  --eval-every 50000 --eval-episodes 20 --output outputs/pick_place_robust
```

```bash
uv run python src/view.py --pick-place evaluate \
  --run outputs/pick_place_robust --episodes 10 --seed 41000 --watch
```

評価時は保存された設定が自動適用される。
`--no-domain-randomization --no-disturbances`で通常条件に切り替えられる。
範囲と再開方法は[外乱・物理ランダム化](docs/domain-randomization.md)を参照。

## 途中で位置を変え、掴み直しを学習する

`--recovery --cube-relocations`で、持ち上げたキューブを別の机上位置へ移し、
再接近・掴み直し・再配置まで練習する。強い外乱は
`--disturbance-strength 3 --domain-scale 3`で指定できる。

```bash
uv run python src/view.py --pick-place evaluate \
  --run outputs/pick_place_recovery/selected --episodes 10 --seed 42000 --watch
```

設定と学習・再開の手順は[掴み直しの学習](docs/regrasp-training.md)を参照。

## 手元＋俯瞰の2カメラACTをColabで学習する

先に[配置・解放・退避・帰還を手元で追加学習](docs/return-home.md)し、評価した方策をバンドルへ入れる。
[Colabノートブック](notebooks/act_colab.ipynb)をColabにアップロードし、GPUを選んで上から実行する。
要求されたら`outputs/act_colab_bundle.zip`をアップロードする。
教師PPOによるデータ収集、公式ACTの学習、中断後の再開、未知配置での評価、モデルのダウンロードまで含む。

バンドルを作り直す場合:

```bash
uv run python src/act.py bundle \
  --run outputs/pick_place_return_home/selected \
  --output outputs/act_colab_bundle.zip
```

入出力・データ形式・学習後のwatch方法は[ACTとColab](docs/act-colab.md)を参照。

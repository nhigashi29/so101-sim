# SO-101 Simulation

SO-101の公式MuJoCoモデルを使う最小シミュレーション。
`uv`でPythonと依存関係を管理する。Python 3.12、macOS arm64で動作確認済み。

## 起動

```bash
git clone https://github.com/nhigashi29/so101-sim.git
cd so101-sim
uv sync --locked
uv run python view.py
```

マウスで視点変更・拡大縮小ができる。右のControlパネルで各関節の目標角度を変更できる。
ウィンドウを閉じると終了する。

旋回とグリッパ開閉のデモ:

```bash
uv run python view.py --demo
```

macOSでは`view.py`が`mjpython`を使用し、uvのPythonに必要な共有ライブラリの
探索パスを設定する。単に`uv run mjpython main.py`とすると環境によって
`libpython3.12.dylib`が見つからないため、このランチャーを使う。

## GUIなしで物理計算・画像保存

```bash
uv run python main.py --headless --demo --seconds 5 --snapshot outputs/so101.png
```

6関節・6アクチュエータの読み込み、有限な関節状態、MuJoCo警告の有無を確認し、
640×480のRGB画像を保存する。画像保存にはOpenGL環境が必要。
`--snapshot`を省略すればレンダリングせず物理計算のみ実行できる。

## ファイル

- `main.py`: モデル読み込み、位置制御、物理計算、表示、画像保存。
- `view.py`: macOS対応のGUIランチャー。
- `assets/so101/`: 公式のMJCF・URDF・メッシュ・ライセンス・取得元。
- `uv.lock`: 依存関係の固定。

公式モデル: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
取得したコミットは`assets/so101/SOURCE.md`を参照。
関節・グリッパ指令はrad。LeRobotのグリッパ0–100表現との変換は未実装。

## 次に進む段階

1. キューブと目標位置を配置し、Gymnasiumの`reset` / `step`を実装する。
2. `state`（関節・物体・目標の状態）と`vision`（RGB・関節状態）を分離する。
3. 特権状態のPPO教師を学習する。
4. 教師の軌跡と画像を収集し、ACT / Flow Matchingを学習する。

現在のサンプルは表示と動作確認まで。報酬・RL学習・蒸留は未実装。

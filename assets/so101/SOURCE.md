# モデルの出典

- Upstream: https://github.com/TheRobotStudio/SO-ARM100
- Commit: `a758567c3978dfeefe282ede0500085a48fe8f78`
- Directory: `Simulation/SO101`
- License: 同梱の `LICENSE` を参照（Apache-2.0）。

モデルとメッシュは上記コミットから取得しており、実行時の追加ダウンロードは不要。
ローカル変更として、`scene.xml` は `so101_new_calib_camera.xml`
（関節可動域の中央をゼロとする校正、手首カメラとマウント付き）を使用する。
同モデルには画像取得用のMuJoCoカメラ`wrist`を追加している。
カメラ設定は`wrist_camera.xml`に分離し、`include`で読み込む（垂直画角60度の仮設定）。
MuJoCoのグリッパ指令はradであり、LeRobotの0–100表現との変換は未実装。

`pick_place.xml`はローカルで追加したタスクシーンで、天板・キューブ・目標を
MuJoCoの基本形状で定義している。追加の外部アセットは使用しない。
`pick_place.py`では、タスク用にグリッパの衝突メッシュを箱の接触形状に置換する。
接触面・摩擦は学習環境用の近似であり、実機で校正された値ではない。

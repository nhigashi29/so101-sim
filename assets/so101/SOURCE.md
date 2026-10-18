# モデルの出典

- Upstream: https://github.com/TheRobotStudio/SO-ARM100
- Commit: `a758567c3978dfeefe282ede0500085a48fe8f78`
- Directory: `Simulation/SO101`
- License: 同梱の `LICENSE` を参照（Apache-2.0）。

モデルとメッシュをそのままコピーしているため、実行時の追加ダウンロードは不要。
`scene.xml` は `so101_new_calib.xml`（関節可動域の中央をゼロとする校正）を使用する。
MuJoCoのグリッパ指令はradであり、LeRobotの0–100表現との変換は未実装。

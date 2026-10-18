# Pick and place v5 の検証結果

タスク契約・報酬・学習順の詳細は[設計書](pick-place-design.md)を参照。

固定モデル: `outputs/final/pick_place_v5_2003440/`。
モデル選択は進級検証用シード10000〜10024で行い、最終評価のシード30000以降は選択に使っていない。
選択した方策は2,003,440ステップ時点。単一PPO・状態観測・教師動作なし。

## 最終評価

初期状態を易しくする補助は無効。pickは標準配置、randomは物体XY・目標XY・物体の向きを変えた全工程。
正規化統計は固定。推論方式を分けて記録した。

| 初期条件 | 推論 | 成功 | 終了理由 |
| --- | --- | --- | --- |
| pick | 決定的 | 20/20 (100%) | {'success': 20} |
| random | 決定的 | 45/50 (90%) | {'success': 45, 'released_outside_goal': 4, 'timeout': 1} |
| random | サンプリング | 43/50 (86%) | {'success': 43, 'released_outside_goal': 6, 'timeout': 1} |

randomの残りの失敗は目標外での解放と時間切れ。全試行での成功には至っていない。

## 接触と静止の監査

成功した全試行で、次を実際の計測値と照合した。

- 物体の最大高さ65mm以上。
- 解放確定時に机支持があり、目標距離22mm未満。
- 終了時のロボット接触・把持なし、物体速度20mm/s未満、角速度0.5rad/s未満。
- 解放後は次の方策行動を要求せず、最後の指令で400ms連続静止を確認。

最後の項は回帰テストでも、静止確認中の全アクチュエータ指令が同じであることを検証した。
速度や位置を直接ゼロにする処理や、物体を接着する拘束は使っていない。
各試行の数値は評価JSON、範囲・平均は同じ保存先の `audit.json` を参照。

## 実装の確認

22件のテスト、Ruff lint・format、Pyrightを通過。
物理検証、終了条件、学習・検証の開始分布、複数環境への段階割り当て、
再開時のoptimizer設定、中断保存を含む。
固定動作は物理検証専用で、学習・推論へ教師動作を与えていない。

開発途中のv4で報告した配置20/20は、指が開き切る前の落下を見逃したため無効。
ここに記載した結果はその成績を使わず、v5の支持・解放・静止条件で評価し直したもの。

## 再現

```bash
uv run python src/view.py --pick-place evaluate --run outputs/final/pick_place_v5_2003440 --stage random --seed 30000 --episodes 5 --watch
uv run python src/task.py evaluate --run outputs/final/pick_place_v5_2003440 --stage random --seed 30000 --episodes 50
```

続きの学習は `outputs/pick_place_v5/` を使用する。固定評価用モデルは上書きしない。

```bash
uv run python src/task.py train --resume outputs/pick_place_v5 --steps 500000 --eval-episodes 25 --output outputs/pick_place_v5
```

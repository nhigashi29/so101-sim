# 強い外乱と、位置変更後の掴み直し

`--recovery`で、把持失敗を再挑戦可能にしたv6タスクを使える。
既存PPOの69次元観測・4次元行動を引き継いで学習する。
成功条件と、正しく離した後に方策を止める方式は維持する。

## 外乱とdomain randomizationの強さ

`--disturbance-strength 3 --domain-scale 3`の場合:

| 対象 | 範囲 |
| --- | --- |
| キューブの質量 | 10〜40 g（基準25 gの0.4〜1.6倍） |
| キューブの慣性 | 質量と同じ倍率 |
| 机・キューブ・左右パッドの摩擦 | 各形状独立に基準の0.4〜1.6倍 |
| 関節の粘性 | 各関節独立に基準の0.55〜1.45倍 |
| モーターゲイン | 各モーター独立に基準の0.7〜1.3倍 |
| 机上の横押し | 重力の1.3〜1.9倍相当、0.06〜0.10秒 |
| 運搬中の横押し | 2.4〜4.5 m/s²相当、0.08〜0.12秒 |

「外乱強度3」は空中の押しを3倍、机上では摩擦に打ち勝つための基準1gからの増分を3倍にする。
実際の力はキューブ質量で変わる。通常の強さ1のモデルも引き続き読み込める。
質量・摩擦等はresetごとに基準値から設定し、累積させない。

## 動作途中の位置変更

`--cube-relocations`を有効にすると、物理把持して持ち上げ、運搬段階に入った時点で
キューブを別の机上位置へ移す。これは短い力を加える外乱とは別の、明示的な位置変更試験。

- X 20〜24 cm、Y −2〜4 cm、水平半径24.5 cm以内から移動先を選ぶ。
- 元の位置から2 cm以上、目標から3.5 cm以上離れた位置にする。
- ロボットとの重なりを避け、キューブの速度と接触計算の初期推定をクリアする。
- ロボットの関節位置・モーター指令・閉じたグリッパ・経過時間は変えない。
- タスクは再接近へ戻り、方策が自分で開いて追いかけ、掴み直す。

既定は1回。`--relocation-count 2`または`3`なら、掴み直して持ち上げるたびに再度移動させる。
成功したfull-task試行は、指定回数の位置変更を経ている必要がある。
最終結果には移動時刻・移動前後の座標・把持回数を記録する。

## 失敗からの回復

v5では空中で離す・目標外で離す・把持を失うと即終了していた。
v6ではこの3種類を検出すると再接近へ戻し、接触判定用のタイマーをクリアする。
片側だけが接触して両指の把持が成立していない状態が0.12秒続いた場合も再試行する。
落下中の物体を目標上で静止させるだけでは成功にならない。
改めて把持・持ち上げ・運搬・支持されたリリースを行う必要がある。

期限は合計24秒で、失敗しても時計をリセットしない。
机から落ちた場合や操作可能な領域を外れた場合は終了する。
正しいリリース後は従来どおり操作を止め、0.4秒の静止を確認する。
その間は外乱・位置変更を加えず、ロボットの再接触は失敗とする。

追加の把持・自然な失敗からの回復回数そのものには正の報酬を与えない。
自然に落としたときはpotentialの後退もそのまま計算する。
明示的な外部の位置変更だけは、位置変更による不可避のpotentialの後退を補正する。
位置変更は既定1回、設定可能な上限も3回であり、この補正を無制限に繰り返して稼げない。
自分で落とした場合には補正せず、外部の位置変更回数も増やさない。
終端の成功報酬は最終的な配置を満たしたときだけ。
再接近では位置合わせとグリッパの開き具合もpotentialに含める。
補正量は`relocation_compensation`と`episode_relocation_compensation`に記録する。
この報酬リビジョンは`recovery-external-relocation-compensation-v2`。

## 学習と検証

学習中は約25%のresetで通常の物理条件を復習する。
外乱が有効なfull-task開始の30%は、グリッパが部分的に閉じた状態から始める。
通常条件・強い外乱と位置変更の組合せ・位置変更だけの条件を分けて検証する。
シード10000からの検証は、モデル選択・昇格に使用する。
最終評価は別のシード42000から。

`regrasp_count`は初回把持を除いた把持回数、`recovered_successes`は
掴み直した上で最後まで配置できた試行数。タイムアウトも含め、全試行を集計する。
新しいタスクのバージョンは`recoverable-place-v6`。
観測・行動の形状は同じなのでv5からの追加学習が可能だが、旧タスクの成績とは区別する。

```bash
uv run python src/task.py train \
  --resume outputs/pick_place_robust \
  --recovery --cube-relocations --disturbance-strength 3 --domain-scale 3 \
  --exploration-std 0.35 --entropy-coefficient 0.003 \
  --steps 500000 --envs 8 --learning-rate 0.0001 \
  --eval-every 100000 --eval-episodes 25 \
  --output outputs/pick_place_recovery
```

`--exploration-std`は学習開始時だけ探索幅を変える指定。通常の継続学習では省略する。
報酬・外乱・位置変更・回復設定はconfig.jsonに保存し、再開時に復元する。

```bash
uv run python src/view.py --pick-place evaluate \
  --run outputs/pick_place_recovery/selected --episodes 10 --seed 42000 --watch
```

viewerのログに回復の理由と掴み直しの回数を表示する。
位置変更だけを見る場合は`--no-domain-randomization --no-disturbances`を追加する。
通常条件を厳格なv5の終了規則で比較するには、さらに`--no-cube-relocations --no-recovery`を付ける。


継続学習（検証で選んだ方策から、別フォルダーへ保存）:

```bash
uv run python src/task.py train \
  --resume outputs/pick_place_recovery/selected \
  --steps 1000000 --envs 8 --learning-rate 0.0001 \
  --eval-every 100000 --eval-episodes 25 \
  --output outputs/pick_place_recovery_next
```

掴み直しだけを重点的に練習する場合は、学習にも
`--no-domain-randomization --no-disturbances`を指定できる。
この場合も明示的な位置変更と掴み直しは継続する。


## 掴み直しを先に学ぶカリキュラム

初めから広い配置・強い物理ランダム化・横押し・位置変更を全部加えた初回の学習では、
未知50配置での成功が強い条件2/50、位置変更だけ3/50にとどまった。
そのため、外部の位置変更のpotential補正を導入し、最初の配置を固定したpickで
位置変更後の掴み直しを先に練習するようにした。
この段階でも、ロボットの行動はPPOだけが決める。

```bash
uv run python src/task.py train \
  --resume outputs/pick_place_recovery/selected \
  --stage pick --fixed-stage --start-level 0 --max-start-level 0 \
  --no-domain-randomization --no-disturbances \
  --steps 300000 --envs 8 --learning-rate 0.0001 \
  --eval-every 100000 --eval-episodes 10 \
  --output outputs/pick_place_regrasp_warmup
```

その検証で選んだ方策を、再び広い開始範囲・強さ3の物理条件に戻して学習する:

```bash
uv run python src/task.py train \
  --resume outputs/pick_place_regrasp_warmup/best/pick_robust_recovery \
  --stage random --start-level 5 --max-start-level 5 \
  --domain-randomization --disturbances --recovery --cube-relocations \
  --disturbance-strength 3 --domain-scale 3 \
  --steps 300000 --envs 8 --learning-rate 0.0001 \
  --eval-every 100000 --eval-episodes 10 \
  --output outputs/pick_place_recovery_v2
```

この段階の最終評価には、初回の確認に使用した42000系列とは別の43000系列を使う。

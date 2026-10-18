"""2カメラACTのColab用バンドル・教師データ収集・学習・評価。"""

import argparse
from pathlib import Path

from act_data import ROOT, CameraPair, bundle, collect
from pick_place import PickPlaceEnv
from task import positive, positive_float


def train(args):
    from act_policy import train as run

    run(args)


def evaluate(args):
    from act_policy import evaluate as run

    run(args)


def preview(args):
    from PIL import Image

    env = PickPlaceEnv(stage="pick")
    pair = CameraPair(env, args.image_size)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, pixels in pair.capture().items():
            Image.fromarray(pixels).save(args.output / f"{name}.png")
    finally:
        pair.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("bundle")
    packing.add_argument("--run", type=Path, required=True)
    packing.add_argument(
        "--output", type=Path, default=ROOT / "outputs/act_colab_bundle.zip"
    )
    packing.set_defaults(func=bundle)
    capture = commands.add_parser("collect")
    capture.add_argument("--run", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--episodes", type=positive, default=200)
    capture.add_argument("--max-attempts", type=positive, default=None)
    capture.add_argument("--seed", type=int, default=60000)
    capture.add_argument("--stage", choices=("pick", "random"), default=None)
    capture.add_argument("--image-size", type=positive, default=256)
    capture.add_argument(
        "--scenarios",
        nargs="+",
        choices=("standard", "recovery", "strong"),
        default=["standard", "recovery", "strong"],
    )
    capture.add_argument("--resume", action="store_true")
    capture.set_defaults(func=collect)
    fitting = commands.add_parser("train")
    fitting.add_argument("--dataset", type=Path, required=True)
    fitting.add_argument("--output", type=Path, required=True)
    fitting.add_argument("--resume", type=Path)
    fitting.add_argument("--steps", type=positive, default=30000, help="追加の更新数")
    fitting.add_argument("--batch-size", type=positive, default=8)
    fitting.add_argument("--chunk-size", type=positive, default=25)
    fitting.add_argument("--save-every", type=positive, default=5000)
    fitting.add_argument("--validation-batches", type=positive, default=32)
    fitting.add_argument("--learning-rate", type=positive_float, default=1e-5)
    fitting.add_argument("--workers", type=int, default=2)
    fitting.add_argument("--seed", type=int, default=7)
    fitting.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    fitting.add_argument(
        "--tiny", action="store_true", help="接続確認用の小さな構成。実学習には使わない"
    )
    fitting.set_defaults(func=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument(
        "--run",
        type=Path,
        required=True,
        help="policy/・normalization.jsonを含むACTチェックポイント",
    )
    evaluation.add_argument("--episodes", type=positive, default=20)
    evaluation.add_argument("--seed", type=int, default=90000)
    evaluation.add_argument(
        "--scenario", choices=("standard", "recovery", "strong"), default="recovery"
    )
    evaluation.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    evaluation.add_argument(
        "--action-steps",
        type=positive,
        default=1,
        help="1回の推論で実行する動作数。大きいほど速いが、外乱への反応が遅れる",
    )
    evaluation.add_argument("--watch", action="store_true")
    evaluation.set_defaults(func=evaluate)
    camera = commands.add_parser("preview")
    camera.add_argument("--image-size", type=positive, default=256)
    camera.add_argument("--output", type=Path, default=Path("outputs/act_cameras"))
    camera.set_defaults(func=preview)
    args = parser.parse_args()
    if args.command == "train" and args.workers < 0:
        parser.error("workersは0以上にしてください。")
    args.func(args)


if __name__ == "__main__":
    main()

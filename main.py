"""SO-101を表示し、関節状態とRGB画像を確認する最小サンプル。"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "assets/so101/scene.xml"


def make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.16, 0, 0.15]
    camera.distance = 0.85
    camera.azimuth = 135
    camera.elevation = -25
    return camera


def step(model: mujoco.MjModel, data: mujoco.MjData, demo: bool) -> None:
    if demo:
        # 他の関節は初期位置を保持し、旋回とグリッパだけをゆっくり動かす。
        data.ctrl[0] = 0.3 * np.sin(data.time)
        data.ctrl[5] = 0.4 + 0.2 * np.sin(data.time)
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("関節状態が非有限値になりました。")
    if any(warning.number for warning in data.warning):
        raise RuntimeError("MuJoCoがシミュレーションの警告を検出しました。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="GUIを開かず実行")
    parser.add_argument("--demo", action="store_true", help="旋回と開閉の動作デモ")
    parser.add_argument("--seconds", type=float, help="実行するシミュレーション秒数")
    parser.add_argument("--snapshot", type=Path, help="実行後のRGB画像の保存先")
    args = parser.parse_args()
    if args.seconds is not None and (
        not np.isfinite(args.seconds) or args.seconds <= 0
    ):
        parser.error("--seconds は正の有限値にしてください。")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    # 位置制御の目標を初期関節位置に合わせる。単位はrad。
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    mujoco.mj_forward(model, data)
    print(f"SO-101 loaded: {model.nq} joints, {model.nu} actuators", flush=True)

    duration = (
        args.seconds
        if args.seconds is not None
        else (2.0 if args.headless else float("inf"))
    )
    if args.headless:
        while data.time < duration:
            step(model, data, args.demo)
    else:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                camera = make_camera()
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
            while viewer.is_running() and data.time < duration:
                started = time.monotonic()
                with viewer.lock():
                    step(model, data, args.demo)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - started)))

    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera=make_camera())
            Image.fromarray(renderer.render()).save(args.snapshot)
        print(f"RGB image saved: {args.snapshot.resolve()}")
    print(f"Simulated {data.time:.3f}s; qpos(rad)={np.round(data.qpos, 3).tolist()}")


if __name__ == "__main__":
    main()

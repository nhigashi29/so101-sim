"""PPOの成功軌跡を、同期した2画像と6関節目標として保存する。"""

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from pick_place import PickPlaceEnv
from task import require_current_config

ROOT = Path(__file__).resolve().parent.parent
CAMERAS = ("wrist", "overhead")
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
DATA_VERSION = "two-camera-joint-target-v1"


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class CameraPair:
    def __init__(self, env: PickPlaceEnv, size: int) -> None:
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=size, width=size)

    def capture(self) -> dict[str, np.ndarray]:
        images = {}
        for name in CAMERAS:
            self.renderer.update_scene(self.env.data, camera=name)
            images[name] = self.renderer.render().copy()
        return images

    def close(self) -> None:
        self.renderer.close()


def bundle(args: Any) -> None:
    config = json.loads((args.run / "config.json").read_text())
    require_current_config(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        *sorted((ROOT / "src").glob("*.py")),
        *sorted((ROOT / "assets" / "so101").rglob("*")),
        ROOT / "requirements" / "act-colab.txt",
    ]
    manifest: dict[str, Any] = dict(
        format=DATA_VERSION,
        teacher_steps=config["num_timesteps"],
        teacher_source=str(args.run),
        teacher_return_home=bool(config.get("return_home", False)),
        teacher_task_version=config["task_version"],
        files={},
    )
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            if not file.is_file() or file.name == ".DS_Store":
                continue
            relative = str(file.relative_to(ROOT))
            archive.write(file, f"so101_act/{relative}")
            manifest["files"][relative] = digest(file)
        for name in ("policy.zip", "normalize.pkl", "config.json"):
            file = args.run / name
            archive.write(file, f"so101_act/teacher/{name}")
            manifest["files"][f"teacher/{name}"] = digest(file)
        archive.writestr("so101_act/bundle.json", json.dumps(manifest, indent=2))
    print(
        f"Colab用: {args.output.resolve()} ({args.output.stat().st_size / 1e6:.1f} MB)"
    )


def collect(args: Any) -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    config = json.loads((args.run / "config.json").read_text())
    require_current_config(config)
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    teacher_hash = digest(args.run / "policy.zip")
    index: dict[str, Any]
    if index_path.exists():
        if not args.resume:
            raise ValueError(
                "既存データには --resume を指定するか、別の保存先を使ってください。"
            )
        index = json.loads(index_path.read_text())
        if (index["teacher_sha256"], index["image_size"], index["scenarios"]) != (
            teacher_hash,
            args.image_size,
            args.scenarios,
        ):
            raise ValueError("再開元の教師・画像サイズ・シナリオが一致しません。")
    else:
        index = dict(
            format=DATA_VERSION,
            fps=25,
            image_size=args.image_size,
            cameras=list(CAMERAS),
            joint_names=list(JOINTS),
            joint_units="radians",
            action="desired motor joint positions before interpolation; radians",
            teacher_sha256=teacher_hash,
            teacher_steps=config["num_timesteps"],
            scenarios=args.scenarios,
            next_seed=args.seed,
            attempts=0,
            episodes=[],
        )
    raw = PickPlaceEnv(
        stage=args.stage or config["stage"],
        start_level=int(config.get("start_level", 0)),
        workspace_scale=float(config.get("workspace_scale", 0)),
        return_home=bool(config.get("return_home", False)),
        recovery=True,
        disturbance_strength=float(config.get("disturbance_strength", 3)),
        domain_scale=float(config.get("domain_scale", 3)),
        relocation_count=int(config.get("relocation_count", 1)),
    )
    simulation = dict(
        stage=raw.pending_stage,
        start_level=raw.pending_start_level,
        workspace_scale=raw.workspace_scale,
        return_home=raw.return_home,
        recovery=True,
        disturbance_strength=raw.disturbance_strength,
        domain_scale=raw.domain_scale,
        relocation_count=raw.relocation_limit,
    )
    calibration = {
        name: dict(
            position=raw.model.cam_pos[raw.model.camera(name).id].tolist(),
            quaternion=raw.model.cam_quat[raw.model.camera(name).id].tolist(),
            fovy=float(raw.model.cam_fovy[raw.model.camera(name).id]),
        )
        for name in CAMERAS
    }
    if (
        index.get("simulation", simulation) != simulation
        or index.get("camera_calibration", calibration) != calibration
    ):
        raise ValueError("再開元とシミュレーター・カメラ設定が異なります。")
    index["simulation"], index["camera_calibration"] = simulation, calibration
    normalizer = VecNormalize.load(
        str(args.run / "normalize.pkl"), DummyVecEnv([lambda: raw])
    )
    normalizer.training = False
    normalizer.norm_reward = False
    teacher = PPO.load(str(args.run / "policy.zip"), device="cpu")
    pair = CameraPair(raw, args.image_size)
    try:
        for _ in range(args.max_attempts or args.episodes * 8):
            if len(index["episodes"]) >= args.episodes:
                break
            scenario = args.scenarios[len(index["episodes"]) % len(args.scenarios)]
            raw.domain_randomization = raw.disturbances = scenario == "strong"
            raw.cube_relocations = scenario != "standard"
            seed = index["next_seed"]
            torch.manual_seed(seed)
            obs, _ = raw.reset(seed=seed)
            rows: dict[str, list] = {
                key: []
                for key in (*CAMERAS, "state", "action", "time", "segment_start")
            }
            segment_start, previous_interventions = True, (0, 0)
            info: dict[str, Any] = {}
            for _step in range(raw.max_steps):
                views = pair.capture()
                for name in CAMERAS:
                    rows[name].append(views[name])
                rows["state"].append(raw.data.qpos[:6].copy().astype(np.float32))
                rows["time"].append(float(raw.data.time))
                rows["segment_start"].append(segment_start)
                action, _ = teacher.predict(
                    normalizer.normalize_obs(obs), deterministic=True
                )
                obs, _, done, _, info = raw.step(action)
                rows["action"].append(raw.last_command_target.copy().astype(np.float32))
                interventions = (info["relocation_count"], info["disturbance_count"])
                segment_start = interventions != previous_interventions
                previous_interventions = interventions
                if done:
                    break
            if not info.get("termination_reason"):
                raise RuntimeError("教師の試行が期限内に終了していません。")
            accepted = bool(info["is_success"])
            if raw.cube_relocations:
                accepted &= (
                    info["relocation_count"] == raw.relocation_limit
                    and info["regrasp_count"] >= raw.relocation_limit
                )
            index["attempts"] += 1
            index["next_seed"] += 1
            with (args.output / "attempts.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        dict(seed=seed, scenario=scenario, accepted=accepted, **info)
                    )
                    + "\n"
                )
            if accepted:
                number = len(index["episodes"])
                file = args.output / f"episode_{number:05d}.npz"
                arrays: dict[str, Any] = {
                    key: np.asarray(value) for key, value in rows.items()
                }
                np.savez_compressed(file, **arrays)
                if number == 0:
                    for name in CAMERAS:
                        Image.fromarray(arrays[name][0]).save(
                            args.output / f"preview_{name}.png"
                        )
                index["episodes"].append(
                    dict(
                        file=file.name,
                        seed=seed,
                        frames=len(rows["state"]),
                        scenario=scenario,
                        sha256=digest(file),
                        result=info,
                    )
                )
                print(
                    f"accepted {number + 1}/{args.episodes}: {scenario}, seed={seed}, regrasp={info['regrasp_count']}",
                    flush=True,
                )
            else:
                print(
                    f"retry: {scenario}, seed={seed}, {info['termination_reason']}",
                    flush=True,
                )
            write_json(index_path, index)
        if len(index["episodes"]) < args.episodes:
            raise RuntimeError(
                f"成功軌跡は{len(index['episodes'])}/{args.episodes}件。試行上限を増やして --resume できます。"
            )
    finally:
        pair.close()
        normalizer.close()


def chunk_indices(
    segment_starts: np.ndarray, start: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """予測できない位置変更や外力の開始をまたぐ未来ラベルはマスクする。"""
    count = len(segment_starts)
    if not 0 <= start < count or length < 1:
        raise ValueError("無効なチャンク範囲です。")
    boundaries = np.flatnonzero(segment_starts[start + 1 :])
    end = start + 1 + int(boundaries[0]) if len(boundaries) else count
    offsets = start + np.arange(length)
    return np.minimum(offsets, end - 1), offsets >= end


def split_episodes(episodes: list[dict], seed: int) -> tuple[list[int], list[int]]:
    """同じエピソードのフレームを訓練と検証に混ぜない。シナリオで層別化。"""
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for scenario in sorted({e["scenario"] for e in episodes}):
        group = [i for i, e in enumerate(episodes) if e["scenario"] == scenario]
        if len(group) < 2:
            raise ValueError(f"{scenario}は訓練・検証用に最低2エピソード必要です。")
        rng.shuffle(group)
        n = max(1, round(len(group) * 0.2))
        validation.extend(group[:n])
        train.extend(group[n:])
    return train, validation

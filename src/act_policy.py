"""LeRobot 0.6.1のACTを使う、画像方策の学習・保存・推論。"""

import bisect
import importlib
import json
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from act_data import (
    CAMERAS,
    DATA_VERSION,
    CameraPair,
    chunk_indices,
    digest,
    split_episodes,
    write_json,
)
from pick_place import PickPlaceEnv


def act_classes():
    try:
        config = importlib.import_module("lerobot.policies.act.configuration_act")
        model = importlib.import_module("lerobot.policies.act.modeling_act")
        types = importlib.import_module("lerobot.configs.types")
    except ImportError as error:
        raise RuntimeError(
            "ACTはColabノートブック、またはrequirements/act-colab.txtの独立環境で実行してください。"
        ) from error
    return config.ACTConfig, model.ACTPolicy, types.PolicyFeature, types.FeatureType


def normalization(
    root: Path, entries: list[dict], indices: list[int], chunk_size: int
) -> dict:
    stats: dict[str, Any] = {}
    # Rate-limited targets sit within one control step of the state, so absolute
    # labels bury the motion under the model error. Learn offsets from the
    # observed state, normalized per future step because they grow with horizon.
    count = np.zeros((chunk_size, 1))
    total, squares = np.zeros((chunk_size, 6)), np.zeros((chunk_size, 6))
    for i in indices:
        with np.load(root / entries[i]["file"], allow_pickle=False) as episode:
            state = episode["state"].astype(np.float64)
            action = episode["action"].astype(np.float64)
            starts = episode["segment_start"]
        for offset in range(len(state)):
            future, pad = chunk_indices(starts, offset, chunk_size)
            valid = ~pad[:, None]
            delta = (action[future] - state[offset]) * valid
            count += valid
            total += delta
            squares += np.square(delta)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(squares / np.maximum(count, 1) - mean**2, 1e-8))
    stats["delta"] = dict(mean=mean.tolist(), std=std.tolist())
    for key in ("state", "action"):
        count, total, squares = 0, np.zeros(6), np.zeros(6)
        for i in indices:
            with np.load(root / entries[i]["file"], allow_pickle=False) as episode:
                values = episode[key].astype(np.float64)
                total += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                count += len(values)
        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean**2, 1e-6))
        stats[key] = dict(mean=mean.tolist(), std=std.tolist())
    stats["image"] = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return stats


def prepare_frames(root: Path, entries: list[dict]) -> dict[int, Path]:
    """圧縮NPZを一度だけ展開。ランダムな1フレーム取得で全軌跡を解凍しない。"""
    paths = {}
    for i, entry in enumerate(entries):
        expected = entry.get("sha256") or digest(root / entry["file"])
        folder = root / ".act_frames" / f"{Path(entry['file']).stem}_{expected[:12]}"
        ready = folder / "ready.json"
        paths[i] = folder
        if ready.exists() and json.loads(ready.read_text()).get("sha256") == expected:
            continue
        if digest(root / entry["file"]) != expected:
            raise ValueError(f"データのSHA-256が一致しません: {entry['file']}")
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(root / entry["file"], allow_pickle=False) as episode:
            for key in (*CAMERAS, "state", "action", "segment_start"):
                np.save(folder / f"{key}.npy", episode[key], allow_pickle=False)
        write_json(ready, dict(sha256=expected))
    return paths


class EpisodeDataset(Dataset):
    def __init__(
        self,
        root: Path,
        entries: list[dict],
        indices: list[int],
        stats: dict,
        chunk_size: int,
    ) -> None:
        self.root, self.entries, self.indices = root, entries, indices
        self.frame_paths = prepare_frames(root, entries)
        self.stats, self.chunk_size = stats, chunk_size
        self.delta = "delta" in stats
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.ends = np.cumsum([entries[i]["frames"] for i in indices]).tolist()

    def __len__(self) -> int:
        return self.ends[-1]

    def episode(self, number: int) -> dict[str, np.ndarray]:
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        folder = self.frame_paths[self.indices[number]]
        result = {
            key: np.load(folder / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in (*CAMERAS, "state", "action", "segment_start")
        }
        self.cache[number] = result
        if len(self.cache) > 2:
            self.cache.popitem(last=False)
        return result

    def __getitem__(self, number: int) -> dict[str, torch.Tensor]:
        episode_number = bisect.bisect_right(self.ends, number)
        offset = number - (self.ends[episode_number - 1] if episode_number else 0)
        episode = self.episode(episode_number)
        indices, pad = chunk_indices(episode["segment_start"], offset, self.chunk_size)
        batch = {
            "observation.state": torch.tensor(
                (episode["state"][offset] - np.array(self.stats["state"]["mean"]))
                / np.array(self.stats["state"]["std"]),
                dtype=torch.float32,
            ),
            "action": torch.tensor(
                (
                    episode["action"][indices]
                    - episode["state"][offset]
                    - np.array(self.stats["delta"]["mean"])
                )
                / np.array(self.stats["delta"]["std"])
                if self.delta
                else (
                    episode["action"][indices] - np.array(self.stats["action"]["mean"])
                )
                / np.array(self.stats["action"]["std"]),
                dtype=torch.float32,
            ),
            "action_is_pad": torch.tensor(pad, dtype=torch.bool),
        }
        for name in CAMERAS:
            image = (
                torch.from_numpy(episode[name][offset].copy()).permute(2, 0, 1).float()
                / 255
            )
            mean = torch.tensor(self.stats["image"]["mean"])[:, None, None]
            std = torch.tensor(self.stats["image"]["std"])[:, None, None]
            batch[f"observation.images.{name}"] = (image - mean) / std
        return batch


def load_policy(checkpoint: Path, device: str):
    config_class, policy_class, _, _ = act_classes()
    config = config_class.from_pretrained(checkpoint / "policy")
    config.device = device
    config.pretrained_backbone_weights = None
    policy = policy_class.from_pretrained(checkpoint / "policy", config=config)
    return (
        policy,
        json.loads((checkpoint / "normalization.json").read_text()),
        json.loads((checkpoint / "training.json").read_text()),
    )


def train(args: Any) -> None:
    config_class, policy_class, feature, feature_type = act_classes()
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ColabのランタイムをGPUに変更してください。")
    torch.manual_seed(args.seed)
    index = json.loads((args.dataset / "index.json").read_text())
    if index["format"] != DATA_VERSION:
        raise ValueError("異なる形式のデータセットです。")
    fingerprint = digest(args.dataset / "index.json")
    if args.resume:
        policy, stats, metadata = load_policy(args.resume, args.device)
        if metadata["dataset_sha256"] != fingerprint:
            raise ValueError("再開元とデータセットが異なります。")
        train_indices, val_indices = (
            metadata["train_episodes"],
            metadata["validation_episodes"],
        )
        start = metadata["step"]
    else:
        train_indices, val_indices = split_episodes(index["episodes"], args.seed)
        stats = normalization(
            args.dataset, index["episodes"], train_indices, args.chunk_size
        )
        size = index["image_size"]
        config = config_class(
            device=args.device,
            push_to_hub=False,
            chunk_size=args.chunk_size,
            n_action_steps=1,
            input_features={
                "observation.state": feature(type=feature_type.STATE, shape=(6,)),
                **{
                    f"observation.images.{name}": feature(
                        type=feature_type.VISUAL, shape=(3, size, size)
                    )
                    for name in CAMERAS
                },
            },
            output_features={"action": feature(type=feature_type.ACTION, shape=(6,))},
            pretrained_backbone_weights=None
            if args.tiny
            else "ResNet18_Weights.IMAGENET1K_V1",
            **(
                dict(
                    dim_model=64,
                    n_heads=4,
                    dim_feedforward=128,
                    n_encoder_layers=1,
                    n_decoder_layers=1,
                    n_vae_encoder_layers=1,
                )
                if args.tiny
                else {}
            ),
        )
        policy = policy_class(config).to(args.device)
        metadata = dict(
            dataset_sha256=fingerprint,
            train_episodes=train_indices,
            validation_episodes=val_indices,
            image_size=size,
            fps=index["fps"],
            simulation=index["simulation"],
            joint_names=index["joint_names"],
            action=index["action"],
            preprocessing="manual mean/std, ImageNet image normalization",
            action_representation="joint-target offsets from the observed state, normalized per future step",
            lerobot_version="0.6.1",
            chunk_size=args.chunk_size,
            n_action_steps=1,
        )
        start = 0
    policy.config.optimizer_lr = args.learning_rate
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.learning_rate, weight_decay=1e-4
    )
    if args.resume:
        saved = torch.load(
            args.resume / "optimizer.pt", map_location=args.device, weights_only=True
        )
        optimizer.load_state_dict(saved["optimizer"])
        optimizer.param_groups[0]["lr"] = args.learning_rate
        optimizer.param_groups[1]["lr"] = policy.config.optimizer_lr_backbone
    args.output.mkdir(parents=True, exist_ok=True)
    chunk_size = policy.config.chunk_size
    print("Preparing local memory-mapped frame cache...", flush=True)
    train_data = EpisodeDataset(
        args.dataset, index["episodes"], train_indices, stats, chunk_size
    )
    val_data = EpisodeDataset(
        args.dataset, index["episodes"], val_indices, stats, chunk_size
    )
    generator = torch.Generator().manual_seed(args.seed + start)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
        multiprocessing_context="spawn" if args.workers > 0 else None,
    )
    val_samples = (
        np.random.default_rng(args.seed + 1)
        .choice(
            len(val_data),
            min(len(val_data), args.batch_size * args.validation_batches),
            replace=False,
        )
        .tolist()
    )
    val_loader = DataLoader(
        Subset(val_data, val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    batches = iter(loader)
    best_path = args.output / "best.json"
    best = (
        json.loads(best_path.read_text())["validation_l1"]
        if best_path.exists()
        else float("inf")
    )
    print(
        f"ACT: train={len(train_indices)} episodes/{len(train_data)} frames, validation={len(val_indices)} episodes, device={args.device}",
        flush=True,
    )
    for step in range(start + 1, start + args.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        batch = {
            key: value.to(args.device, non_blocking=True)
            for key, value in batch.items()
        }
        policy.train()
        loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("ACTのlossが非有限になりました。")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step}, train_loss={loss.item():.5f}", flush=True)
        if step % args.save_every != 0 and step != start + args.steps:
            continue
        policy.eval()
        validation = []
        with torch.inference_mode():
            for i, val_batch in enumerate(val_loader):
                val_batch = {
                    key: value.to(args.device) for key, value in val_batch.items()
                }
                _, metrics = policy(val_batch)
                validation.append(metrics["l1_loss"])
                if i + 1 >= args.validation_batches:
                    break
        score = float(np.mean(validation))
        checkpoint = args.output / f"step_{step:07d}"
        if checkpoint.exists():
            raise ValueError(f"保存済みチェックポイントは上書きしません: {checkpoint}")
        staging = args.output / f".step_{step:07d}.incomplete"
        staging.mkdir(exist_ok=True)
        policy.save_pretrained(staging / "policy")
        write_json(staging / "normalization.json", stats)
        write_json(
            staging / "training.json",
            {**metadata, "step": step, "validation_l1": score},
        )
        torch.save({"optimizer": optimizer.state_dict()}, staging / "optimizer.pt")
        staging.rename(checkpoint)
        write_json(
            args.output / "latest.json", dict(checkpoint=checkpoint.name, step=step)
        )
        if score < best:
            best = score
            write_json(
                args.output / "best.json",
                dict(checkpoint=checkpoint.name, step=step, validation_l1=score),
            )
        with (args.output / "metrics.jsonl").open("a") as file:
            file.write(
                json.dumps(dict(step=step, train_loss=loss.item(), validation_l1=score))
                + "\n"
            )
        print(f"saved {checkpoint}, validation_l1={score:.5f}", flush=True)


def visual_batch(
    images: dict[str, np.ndarray], joints: np.ndarray, stats: dict, device: str
) -> dict:
    batch = {
        "observation.state": torch.tensor(
            (joints - np.array(stats["state"]["mean"]))
            / np.array(stats["state"]["std"]),
            dtype=torch.float32,
            device=device,
        )[None]
    }
    for name in CAMERAS:
        image = (
            torch.tensor(
                images[name].copy(), dtype=torch.float32, device=device
            ).permute(2, 0, 1)
            / 255
        )
        mean = torch.tensor(stats["image"]["mean"], device=device)[:, None, None]
        std = torch.tensor(stats["image"]["std"], device=device)[:, None, None]
        batch[f"observation.images.{name}"] = ((image - mean) / std)[None]
    return batch


def evaluate(args: Any) -> None:
    torch.set_num_threads(2)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    policy, stats, metadata = load_policy(args.run, device)
    policy.eval()
    if not 1 <= args.action_steps <= policy.config.chunk_size:
        raise ValueError(f"action-stepsは1〜{policy.config.chunk_size}にしてください。")
    simulation = metadata["simulation"]
    raw = PickPlaceEnv(
        **simulation,
        domain_randomization=args.scenario == "strong",
        disturbances=args.scenario == "strong",
        cube_relocations=args.scenario != "standard",
    )
    pair = CameraPair(raw, metadata["image_size"])
    viewer = None
    try:
        if args.watch:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(raw.model, raw.data)
            with viewer.lock():
                viewer.cam.lookat[:] = [0.20, 0.02, 0.08]
                viewer.cam.distance = 0.75
                viewer.cam.azimuth, viewer.cam.elevation = 135, -30
        results = []
        for episode in range(args.episodes):
            policy.reset()
            # Execute the first k actions of each predicted chunk before re-planning.
            planned: deque[np.ndarray] = deque()
            if viewer is not None:
                with viewer.lock():
                    raw.reset(seed=args.seed + episode)
            else:
                raw.reset(seed=args.seed + episode)
            for _ in range(raw.max_steps):
                started = time.monotonic()
                if not planned:
                    images = pair.capture()
                    batch = visual_batch(
                        images, raw.data.qpos[:6].copy(), stats, device
                    )
                    with torch.inference_mode():
                        chunk = policy.predict_action_chunk(batch)
                    chunk = chunk[0, : args.action_steps].cpu().numpy()
                    if "delta" in stats:
                        mean = np.array(stats["delta"]["mean"])[: len(chunk)]
                        std = np.array(stats["delta"]["std"])[: len(chunk)]
                        targets = raw.data.qpos[:6] + chunk * std + mean
                    else:
                        targets = chunk * np.array(stats["action"]["std"]) + np.array(
                            stats["action"]["mean"]
                        )
                    planned.extend(targets)
                target = planned.popleft()
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    with viewer.lock():
                        _, _, done, _, info = raw.step_joint_targets(target)
                    viewer.sync()
                    time.sleep(max(0, raw.dt - (time.monotonic() - started)))
                else:
                    _, _, done, _, info = raw.step_joint_targets(target)
                if done:
                    results.append(dict(seed=args.seed + episode, **info))
                    print(
                        f"episode {episode + 1}: {info['termination_reason']}, regrasp={info['regrasp_count']}",
                        flush=True,
                    )
                    break
            else:
                raise RuntimeError("画像方策の試行が期限内に終了していません。")
        report = dict(
            episodes=len(results),
            successes=sum(r["is_success"] for r in results),
            recovered_successes=sum(
                r["is_success"] and r["regrasp_count"] > 0 for r in results
            ),
            scenario=args.scenario,
            seed=args.seed,
            action_steps=args.action_steps,
            trials=results,
        )
        write_json(
            args.run
            / f"evaluation_{args.scenario}_k{args.action_steps}_seed{args.seed}_n{args.episodes}.json",
            report,
        )
        print(f"ACT success: {report['successes']}/{report['episodes']}")
    finally:
        if viewer is not None:
            viewer.close()
        pair.close()
        raw.close()

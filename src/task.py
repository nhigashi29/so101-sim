"""単一PPO方策の学習・評価と、物理接触を検証する固定動作デモ。"""

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pick_place import (
    GAMMA,
    RECOVERY_REWARD_VERSION,
    RECOVERY_TASK_VERSION,
    RETURN_REWARD_VERSION,
    RETURN_TASK_VERSION,
    REWARD_VERSION,
    STAGES,
    TASK_VERSION,
    PickPlaceEnv,
)


def scripted_actions(env: PickPlaceEnv):
    """検証用の固定動作。方策の推論や学習には使用しない。"""
    start = env.cube[:2].copy() + [0, 0.004]
    goal = env.goal[:2].copy() - [0, 0.003]
    waypoints = []
    if env.stage in ("grasp", "pick", "random"):
        waypoints += [
            ("descend", start, 0.026, 0.8, 45),
            ("grasp", start, 0.026, -0.12, 25),
            ("lift", start, 0.081, -0.12, 40),
        ]
    if env.stage == "lift":
        waypoints += [("lift", env.tip[:2].copy(), 0.081, -0.12, 40)]
    if env.stage in ("transport", "lift", "grasp", "pick", "random"):
        waypoints += [("transport", goal, 0.081, -0.12, 50)]
    waypoints += [
        ("lower", goal, 0.024, -0.12, 45),
        ("open", goal + [0, 0.004], 0.024, 0.8, 25),
    ]
    for name, xy, z, jaw, steps in waypoints:
        for _ in range(steps):
            action = np.r_[
                np.clip((np.r_[xy, z] - env.tip) * 12 / env.tcp_speed, -1, 1),
                np.clip((jaw - env.data.ctrl[5]) * 8 / env.jaw_speed, -1, 1),
            ]
            yield name, action.astype(np.float32)
    if env.return_home:
        # Test-only motion; PPO and ACT must learn and execute their own actions.
        while not env.state.reason:
            target = (
                np.r_[env.retreat_anchor[:2], 0.085]
                if env.state.phase.name == "RETREAT"
                else env.home_tip
            )
            if env.state.phase.name in ("LOWER", "RELEASE", "SETTLE"):
                target = np.r_[goal + [0, 0.004], 0.024]
            action = np.r_[
                np.clip((target - env.tip) * 12 / env.tcp_speed, -1, 1),
                np.clip((0.8 - env.data.ctrl[5]) * 8 / env.jaw_speed, -1, 1),
            ]
            yield "return", action.astype(np.float32)


def run_demo(args: argparse.Namespace) -> None:
    env = PickPlaceEnv(stage=args.stage, camera=args.camera)
    env.reset(seed=args.seed)
    viewer = None
    frames: list[Image.Image] = []
    try:
        if not args.headless:
            import mujoco
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(env.model, env.data)
            with viewer.lock():
                viewer.cam.lookat[:] = [0.19, 0.01, 0.10]
                viewer.cam.distance = 0.8
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -30
                if args.camera != "free":
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = env.model.camera(args.camera).id
        results = []
        for episode in range(args.episodes):
            seed = args.seed + episode
            if viewer is not None:
                if not viewer.is_running():
                    return
                with viewer.lock():
                    env.reset(seed=seed)
            else:
                env.reset(seed=seed)
            print(f"Trial {episode + 1}/{args.episodes}, seed={seed}", flush=True)
            last_phase = ""
            info: dict[str, Any] = {}
            for phase, action in scripted_actions(env):
                started = time.monotonic()
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    with viewer.lock():
                        _, _, terminated, truncated, info = env.step(action)
                    viewer.sync()
                    time.sleep(max(0, env.dt - (time.monotonic() - started)))
                else:
                    _, _, terminated, truncated, info = env.step(action)
                if phase != last_phase:
                    print(f"{env.data.time:.2f}s: {phase}", flush=True)
                    last_phase = phase
                if args.gif and env.steps % 5 == 0:
                    frames.append(Image.fromarray(env.render()))
                if terminated or truncated:
                    break
            results.append({"seed": seed, **info})
            print(json.dumps(results[-1], ensure_ascii=False, indent=2), flush=True)
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(env.render()).save(args.snapshot)
        if args.gif and frames:
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                args.gif,
                save_all=True,
                append_images=frames[1:],
                duration=round(PickPlaceEnv.dt * 5000),
                loop=0,
            )
        successes = sum(bool(result["is_success"]) for result in results)
        print(f"Success: {successes}/{len(results)}", flush=True)
        if successes != len(results):
            raise SystemExit("成功条件を満たさない固定動作デモがありました。")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def require_current_config(config: dict[str, Any]) -> None:
    if config.get("task_version") not in (
        TASK_VERSION,
        RECOVERY_TASK_VERSION,
        RETURN_TASK_VERSION,
    ):
        raise ValueError(
            f"この方策は旧タスク {config.get('task_version')} 用です。"
            f"{TASK_VERSION} は観測・行動または終了条件が異なるため、新規学習が必要です。"
        )


def training_stages(stage: str, envs: int, fixed: bool = False) -> list[str]:
    assignments = [stage] * envs
    if not fixed and envs > 1 and stage != STAGES[0]:
        previous = STAGES[STAGES.index(stage) - 1]
        count = min(envs - 1, max(1, round(envs * 0.25)))
        assignments[-count:] = [previous] * count
    return assignments


def start_assignments(stage: str, envs: int, level: int, expanding: bool, fixed: bool):
    if stage == "random" and expanding:
        levels = [level] * envs
        if envs > 1:
            count = min(envs - 1, max(1, round(envs * 0.25)))
            levels[-count:] = [max(0, level - 1)] * count
        return [("random", item) for item in levels]
    return [
        (item, level if item == "random" else 0)
        for item in training_stages(stage, envs, fixed)
    ]


def start_curriculum_passed(reports: list[dict[str, Any]]) -> bool:
    return bool(reports) and all(
        r["success_rate"] >= (0.8 if r["deterministic"] else 0.6) for r in reports
    )


def simulation_options(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    defaults = dict(
        recovery=False,
        cube_relocations=False,
        relocation_count=1,
        disturbance_strength=1.0,
        domain_scale=1.0,
        workspace_scale=0.0,
        return_home=False,
    )
    return {
        key: getattr(args, key)
        if getattr(args, key) is not None
        else config.get(key, default)
        for key, default in defaults.items()
    }


def train(args: argparse.Namespace) -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {}
    start_stage = args.stage or STAGES[0]
    if args.resume:
        config = json.loads((args.resume / "config.json").read_text())
        start_stage = args.stage or config["stage"]
        require_current_config(config)

    domain_randomization = (
        args.domain_randomization
        if args.domain_randomization is not None
        else bool(config.get("domain_randomization", False))
    )
    disturbances = (
        args.disturbances
        if args.disturbances is not None
        else bool(config.get("disturbances", False))
    )
    options = simulation_options(args, config)
    if options["return_home"] and not args.resume:
        raise ValueError(
            "帰還は配置方策から転移学習します。--resumeを指定してください。"
        )
    task_version = (
        RETURN_TASK_VERSION
        if options["return_home"]
        else RECOVERY_TASK_VERSION
        if options["recovery"]
        else TASK_VERSION
    )
    migrating_return = bool(
        args.resume
        and options["return_home"]
        and (
            config["task_version"] != RETURN_TASK_VERSION
            or config.get("policy_architecture") != "preserved-pick-and-return-v2"
        )
    )
    if (
        args.resume
        and config["task_version"] == RETURN_TASK_VERSION
        and not options["return_home"]
    ):
        raise ValueError("帰還方策を旧タスクへ戻す再開はできません。")
    if migrating_return and args.resume.resolve() == args.output.resolve():
        raise ValueError(
            "帰還タスクへの転移は、元の方策とは別のoutputに保存してください。"
        )
    reward_version = (
        RETURN_REWARD_VERSION
        if options["return_home"]
        else RECOVERY_REWARD_VERSION
        if options["recovery"]
        else REWARD_VERSION
    )
    robust = domain_randomization or disturbances or options["cube_relocations"]
    physics_changed = domain_randomization != config.get(
        "domain_randomization", False
    ) or disturbances != config.get("disturbances", False)
    physics_changed = physics_changed or any(
        options[key] != config.get(key, default)
        for key, default in dict(
            recovery=False,
            cube_relocations=False,
            relocation_count=1,
            disturbance_strength=1.0,
            domain_scale=1.0,
            workspace_scale=0.0,
            return_home=False,
        ).items()
    )
    physics_changed = physics_changed or config.get("reward_version") != reward_version
    expanding = args.expand_starts or config.get("expand_starts", False)
    start_level = (
        args.start_level
        if args.start_level is not None
        else int(config.get("start_level", 0))
    )
    max_start_level = (
        args.max_start_level
        if args.max_start_level is not None
        else int(config.get("max_start_level", 5))
    )
    if start_level > max_start_level:
        raise ValueError("start-level は max-start-level 以下にしてください。")

    class Curriculum(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.stage = start_stage
            self.start_level = start_level
            self.promotion_streak = (
                int(config.get("start_promotion_streak", 0))
                if not physics_changed and start_level == config.get("start_level", 0)
                else 0
            )
            self.last_validation = int(config.get("num_timesteps", 0))
            self.best_scores: dict[str, float] = (
                config.get("best_scores", {})
                if not physics_changed
                and config.get("validation_episodes", 10) == args.eval_episodes
                else {}
            )
            self.challenge_results: deque[bool] = deque(maxlen=50)
            self.regrasp_results: deque[bool] = deque(maxlen=50)
            self.results: deque[bool] = deque(
                config.get("curriculum_results", [])
                if args.resume
                and not physics_changed
                and start_stage == config["stage"]
                else [],
                maxlen=50,
            )

        def _on_step(self) -> bool:
            for done, info in zip(
                self.locals["dones"], self.locals["infos"], strict=True
            ):
                if (
                    done
                    and info["stage"] == self.stage
                    and info["start_level"] == self.start_level
                ):
                    self.results.append(bool(info["is_success"]))
                    if (
                        info["domain_randomization"]
                        or info["disturbances"]
                        or info["relocations_enabled"]
                    ):
                        self.challenge_results.append(bool(info["is_success"]))
                        self.regrasp_results.append(info["regrasp_count"] > 0)
            if self.results:
                self.logger.record(
                    "curriculum/success_rate", float(np.mean(self.results))
                )
            if self.challenge_results:
                self.logger.record(
                    "curriculum/challenge_success",
                    float(np.mean(self.challenge_results)),
                )
                self.logger.record(
                    "curriculum/regrasp_rate", float(np.mean(self.regrasp_results))
                )
            self.logger.record("curriculum/stage", STAGES.index(self.stage))
            self.logger.record("curriculum/start_level", self.start_level)
            if self.num_timesteps - self.last_validation >= args.eval_every:
                self.last_validation = self.num_timesteps
                self.validate()
                save_bundle(args.output / "checkpoints" / f"step_{self.num_timesteps}")
            return True

        def validate(self) -> None:
            # Evaluation uses frozen training statistics, held-out seeds and its
            # own physics instance. Preserve torch RNG so evaluation does not
            # change subsequent rollout exploration.
            rng = torch.random.get_rng_state()
            reports = []
            stages = [(self.stage, self.start_level if self.stage == "random" else 0)]
            if self.stage == "random" and expanding:
                stages.extend(
                    ("random", level)
                    for level in sorted({0, max(0, self.start_level - 1)})
                    if level != self.start_level
                )
                stages.append(("pick", 0))
            elif self.stage != STAGES[0]:
                stages.append((STAGES[STAGES.index(self.stage) - 1], 0))
            scenarios = [(stage, level, False, False) for stage, level in stages]
            if robust:
                scenarios.append(
                    (
                        self.stage,
                        self.start_level if self.stage == "random" else 0,
                        True,
                        options["cube_relocations"],
                    )
                )
            if options["cube_relocations"]:
                scenarios.append(
                    (
                        self.stage,
                        self.start_level if self.stage == "random" else 0,
                        False,
                        True,
                    )
                )
            try:
                for stage, level, randomized, relocate in scenarios:
                    evaluation = PickPlaceEnv(
                        stage=stage,
                        start_level=level,
                        domain_randomization=randomized and domain_randomization,
                        disturbances=randomized and disturbances,
                        **{**options, "cube_relocations": relocate},
                    )
                    try:
                        for deterministic in (True, False):
                            results = []
                            episode_infos = []
                            reasons: dict[str, int] = {}
                            for seed in range(10000, 10000 + args.eval_episodes):
                                torch.manual_seed(seed)
                                obs, _ = evaluation.reset(seed=seed)
                                for _ in range(evaluation.max_steps):
                                    action, _ = self.model.predict(
                                        env.normalize_obs(obs),
                                        deterministic=deterministic,
                                    )
                                    obs, _, done, timeout, info = evaluation.step(
                                        action
                                    )
                                    if done or timeout:
                                        results.append(bool(info["is_success"]))
                                        episode_infos.append(info)
                                        reason = info["termination_reason"]
                                        reasons[reason] = reasons.get(reason, 0) + 1
                                        break
                            if len(results) != args.eval_episodes:
                                raise RuntimeError(
                                    "検証の試行が期限内に終了していません。"
                                )
                            reports.append(
                                dict(
                                    stage=stage,
                                    start_level=level,
                                    randomized=randomized,
                                    cube_relocations=relocate,
                                    task_version=evaluation.task_version,
                                    relocated_episodes=sum(
                                        i["relocation_count"] > 0 for i in episode_infos
                                    ),
                                    regrasp_episodes=sum(
                                        i["regrasp_count"] > 0 for i in episode_infos
                                    ),
                                    recovered_successes=sum(
                                        i["is_success"] and i["regrasp_count"] > 0
                                        for i in episode_infos
                                    ),
                                    deterministic=deterministic,
                                    success_rate=float(np.mean(results)),
                                    episodes=len(results),
                                    reasons=reasons,
                                )
                            )
                    finally:
                        evaluation.close()
            finally:
                torch.random.set_rng_state(rng)
            record = dict(
                timesteps=self.num_timesteps,
                task_version=task_version,
                start_level=self.start_level,
                evaluations=reports,
            )
            with (args.output / "validation.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            for r in reports:
                mode = "det" if r["deterministic"] else "sample"
                condition = "dr" if r["randomized"] else "base"
                if r["cube_relocations"]:
                    condition += "_move"
                self.logger.record(
                    f"validation/{r['stage']}/l{r['start_level']}/{condition}/{mode}",
                    r["success_rate"],
                )
            current_score = min(
                r["success_rate"] for r in reports if r["stage"] == self.stage
            )
            key = (
                f"random_level{self.start_level}"
                if self.stage == "random" and expanding
                else self.stage
            )
            if robust:
                key += "_robust"
            if options["recovery"]:
                key += "_recovery"
            if current_score > self.best_scores.get(key, -1):
                self.best_scores[key] = current_score
                save_bundle(args.output / "best" / key)
            print(f"Validation: {record}", flush=True)
            # Require held-out performance and retention, rather than just the
            # stochastic training scoreboard, before increasing difficulty.
            passed = start_curriculum_passed(reports)
            if self.stage == "random" and expanding:
                self.promotion_streak = self.promotion_streak + 1 if passed else 0
                if self.promotion_streak >= 2 and self.start_level < max_start_level:
                    self.start_level += 1
                    self.promotion_streak = 0
                    assign_stage(self.stage)
                    self.results.clear()
                    print(
                        f"開始範囲 -> level {self.start_level} (next reset)", flush=True
                    )
            elif not args.fixed_stage and self.stage != "random" and passed:
                self.stage = STAGES[STAGES.index(self.stage) + 1]
                assign_stage(self.stage)
                self.results.clear()
                print(f"Curriculum -> {self.stage} (next reset)", flush=True)

    raw = make_vec_env(
        PickPlaceEnv,
        n_envs=args.envs,
        seed=args.seed,
        env_kwargs={
            "stage": start_stage,
            "start_level": start_level,
            "domain_randomization": domain_randomization,
            "disturbances": disturbances,
            "randomization_probability": 0.75,
            "return_start_probability": (
                args.return_start_probability if options["return_home"] else 0.0
            ),
            **options,
        },
        vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv,
        vec_env_kwargs={"start_method": "spawn"} if args.envs > 1 else {},
        monitor_dir=str(args.output / "monitor" / f"run_{time.time_ns()}"),
        monitor_kwargs={
            "info_keywords": (
                "is_success",
                "stage",
                "start_level",
                "phase",
                "termination_reason",
                "full_task",
                "episode_torque_cost",
                "curriculum_start",
                "elapsed_seconds",
                "eased_start",
            )
        },
    )

    def assign_stage(stage: str) -> None:
        level = curriculum.start_level
        for index, (assigned, spatial_level) in enumerate(
            start_assignments(stage, args.envs, level, expanding, args.fixed_stage)
        ):
            raw.env_method(
                "set_stage",
                assigned,
                not args.fixed_stage,
                args.envs == 1 and not args.fixed_stage and not expanding,
                indices=[index],
            )
            raw.env_method(
                "set_start_level",
                spatial_level,
                expanding and args.envs == 1,
                indices=[index],
            )

    curriculum = Curriculum()
    assign_stage(start_stage)
    entropy_coefficient = (
        args.entropy_coefficient
        if args.entropy_coefficient is not None
        else float(config.get("entropy_coefficient", 0.001))
    )
    if migrating_return:
        from policy_transfer import transfer_to_return

        env, model = transfer_to_return(
            args.resume, raw, args.learning_rate, args.target_kl, entropy_coefficient
        )
        print(
            "帰還タスクへ転移: 既習重み・正規化を保持、退避・帰還の出力とoptimizerを初期化"
        )
    elif args.resume:
        env = VecNormalize.load(str(args.resume / "normalize.pkl"), raw)
        env.training = True
        model = PPO.load(
            str(args.resume / "policy.zip"),
            env=env,
            device="cpu",
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            ent_coef=entropy_coefficient,
        )
    else:
        env = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            n_steps=args.rollout,
            batch_size=min(256, args.rollout * args.envs),
            n_epochs=10,
            gamma=GAMMA,
            gae_lambda=0.95,
            ent_coef=entropy_coefficient,
            policy_kwargs={"net_arch": [128, 128], "log_std_init": -0.8},
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
    if args.exploration_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.exploration_std)))
    if options["return_home"]:
        # Keep the frozen pick actor's input distribution identical to its source.
        env.training = False
    print(
        f"PPO: learning_rate={model.learning_rate}, target_kl={model.target_kl}",
        flush=True,
    )

    def save_bundle(directory: Path) -> None:
        action_shape = env.action_space.shape
        assert action_shape is not None
        directory.mkdir(parents=True, exist_ok=True)
        model.save(str(directory / "policy.zip"))
        env.save(str(directory / "normalize.pkl"))
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "stage": curriculum.stage,
                    "training_env_stages": [
                        item[0]
                        for item in start_assignments(
                            curriculum.stage,
                            args.envs,
                            curriculum.start_level,
                            expanding,
                            args.fixed_stage,
                        )
                    ],
                    "training_env_start_levels": [
                        item[1]
                        for item in start_assignments(
                            curriculum.stage,
                            args.envs,
                            curriculum.start_level,
                            expanding,
                            args.fixed_stage,
                        )
                    ],
                    "expand_starts": expanding,
                    "start_level": curriculum.start_level,
                    "max_start_level": max_start_level,
                    "start_promotion_streak": curriculum.promotion_streak,
                    "start_distribution_version": "spatial-v1",
                    "domain_randomization": domain_randomization,
                    "disturbances": disturbances,
                    "physics_randomization_version": "cube-push-and-physics-v1",
                    **options,
                    "training_randomization_probability": 0.75,
                    "task_version": task_version,
                    "policy_architecture": "preserved-pick-and-return-v2"
                    if options["return_home"]
                    else "standard-mlp",
                    "failure_penalty": PickPlaceEnv.failure_penalty,
                    "timeout_penalty": PickPlaceEnv.timeout_penalty,
                    "seed": args.seed,
                    "num_timesteps": model.num_timesteps,
                    "learning_rate": model.learning_rate,
                    "target_kl": model.target_kl,
                    "observation_dim": int(env.observation_space.shape[0]),
                    "action": "world TCP velocity xyz (0.08 m/s) + jaw rate (1.5 rad/s)",
                    "gamma": GAMMA,
                    "control_dt": PickPlaceEnv.dt,
                    "deadline_terminal": True,
                    "post_release_control": (
                        "policy: vertical retreat then return home and stay still"
                        if options["return_home"]
                        else "hold last command; no further policy action"
                    ),
                    "settle_seconds": 0.4,
                    "reward_version": reward_version,
                    "episode_seconds": (24 if options["recovery"] else 16)
                    + (6 if options["return_home"] else 0),
                    "return_start_probability": (
                        args.return_start_probability if options["return_home"] else 0.0
                    ),
                    "home_tip": [0.23, 0.0, 0.09],
                    "transfer_source": str(args.resume)
                    if migrating_return
                    else config.get("transfer_source"),
                    "entropy_coefficient": model.ent_coef,
                    "exploration_reset_std": args.exploration_std,
                    "training_start_distribution": (
                        "75% current spatial level + 25% previous spatial level; full task, no eased starts"
                        if expanding and curriculum.stage == "random"
                        else "50% canonical + 50% eased starts; fixed previous-stage environments"
                    ),
                    "action_dim": int(action_shape[0]),
                    "curriculum_results": list(curriculum.results),
                    "best_scores": curriculum.best_scores,
                    "validation_episodes": args.eval_episodes,
                    "validation_seed": 10000,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    interrupted = False
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=curriculum,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        interrupted = True
        print("中断時点の方策と正規化統計を保存します。", flush=True)
    finally:
        save_bundle(args.output)
        if interrupted and isinstance(raw, SubprocVecEnv):
            # A SIGINT can interrupt recv halfway through step_wait. close() would
            # wait for a reply already consumed; stop our own workers explicitly.
            for process in raw.processes:
                if process.is_alive():
                    process.terminate()
            for process in raw.processes:
                process.join(timeout=5)
            for remote in raw.remotes:
                remote.close()
            raw.closed = True
        else:
            env.close()
        print(f"保存先: {args.output.resolve()}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    config = json.loads((args.run / "config.json").read_text())
    require_current_config(config)
    level = (
        args.start_level
        if args.start_level is not None
        else int(config.get("start_level", 0))
    )
    domain_randomization = (
        args.domain_randomization
        if args.domain_randomization is not None
        else bool(config.get("domain_randomization", False))
    )
    disturbances = (
        args.disturbances
        if args.disturbances is not None
        else bool(config.get("disturbances", False))
    )
    options = simulation_options(args, config)
    if options["return_home"] != (config["task_version"] == RETURN_TASK_VERSION):
        raise ValueError(
            "帰還の有無が方策と一致しません。先に--return-homeで追加学習してください。"
        )
    raw = PickPlaceEnv(
        stage=args.stage,
        camera=args.camera,
        start_level=level,
        domain_randomization=domain_randomization,
        disturbances=disturbances,
        **options,
    )
    vec = DummyVecEnv([lambda: raw])
    env = VecNormalize.load(str(args.run / "normalize.pkl"), vec)
    env.training = False
    env.norm_reward = False
    viewer = None
    frames: list[Image.Image] = []
    try:
        if args.watch:
            import mujoco
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(raw.model, raw.data)
            with viewer.lock():
                viewer.cam.lookat[:] = [0.19, 0.01, 0.10]
                viewer.cam.distance = 0.8
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -30
                if args.camera != "free":
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = raw.model.camera(args.camera).id
        model = PPO.load(str(args.run / "policy.zip"), device="cpu")
        results = []
        for episode in range(args.episodes):
            torch.manual_seed(args.seed + episode)
            if viewer is not None:
                with viewer.lock():
                    obs, _ = raw.reset(seed=args.seed + episode)
            else:
                obs, _ = raw.reset(seed=args.seed + episode)
            total_reward = 0.0
            observed_recoveries = observed_regrasps = 0
            for _ in range(raw.max_steps):
                started = time.monotonic()
                action, _ = model.predict(
                    env.normalize_obs(obs), deterministic=not args.stochastic
                )
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    with viewer.lock():
                        obs, reward, done, truncated, info = raw.step(action)
                    viewer.sync()
                    time.sleep(max(0, raw.dt - (time.monotonic() - started)))
                else:
                    obs, reward, done, truncated, info = raw.step(action)
                if args.watch:
                    if info["recoveries"] > observed_recoveries:
                        print(
                            f"{raw.data.time:.2f}s: retry ({info['last_recovery_reason']}), cube moves={info['relocation_count']}",
                            flush=True,
                        )
                        observed_recoveries = info["recoveries"]
                    if info["regrasp_count"] > observed_regrasps:
                        print(
                            f"{raw.data.time:.2f}s: regrasp #{info['regrasp_count']}",
                            flush=True,
                        )
                        observed_regrasps = info["regrasp_count"]
                total_reward += reward
                if args.gif and episode == 0 and raw.steps % 5 == 0:
                    frames.append(Image.fromarray(raw.render()))
                if done or truncated:
                    results.append(
                        {
                            "seed": args.seed + episode,
                            "reward": total_reward,
                            **info,
                        }
                    )
                    if args.watch:
                        print(
                            f"Trial {episode + 1}/{args.episodes}: "
                            f"{info['termination_reason']} "
                            f"(goal distance={info['goal_distance']:.3f}m)",
                            flush=True,
                        )
                        # Show the completed trial before resetting the scene.
                        time.sleep(0.5)
                    break
        if len(results) != args.episodes:
            raise RuntimeError("評価の試行が期限内に終了していません。")
        report = {
            "task_version": raw.task_version,
            **options,
            "failure_penalty": PickPlaceEnv.failure_penalty,
            "timeout_penalty": PickPlaceEnv.timeout_penalty,
            "stage": args.stage,
            "start_level": level if args.stage == "random" else 0,
            "domain_randomization": domain_randomization,
            "disturbances": disturbances,
            "deterministic": not args.stochastic,
            "full_task": args.stage in ("pick", "random"),
            "model_timesteps": model.num_timesteps,
            "seed": args.seed,
            "termination_counts": dict(
                Counter(r["termination_reason"] for r in results)
            ),
            "relocated_episodes": sum(r["relocation_count"] > 0 for r in results),
            "regrasp_episodes": sum(r["regrasp_count"] > 0 for r in results),
            "recovered_successes": sum(
                r["is_success"] and r["regrasp_count"] > 0 for r in results
            ),
            "episodes": len(results),
            "success_rate": float(np.mean([r["is_success"] for r in results])),
            "mean_reward": float(np.mean([r["reward"] for r in results])),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        suffix = "_stochastic" if args.stochastic else ""
        suffix += ("_dr" if domain_randomization else "") + (
            "_push" if disturbances else ""
        )
        if options["recovery"]:
            suffix += "_recovery"
        if options["cube_relocations"]:
            suffix += f"_move{options['relocation_count']}"
        if options["disturbance_strength"] != 1 or options["domain_scale"] != 1:
            suffix += (
                f"_p{options['disturbance_strength']:g}_d{options['domain_scale']:g}"
            )
        if options["workspace_scale"]:
            suffix += f"_ws{options['workspace_scale']:g}"
        report_path = args.run / (
            f"evaluation_{args.stage}_level{level if args.stage == 'random' else 0}{suffix}_steps{model.num_timesteps}"
            f"_seed{args.seed}_n{args.episodes}.json"
        )
        report_path.write_text(
            json.dumps({**report, "trials": results}, ensure_ascii=False, indent=2)
            + "\n"
        )
        print(f"評価結果: {report_path}", flush=True)
        if args.gif and frames:
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                args.gif,
                save_all=True,
                append_images=frames[1:],
                duration=round(PickPlaceEnv.dt * 5000),
                loop=0,
            )
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("正の整数にしてください。")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("正の有限値にしてください。")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="固定動作による物理把持の検証（学習済み方策ではない）"
    )
    demo.add_argument("--headless", action="store_true")
    demo.add_argument("--stage", choices=STAGES, default="release")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument(
        "--episodes", type=positive, default=1, help="シードを変えて連続実行"
    )
    demo.add_argument("--camera", choices=["free", "wrist", "overhead"], default="free")
    demo.add_argument("--snapshot", type=Path)
    demo.add_argument("--gif", type=Path)
    demo.set_defaults(func=run_demo)
    training = commands.add_parser("train", help="単一PPO方策の学習")
    training.add_argument(
        "--stage",
        choices=STAGES,
        help="新規学習の開始段階。再開時に指定すると段階を上書き",
    )
    training.add_argument(
        "--fixed-stage", action="store_true", help="自動難易度変更を無効化"
    )
    training.add_argument("--steps", type=positive, default=2_000_000)
    training.add_argument("--envs", type=positive, default=8)
    training.add_argument("--rollout", type=positive, default=256)
    training.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1e-4,
        help="学習率。再開時も適用（既定: 1e-4）",
    )
    training.add_argument(
        "--target-kl",
        type=positive_float,
        default=0.02,
        help="KLに基づく更新の早期停止。再開時も適用（既定: 0.02）",
    )
    training.add_argument(
        "--eval-every",
        type=positive,
        default=50_000,
        help="独立検証と再開可能な保存を行うステップ間隔",
    )
    training.add_argument(
        "--eval-episodes",
        type=positive,
        default=20,
        help="各開始段階・各推論方式の検証試行数",
    )
    training.add_argument(
        "--expand-starts",
        action="store_true",
        help="成功率と既習範囲の維持を確認して開始範囲を拡大",
    )
    training.add_argument("--start-level", type=int, choices=range(6), default=None)
    training.add_argument("--max-start-level", type=int, choices=range(6), default=None)
    training.add_argument("--entropy-coefficient", type=positive_float, default=None)
    training.add_argument(
        "--return-start-probability",
        type=float,
        default=0.25,
        help="帰還学習時、後半の退避・帰還から練習する割合（評価では常に0）",
    )
    training.add_argument(
        "--exploration-std",
        type=positive_float,
        default=None,
        help="学習開始時だけ探索の標準偏差を設定。通常の再開時は省略",
    )
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--output", type=Path, default=Path("outputs/pick_place_v5"))
    training.add_argument(
        "--resume", type=Path, help="policy.zipとnormalize.pklがある保存先"
    )
    for command in (training,):
        command.add_argument(
            "--domain-randomization",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="物体質量・接触摩擦・関節粘性・モーターゲインをランダム化",
        )
        command.add_argument(
            "--disturbances",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="机上と運搬中にキューブへの短い横押しを加える",
        )
    training.set_defaults(func=train)
    evaluation = commands.add_parser(
        "evaluate", help="保存した方策を未知のシードで評価"
    )
    evaluation.add_argument("--run", type=Path, default=Path("outputs/pick_place_v5"))
    evaluation.add_argument("--episodes", type=positive, default=20)
    evaluation.add_argument("--stage", choices=STAGES, default="random")
    evaluation.add_argument(
        "--start-level",
        type=int,
        choices=range(6),
        default=None,
        help="省略時は保存された学習範囲",
    )
    evaluation.add_argument("--seed", type=int, default=20000)
    evaluation.add_argument(
        "--camera", choices=["free", "wrist", "overhead"], default="free"
    )
    evaluation.add_argument(
        "--stochastic", action="store_true", help="方策から行動をサンプルして評価"
    )
    evaluation.add_argument("--watch", action="store_true", help="GUIで評価を表示")
    evaluation.add_argument("--gif", type=Path, help="最初の評価エピソードをGIF保存")
    evaluation.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="省略時は学習時の設定を使用",
    )
    evaluation.add_argument(
        "--disturbances",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="省略時は学習時の設定を使用",
    )
    for command in (training, evaluation):
        command.add_argument(
            "--return-home",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="十分に開いて配置後、上方退避し待機姿勢で0.3秒静止するまで継続",
        )
        command.add_argument(
            "--recovery",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="把持失敗後は再接近して掴み直す。期限24秒",
        )
        command.add_argument(
            "--cube-relocations",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="持ち上げ後にキューブを別の机上位置に移動",
        )
        command.add_argument(
            "--relocation-count", type=int, choices=(1, 2, 3), default=None
        )
        command.add_argument(
            "--disturbance-strength",
            type=positive_float,
            default=None,
            help="横押しの強さ（0より大きく5以下）",
        )
        command.add_argument(
            "--domain-scale",
            type=positive_float,
            default=None,
            help="物理特性のばらつき倍率（0より大きく3以下）",
        )
        command.add_argument(
            "--workspace-scale",
            type=float,
            default=None,
            help="手先の指令範囲の広さ（0=従来の半径19〜25cm、1=半径12〜29cm）",
        )
    evaluation.set_defaults(func=evaluate)
    args = parser.parse_args()
    if args.command == "train" and args.rollout * args.envs < 2:
        parser.error("PPOには rollout × envs が2以上必要です。")
    args.func(args)


if __name__ == "__main__":
    main()

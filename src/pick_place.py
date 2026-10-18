"""SO-101の単一方策用、状態観測の机上pick-and-place環境。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from task_logic import Measurement, Phase, TaskState

SCENE = Path(__file__).resolve().parent.parent / "assets/so101/pick_place.xml"
TASK_VERSION = "supported-place-stop-v5"
RECOVERY_TASK_VERSION = "recoverable-place-v6"
RECOVERY_REWARD_VERSION = "recovery-relocation-compensation-timeout-penalty-v3"
REWARD_VERSION = "approach-distance-open-timeout-penalty-v2"
RETURN_TASK_VERSION = "place-retreat-return-home-v7"
RETURN_REWARD_VERSION = "return-distance-cost-upward-release-safety-v5"
STAGES = ("release", "place", "transport", "lift", "grasp", "pick", "random")
GAMMA = 0.995
# Commanded TCP region (x/y min, x/y max, radius min/max) at workspace_scale 0 and 1.
# Top-down IK reaches radius 0.10-0.30 m at grasp height without table contact.
NARROW_COMMAND = np.array([[0.18, -0.04], [0.25, 0.115], [0.19, 0.25]])
WIDE_COMMAND = np.array([[0.10, -0.12], [0.29, 0.18], [0.12, 0.29]])
# Recovery ends the episode once the cube leaves what the hand can reach.
NARROW_CUBE_BOUNDS = np.array([[0.175, -0.045], [0.255, 0.12], [0.185, 0.255]])
WIDE_CUBE_BOUNDS = WIDE_COMMAND + np.array(
    [[-0.01, -0.01], [0.01, 0.01], [-0.01, 0.01]]
)
FloatArray = NDArray[np.float64]


def load_model() -> mujoco.MjModel:
    """配布元モデルは保持し、タスク用に指の凸包を箱の接触面に置き換える。"""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for name in ("gripper", "moving_jaw_so101_v1"):
        body = spec.body(name)
        for geom in body.geoms:
            if geom.group == 3:
                geom.contype = 0
                geom.conaffinity = 0
    # The original gripper's convex hull fills the space between its fingers.
    # Explicit palm/finger boxes keep that gap open for physical grasping.
    for body_name, name, pos, size in (
        ("gripper", "palm", [0, 0, -0.025], [0.025, 0.022, 0.018]),
        ("gripper", "fixed_pad", [-0.022, 0, -0.083], [0.006, 0.008, 0.019]),
        (
            "moving_jaw_so101_v1",
            "moving_pad",
            [-0.0042, -0.0596, 0.0188],
            [0.006, 0.019, 0.008],
        ),
    ):
        spec.body(body_name).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.15, 0.15, 0.15, 0.4],
            group=3,
            contype=1,
            conaffinity=1,
            friction=[1.5, 0.01, 0.001],
            condim=4,
            solref=[0.006, 1],
        )
    spec.body("gripper").add_site(
        name="grasp_center",
        pos=[0.001, 0, -0.083],
        size=[0.003] * 3,
        rgba=[0.2, 0.8, 1, 1],
        group=3,
    )
    return spec.compile()


def inverse_kinematics(
    model: mujoco.MjModel,
    target: FloatArray,
    initial: FloatArray,
) -> FloatArray:
    """初期姿勢の逆運動学。方策への教師データには使用しない。"""
    data = mujoco.MjData(model)
    data.qpos[:6] = initial
    site = model.site("grasp_center").id
    # Fingers point down, with the closing axis approximately along world -Y.
    desired = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    for _ in range(400):
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        orientation = (
            sum(
                (np.cross(rotation[:, i], desired[:, i]) for i in range(3)),
                start=np.zeros(3),
            )
            / 2
        )
        error = np.concatenate([target - data.site_xpos[site], 0.2 * orientation])
        if np.linalg.norm(error) < 1e-5:
            return data.qpos[:6].copy()
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack([jp[:, :5], 0.2 * jr[:, :5]])
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.0001 * np.eye(6), error)
        data.qpos[:5] = np.clip(
            data.qpos[:5] + np.clip(delta, -0.1, 0.1),
            model.jnt_range[:5, 0],
            model.jnt_range[:5, 1],
        )
    raise RuntimeError(f"IKが収束しませんでした: {target.tolist()}")


class PickPlaceEnv(gym.Env):
    """単一方策の4次元手先速度操作。全開始段階で同じ配置成功条件。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}
    dt = 0.04
    cube_half = 0.015
    max_steps = 400
    failure_penalty = 5.0
    # Heavier than failing so a discounted deadline is never a safe place to wait.
    timeout_penalty = 15.0
    tcp_speed = 0.08
    jaw_speed = 1.5
    torque_weight = 0.02

    def __init__(
        self,
        stage: str = "release",
        render_mode: str | None = None,
        camera: str = "free",
        start_level: int = 0,
        domain_randomization: bool = False,
        disturbances: bool = False,
        randomization_probability: float = 1.0,
        recovery: bool = False,
        cube_relocations: bool = False,
        relocation_count: int = 1,
        disturbance_strength: float = 1.0,
        domain_scale: float = 1.0,
        workspace_scale: float = 0.0,
        return_home: bool = False,
        return_start_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if stage not in STAGES:
            raise ValueError(f"stage は {STAGES} から選んでください。")
        if render_mode not in (None, "rgb_array") or camera not in (
            "free",
            "wrist",
            "overhead",
        ):
            raise ValueError("render_mode または camera が不正です。")
        self.render_mode, self.camera = render_mode, camera
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = round(self.dt / self.model.opt.timestep)
        self.cube_body = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube").id
        self.table_geom = self.model.geom("table").id
        self.robot_geoms = {
            i
            for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) not in (0, self.cube_body)
        }
        joint = self.model.joint("cube_joint").id
        self.cube_qpos = int(self.model.jnt_qposadr[joint])
        self.cube_dof = int(self.model.jnt_dofadr[joint])
        self.tip_site = self.model.site("grasp_center").id
        self.target_site = self.model.site("target").id
        self.pad_ids = {self.model.geom(n).id for n in ("fixed_pad", "moving_pad")}
        if not np.isfinite(disturbance_strength) or not 0 < disturbance_strength <= 5:
            raise ValueError("disturbance_strength は0より大きく5以下にしてください。")
        if not np.isfinite(domain_scale) or not 0 < domain_scale <= 3:
            raise ValueError("domain_scale は0より大きく3以下にしてください。")
        if not np.isfinite(workspace_scale) or not 0 <= workspace_scale <= 1:
            raise ValueError("workspace_scale は0以上1以下にしてください。")
        if cube_relocations and not recovery:
            raise ValueError("cube_relocations には recovery が必要です。")
        if relocation_count not in (1, 2, 3):
            raise ValueError("relocation_count は1〜3にしてください。")
        self.recovery = recovery
        self.return_home = return_home
        if not 0 <= return_start_probability <= 1 or (
            return_start_probability and not return_home
        ):
            raise ValueError("帰還の練習開始率は0〜1で、return_homeが必要です。")
        self.return_start_probability = return_start_probability
        self.task_version = (
            RETURN_TASK_VERSION
            if return_home
            else RECOVERY_TASK_VERSION
            if recovery
            else TASK_VERSION
        )
        self.max_steps = (600 if recovery else 400) + (150 if return_home else 0)
        self.cube_relocations = cube_relocations
        self.relocation_limit = relocation_count
        self.disturbance_strength = disturbance_strength
        self.domain_scale = domain_scale
        self.workspace_scale = workspace_scale
        self.command_bounds = NARROW_COMMAND + workspace_scale * (
            WIDE_COMMAND - NARROW_COMMAND
        )
        self.cube_bounds = NARROW_CUBE_BOUNDS + workspace_scale * (
            WIDE_CUBE_BOUNDS - NARROW_CUBE_BOUNDS
        )
        self.domain_randomization = domain_randomization
        self.disturbances = disturbances
        if (
            not np.isfinite(randomization_probability)
            or not 0 <= randomization_probability <= 1
        ):
            raise ValueError("randomization_probability は0〜1にしてください。")
        self.randomization_probability = randomization_probability
        self.domain_rng = np.random.default_rng(0)
        self.nominal_physics = {
            name: getattr(self.model, name).copy()
            for name in (
                "body_mass",
                "body_inertia",
                "geom_friction",
                "dof_damping",
                "actuator_gainprm",
                "actuator_biasprm",
            )
        }
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.set_start_level(start_level)
        self.start_level = start_level
        self.pending_stage = self.stage = stage
        self.curriculum_enabled = False
        self.rehearsal_enabled = False
        self.renderer: mujoco.Renderer | None = None
        self.home_tip = np.array([0.23, 0, 0.09])
        self.home = inverse_kinematics(self.model, self.home_tip, np.zeros(6))
        self.home[5] = 0.8
        self.retreat_anchor = self.home_tip.copy()
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros_like(self.jp)
        self.last_action = np.zeros(4)
        self.filtered_action = np.zeros(4)
        self.goal = np.array([0.20, 0.08, self.cube_half])
        self.state = TaskState()
        self.steps = 0
        self.reset(seed=0)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, self._observation().shape, dtype=np.float32
        )

    def set_stage(
        self, stage: str, curriculum: bool = False, rehearsal: bool = False
    ) -> None:
        if stage not in STAGES:
            raise ValueError(stage)
        self.pending_stage, self.curriculum_enabled = stage, curriculum
        self.rehearsal_enabled = rehearsal

    def set_start_level(self, level: int, rehearsal: bool = False) -> None:
        """次のresetから開始分布を変更する。0は従来の分布。"""
        if not isinstance(level, int) or not 0 <= level <= 5:
            raise ValueError("start_level は0〜5の整数にしてください。")
        self.pending_start_level = level
        self.start_rehearsal = rehearsal

    def _randomize_physics(self, seed: int | None) -> None:
        # Use an independent RNG: nominal/perturbed evaluations with the same
        # seed must still start at exactly the same sampled positions.
        if seed is not None:
            self.domain_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 0xD04A1])
            )
        rng = self.domain_rng
        self.randomization_active = bool(
            (self.domain_randomization or self.disturbances or self.cube_relocations)
            and rng.random() < self.randomization_probability
        )
        for name, values in self.nominal_physics.items():
            getattr(self.model, name)[:] = values
        self.physics_parameters: dict[str, Any] = {}
        if self.randomization_active and self.domain_randomization:
            scale = self.domain_scale
            mass = float(rng.uniform(1 - 0.2 * scale, 1 + 0.2 * scale))
            self.model.body_mass[self.cube_body] *= mass
            self.model.body_inertia[self.cube_body] *= mass
            contact_geoms = [self.table_geom, self.cube_geom, *sorted(self.pad_ids)]
            friction = rng.uniform(1 - 0.2 * scale, 1 + 0.2 * scale, len(contact_geoms))
            self.model.geom_friction[contact_geoms] *= friction[:, None]
            damping = rng.uniform(1 - 0.15 * scale, 1 + 0.15 * scale, 6)
            self.model.dof_damping[:6] *= damping
            gain = rng.uniform(1 - 0.1 * scale, 1 + 0.1 * scale, self.model.nu)
            self.model.actuator_gainprm[:] *= gain[:, None]
            self.model.actuator_biasprm[:] *= gain[:, None]
            self.physics_parameters = dict(
                mass_scale=mass,
                friction_scales=friction.tolist(),
                damping_scales=damping.tolist(),
                gain_scales=gain.tolist(),
            )
        mujoco.mj_setConst(self.model, self.data)
        self.pushes: list[dict[str, Any]] = []
        if self.randomization_active and self.disturbances:
            for phase, delay, duration, acceleration in (
                (
                    Phase.REACH,
                    rng.uniform(0.2, 0.4),
                    rng.uniform(0.06, 0.10),
                    (1 + rng.uniform(0.1, 0.3) * self.disturbance_strength) * 9.81,
                ),
                (
                    Phase.TRANSPORT,
                    rng.uniform(0.05, 0.2),
                    rng.uniform(0.08, 0.12),
                    rng.uniform(0.8, 1.5) * self.disturbance_strength,
                ),
            ):
                angle = rng.uniform(-np.pi, np.pi)
                force = (
                    np.r_[np.cos(angle), np.sin(angle), 0]
                    * acceleration
                    * self.model.body_mass[self.cube_body]
                )
                self.pushes.append(
                    dict(
                        phase=phase,
                        delay=float(delay),
                        duration=float(duration),
                        force=force,
                        start=None,
                        applied=False,
                    )
                )
        self.relocation_events: list[dict[str, Any]] = []
        self._step_relocation_compensation = 0.0
        self.episode_relocation_compensation = 0.0
        self.last_relocated_grasp = -1
        self.disturbance_count = 0
        self.disturbance_impulse = 0.0

    def _apply_disturbance(self) -> None:
        force = np.zeros(3)
        for push in self.pushes:
            if self.state.phase != push["phase"]:
                continue
            if push["start"] is None:
                push["start"] = float(self.data.time) + push["delay"]
            if push["start"] <= self.data.time < push["start"] + push["duration"]:
                if self.state.phase == Phase.REACH:
                    m = self._measurement()
                    if m.robot_contact or not m.supported:
                        continue
                force += push["force"]
                if not push["applied"]:
                    self.disturbance_count += 1
                    push["applied"] = True
        # No pushes during release/settle, when the policy no longer acts.
        self.data.xfrc_applied[self.cube_body, :3] = force
        self.disturbance_impulse += (
            float(np.linalg.norm(force)) * self.model.opt.timestep
        )

    def _maybe_relocate_cube(self) -> None:
        if not (
            self.cube_relocations
            and not self.state.reason
            and self.randomization_active
            and self.state.phase == Phase.TRANSPORT
            and len(self.relocation_events) < self.relocation_limit
            and self.state.grasp_count != self.last_relocated_grasp
            and self._grasped()
        ):
            return
        potential_before = self._potential()
        original = self.cube.copy()
        saved = self.data.qpos[self.cube_qpos : self.cube_qpos + 7].copy()
        for _ in range(1000):
            xy = self.domain_rng.uniform([0.20, -0.02], [0.24, 0.04])
            if (
                np.linalg.norm(xy) > 0.245
                or np.linalg.norm(xy - original[:2]) < 0.02
                or np.linalg.norm(xy - self.goal[:2]) < 0.035
            ):
                continue
            self.data.qpos[self.cube_qpos : self.cube_qpos + 3] = [*xy, self.cube_half]
            mujoco.mj_forward(self.model, self.data)
            overlap = any(
                (
                    (c.geom1 == self.cube_geom and c.geom2 in self.robot_geoms)
                    or (c.geom2 == self.cube_geom and c.geom1 in self.robot_geoms)
                )
                and c.dist < 0.001
                for c in self.data.contact
            )
            if not overlap:
                break
        else:
            self.data.qpos[self.cube_qpos : self.cube_qpos + 7] = saved
            mujoco.mj_forward(self.model, self.data)
            raise RuntimeError("衝突のないキューブの移動先を選べませんでした。")
        self.data.qvel[self.cube_dof : self.cube_dof + 6] = 0
        self.data.qacc_warmstart[:] = 0
        self.data.xfrc_applied[self.cube_body] = 0
        mujoco.mj_forward(self.model, self.data)
        self.last_relocated_grasp = self.state.grasp_count
        self.relocation_events.append(
            dict(
                time=float(self.data.time),
                before=original.tolist(),
                after=self.cube.tolist(),
                grasp_count=self.state.grasp_count,
            )
        )
        self.state.recover("cube_relocated")
        self.release_state = None
        # Externally requested relocations are finite, mandatory challenges.
        # Remove their unavoidable potential jump; natural drops keep the loss.
        compensation = GAMMA * (potential_before - self._potential())
        self._step_relocation_compensation += compensation
        self.episode_relocation_compensation += compensation

    def _start_target(self, source: FloatArray) -> FloatArray:
        # Sample within the controller workspace rather than clipping samples
        # onto its boundary. Independent hand offsets prevent 'always descend'.
        for _ in range(1000):
            offset = self.np_random.uniform(-1, 1, 3)
            target = (
                np.r_[source, 0.065]
                + offset * np.array([0.004, 0.004, 0.003]) * self.start_level
            )
            if (
                0.18 <= target[0] <= 0.245
                and -0.035 <= target[1] <= 0.105
                and 0.195 <= np.linalg.norm(target[:2]) <= 0.248
            ):
                return target
        raise RuntimeError("到達可能な手先の開始位置をサンプルできませんでした。")

    @property
    def cube(self) -> FloatArray:
        return self.data.xpos[self.cube_body].copy()

    @property
    def tip(self) -> FloatArray:
        return self.data.site_xpos[self.tip_site].copy()

    def _measurement(self) -> Measurement:
        forces: dict[int, float] = {}
        f = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            other = b if a == self.cube_geom else a if b == self.cube_geom else -1
            if other >= 0:
                mujoco.mj_contactForce(self.model, self.data, i, f)
                forces[other] = forces.get(other, 0.0) + max(0.0, float(f[0]))
        cube = self.cube
        velocity = self.data.qvel[self.cube_dof : self.cube_dof + 6]
        tipvel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tip_site, tipvel, 0
        )
        return Measurement(
            grasped=all(forces.get(p, 0) > 0.01 for p in self.pad_ids),
            supported=forces.get(self.table_geom, 0)
            > 0.1
            * self.model.body_mass[self.cube_body]
            * np.linalg.norm(self.model.opt.gravity),
            robot_contact=any(forces.get(p, 0) > 0.01 for p in self.robot_geoms),
            opened=bool(self.data.qpos[5] >= (0.65 if self.return_home else 0.25)),
            height=float(cube[2]),
            goal_xy=float(np.linalg.norm(cube[:2] - self.goal[:2])),
            goal_distance=float(np.linalg.norm(cube - self.goal)),
            clearance=float(self.tip[2] - cube[2]),
            linear_speed=float(np.linalg.norm(velocity[:3])),
            angular_speed=float(np.linalg.norm(velocity[3:])),
            tip_speed=float(np.linalg.norm(tipvel[3:])),
            vertical_speed=float(velocity[2]),
            tip_height=float(self.tip[2]),
            retreat_xy=float(np.linalg.norm(self.tip[:2] - self.retreat_anchor[:2])),
            home_distance=float(np.linalg.norm(self.tip - self.home_tip)),
            home_joint_error=float(np.max(np.abs(self.data.qpos[:5] - self.home[:5]))),
            joint_speed=float(np.max(np.abs(self.data.qvel[:6]))),
            outside=bool(
                cube[2] < -0.04
                or (
                    not (
                        (self.cube_bounds[0] < cube[:2]).all()
                        and (cube[:2] < self.cube_bounds[1]).all()
                        and self.cube_bounds[2, 0]
                        < np.linalg.norm(cube[:2])
                        < self.cube_bounds[2, 1]
                    )
                    if self.recovery
                    else not (0.10 < cube[0] < 0.36 and abs(cube[1]) < 0.20)
                )
            ),
        )

    def _grasped(self) -> bool:
        return self._measurement().grasped

    def _observation(self) -> NDArray[np.float32]:
        m = self._measurement()
        return np.concatenate(
            [
                self.data.qpos[:6],
                self.data.qvel[:6],
                self.data.ctrl,
                self.cube,
                self.data.qpos[self.cube_qpos + 3 : self.cube_qpos + 7],
                self.data.qvel[self.cube_dof : self.cube_dof + 6],
                self.tip,
                self.goal,
                self.cube - self.tip,
                self.goal - self.cube,
                self.state.observation(),
                [m.grasped, m.supported, m.robot_contact],
                self.filtered_action,
                self.last_action,
                # Preserve the old pick policy's time feature through transfer.
                [self.data.time / ((600 if self.recovery else 400) * self.dt)],
                *(
                    [
                        [
                            self.state.phase == Phase.RETREAT,
                            self.state.phase == Phase.RETURN,
                            self.state.home_timer,
                        ],
                        self.home_tip - self.tip,
                        self.retreat_anchor[:2] - self.tip[:2],
                    ]
                    if self.return_home
                    else []
                ),
            ]
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.stage = self.pending_stage
        self.start_level = self.pending_start_level if self.stage == "random" else 0
        if self.start_rehearsal and self.start_level and self.np_random.random() < 0.25:
            self.start_level -= 1
        if (
            self.rehearsal_enabled
            and self.stage != STAGES[0]
            and self.np_random.random() < 0.25
        ):
            self.stage = STAGES[STAGES.index(self.stage) - 1]
        self.eased_start = (
            self.curriculum_enabled
            and self.stage not in ("release", "random")
            and self.np_random.random() < 0.5
        )
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_physics(seed)
        source = np.array([0.23, 0.0])
        self.goal = np.array([0.20, 0.08, self.cube_half])
        if self.stage == "random":
            source += self.np_random.uniform(
                [-0.015 - 0.004 * self.start_level, -0.015 - 0.004 * self.start_level],
                [0.015, 0.015 + 0.004 * self.start_level],
            )
            self.goal[:2] += self.np_random.uniform(-0.012, 0.012, 2)
        held = self.stage in ("release", "place", "transport", "lift")
        xy = self.goal[:2].copy() if self.stage in ("release", "place") else source
        if self.eased_start and self.stage == "transport":
            # Bridge the gap to a new phase with successful near-goal starts.
            # Validation always uses the full source-to-goal displacement.
            fraction = self.np_random.random() ** 2
            xy = self.goal[:2] + fraction * (source - self.goal[:2])
        height = (
            0.0148
            if self.stage in ("release", "lift")
            else float(self.np_random.uniform(0.027, 0.040))
            if self.stage == "place"
            else 0.07
        )
        if self.eased_start and self.stage == "lift":
            height = 0.0148 + 0.0552 * np.sqrt(self.np_random.random())
        if self.eased_start and self.stage == "place":
            height = 0.0148 + 0.0252 * self.np_random.random() ** 2
        target = (
            np.r_[xy, height + 0.011]
            if held
            else np.r_[source, 0.026 if self.stage == "grasp" else 0.065]
        )
        if self.eased_start and self.stage in ("pick", "random"):
            target[2] = self.np_random.uniform(0.027, 0.065)
        if self.stage == "random" and self.start_level:
            target = self._start_target(source)
        self.recovery_start = bool(
            self.recovery
            and self.curriculum_enabled
            and self.randomization_active
            and self.stage in ("pick", "random")
            and self.domain_rng.random() < 0.3
        )
        q = inverse_kinematics(self.model, target, self.home)
        q[5] = 0.04 if held else 0.8
        if self.recovery_start:
            q[5] = self.domain_rng.uniform(0.0, 0.5)
        if self.eased_start and self.stage == "grasp":
            q[5] = 0.05 + 0.75 * self.np_random.random() ** 2
        self.data.qpos[:6] = q
        self.data.ctrl[:] = q
        self.data.ctrl[5] = -0.08 if held else q[5]
        self.data.qpos[self.cube_qpos : self.cube_qpos + 3] = (
            np.r_[xy + [0, 0.003], height] if held else np.r_[xy, self.cube_half]
        )
        if self.stage == "random":
            yaw = self.np_random.uniform(-0.25, 0.25)
            self.data.qpos[self.cube_qpos + 3 : self.cube_qpos + 7] = [
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
        self.model.site_pos[self.target_site] = [*self.goal[:2], 0.001]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if held and not self._grasped():
            raise RuntimeError("初期状態で物理把持が成立していません。")
        if held:
            # Keep a holding preload without winding the jaw target far behind
            # its measured contact position. This is independent of task phase.
            self.data.ctrl[5] = self.data.qpos[5] - 0.01
            mujoco.mj_forward(self.model, self.data)
        self.data.time = 0
        self.steps = 0
        self.episode_torque_cost = 0.0
        self.release_state: Measurement | None = None
        self.peak_cube_height = float(self.cube[2])
        self._episode_done = False
        self.last_action[:] = self.filtered_action[:] = 0
        self.state = TaskState(
            Phase.LOWER
            if self.stage in ("release", "place")
            else Phase.LIFT
            if self.stage == "lift"
            else Phase.TRANSPORT
            if held
            else Phase.REACH,
            allow_recovery=self.recovery,
            grasp_count=int(held),
            return_home=self.return_home,
        )
        self.retreat_anchor = self.tip.copy()
        self.return_start = bool(
            self.return_home and self.np_random.random() < self.return_start_probability
        )
        if self.return_start:
            self._reset_return_start()
        self.last_command_target = self.data.ctrl.copy()
        self.previous_potential = self._potential()
        return self._observation(), self._info()

    def _reset_return_start(self) -> None:
        """訓練だけの後半開始。評価とACT収集には使用しない。"""
        retreat = self.np_random.random() < 0.5
        target = self.goal.copy()
        target[2] = self.np_random.uniform(0.026, 0.07) if retreat else 0.085
        target[1] += 0.004
        if not retreat:
            # Include near-home holds so the final stationary success is
            # experienced before trying the whole return displacement.
            mix = (
                self.np_random.uniform(0.9, 1.0)
                if self.np_random.random() < 0.5
                else self.np_random.uniform(0, 0.8)
            )
            target = (1 - mix) * target + mix * self.home_tip
        self.data.qpos[:6] = inverse_kinematics(self.model, target, self.home)
        self.data.qpos[5] = 0.8
        self.data.qpos[self.cube_qpos : self.cube_qpos + 3] = self.goal
        self.data.qpos[self.cube_qpos + 3 : self.cube_qpos + 7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.state = TaskState(
            phase=Phase.RETREAT if retreat else Phase.RETURN,
            allow_recovery=self.recovery,
            grasp_count=1,
            return_home=True,
        )
        self.retreat_anchor = self.tip.copy()
        self.release_state = self._measurement()
        self.eased_start = True

    def _control(self, action: FloatArray) -> None:
        self.filtered_action = 0.5 * self.filtered_action + 0.5 * action
        desired = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
        rot = self.data.site_xmat[self.tip_site].reshape(3, 3)
        orientation = (
            sum(
                (np.cross(rot[:, i], desired[:, i]) for i in range(3)),
                start=np.zeros(3),
            )
            / 2
        )
        mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.tip_site)
        jac = np.vstack([self.jp[:, :5], 0.2 * self.jr[:, :5]])
        velocity = self.filtered_action[:3] * self.tcp_speed
        # Workspace bounds constrain commanded velocity, not task behavior.
        low, high, radii = self.command_bounds
        target = np.clip(self.tip + velocity * self.dt, [*low, 0.023], [*high, 0.09])
        radius = np.linalg.norm(target[:2])
        target[:2] *= np.clip(radius, *radii) / radius
        error = np.r_[target - self.tip, 0.2 * orientation]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
        self.data.ctrl[:5] = np.clip(
            self.data.qpos[:5] + np.clip(delta, -self.dt, self.dt),
            self.model.actuator_ctrlrange[:5, 0],
            self.model.actuator_ctrlrange[:5, 1],
        )
        self.data.ctrl[5] = np.clip(
            self.data.ctrl[5] + self.jaw_speed * self.dt * self.filtered_action[3],
            max(-0.12, float(self.data.qpos[5]) - 0.06),
            min(1.0, float(self.data.qpos[5]) + 0.06),
        )

    def _potential(self) -> float:
        m = self._measurement()
        phase = self.state.phase
        if phase == Phase.REACH:
            score = 1 - np.tanh(
                np.linalg.norm(self.tip - (self.cube + [0, 0, 0.011])) / 0.05
            )
            if self.recovery:
                alignment = float(
                    np.exp(
                        -np.square(
                            np.linalg.norm(self.tip - (self.cube + [0, 0, 0.011]))
                            / 0.012
                        )
                    )
                )
                opening = float(np.clip(self.data.qpos[5] / 0.65, 0, 1))
                score = (
                    0.65 * score
                    + 0.2 * float(m.grasped)
                    + 0.15 * ((1 - alignment) * opening + alignment * float(m.grasped))
                )
            else:
                score = 0.8 * score + 0.2 * float(m.grasped)
        elif phase == Phase.LIFT:
            score = np.clip((m.height - self.cube_half) / 0.05, 0, 1)
        elif phase == Phase.TRANSPORT:
            score = (1 - np.tanh(m.goal_xy / 0.08)) * np.clip(
                (m.height - 0.025) / 0.03, 0, 1
            )
        elif phase == Phase.LOWER:
            score = 0.8 * (1 - np.tanh(m.goal_distance / 0.04)) + 0.2 * float(
                m.supported
            )
        elif phase == Phase.RELEASE:
            placement = 1 - np.tanh(m.goal_distance / 0.04)
            opening = np.clip(self.data.qpos[5] / 0.65, 0, 1)
            score = placement * (0.5 + 0.5 * opening * float(m.supported))
        elif phase == Phase.SETTLE:
            score = np.clip(self.state.timers["settle"] / 0.4, 0, 1)
        elif phase == Phase.RETREAT:
            height = np.clip((self.tip[2] - 0.023) / (0.075 - 0.023), 0, 1)
            vertical = np.exp(-np.square(m.retreat_xy / 0.025))
            score = height * vertical * float(m.opened)
        else:
            distance = 1 - np.tanh(m.home_distance / 0.08)
            joints = 1 - np.tanh(m.home_joint_error / 0.4)
            score = (
                0.7 * distance
                + 0.2 * joints
                + 0.1 * np.clip(self.state.home_timer / 0.3, 0, 1)
            )
        return float(phase + score)

    def _protect_release_target(self, previous: FloatArray) -> None:
        """解放後の動作範囲。軌道や上昇速度は方策が決める。"""
        if not self.return_home or self.state.phase < Phase.RETREAT:
            return
        mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.tip_site)
        delta = self.data.ctrl[:5] - self.data.qpos[:5]
        if self.state.phase == Phase.RETREAT:
            # One upward direction preserves XY and tool orientation. Integrate
            # from the previous motor target so a zero command holds preload
            # instead of repeatedly removing gravity compensation.
            constraints = np.vstack([self.jp[:2, :5], 0.2 * self.jr[:, :5]])
            _, _, vectors = np.linalg.svd(constraints)
            direction = vectors[-1]
            vertical_gain = float(self.jp[2, :5] @ direction)
            if vertical_gain < 0:
                direction = -direction
                vertical_gain = -vertical_gain
            requested = float(self.jp[2, :5] @ (self.data.ctrl[:5] - previous[:5]))
            amount = max(0.0, requested) / max(vertical_gain, 1e-9)
            low = np.maximum(
                self.data.qpos[:5] - self.dt, self.model.actuator_ctrlrange[:5, 0]
            )
            high = np.minimum(
                self.data.qpos[:5] + self.dt, self.model.actuator_ctrlrange[:5, 1]
            )
            for joint, component in enumerate(direction):
                if abs(component) > 1e-9:
                    limit = (
                        high[joint] - previous[joint]
                        if component > 0
                        else low[joint] - previous[joint]
                    ) / component
                    amount = min(amount, max(0.0, float(limit)))
            self.data.ctrl[:5] = previous[:5] + direction * amount
        else:
            vertical = self.jp[2, :5]
            dz = float(vertical @ delta)
            minimum = 0.075 - float(self.tip[2])
            if dz < minimum - 1e-9:
                delta += (
                    vertical * (minimum - dz) / max(float(vertical @ vertical), 1e-9)
                )
            delta *= min(1.0, self.dt / max(float(np.max(np.abs(delta))), 1e-9))
            self.data.ctrl[:5] = np.clip(
                self.data.qpos[:5] + delta,
                self.model.actuator_ctrlrange[:5, 0],
                self.model.actuator_ctrlrange[:5, 1],
            )
        # There is no more grasping in this episode after a valid placement.
        self.data.ctrl[5] = max(float(self.data.ctrl[5]), float(previous[5]))

    def _approach_cost(self, action: FloatArray) -> float:
        if self.state.phase != Phase.REACH:
            return 0.0
        distance = float(np.linalg.norm(self.tip - (self.cube + [0, 0, 0.011])))
        far = float(np.clip((distance - 0.01) / 0.02, 0, 1))
        # State cost gives a learning signal even before a successful grasp.
        # Closing far from the object is optional, but pays an explicit cost.
        return self.dt * (
            0.2 * np.tanh(distance / 0.05) + 0.05 * far * max(0.0, -float(action[3]))
        )

    def _completion_cost(self) -> float:
        if not self.return_home or self.state.phase < Phase.RETREAT:
            return 0.0
        target = (
            np.r_[self.retreat_anchor[:2], 0.085]
            if self.state.phase == Phase.RETREAT
            else self.home_tip
        )
        distance = float(np.linalg.norm(self.tip - target))
        opening = float(np.clip(self.data.qpos[5] / 0.8, 0, 1))
        # Unlike tanh potential, this state cost does not flatten far from home.
        # Waiting at a workspace boundary is consistently worse than returning.
        return self.dt * (2 * min(distance / 0.1, 3) + 0.5 * (1 - opening))

    def _info(self) -> dict[str, Any]:
        m = self._measurement()
        return dict(
            is_success=self.state.reason == "success",
            task_version=self.task_version,
            recovery=self.recovery,
            return_home=self.return_home,
            return_start=self.return_start,
            gripper_opened=m.opened,
            jaw_position=float(self.data.qpos[5]),
            home_distance=m.home_distance,
            home_joint_error=m.home_joint_error,
            home_timer=self.state.home_timer,
            home_verifying=self.state.home_verifying,
            recovery_start=self.recovery_start,
            recoveries=self.state.recoveries,
            last_recovery_reason=self.state.last_recovery_reason,
            grasp_count=self.state.grasp_count,
            regrasp_count=max(0, self.state.grasp_count - 1),
            relocation_count=len(self.relocation_events),
            relocations_enabled=self.cube_relocations and self.randomization_active,
            relocation_compensation=self._step_relocation_compensation,
            episode_relocation_compensation=self.episode_relocation_compensation,
            relocation_events=deepcopy(self.relocation_events),
            disturbance_strength=self.disturbance_strength,
            domain_scale=self.domain_scale,
            workspace_scale=self.workspace_scale,
            stage=self.stage,
            start_level=self.start_level,
            domain_randomization=self.domain_randomization
            and self.randomization_active,
            disturbances=self.disturbances and self.randomization_active,
            physics_parameters=deepcopy(self.physics_parameters),
            cube_mass=float(self.model.body_mass[self.cube_body]),
            disturbance_count=self.disturbance_count,
            disturbance_impulse=self.disturbance_impulse,
            phase=self.state.phase.name.lower(),
            grasped=m.grasped,
            robot_contact=m.robot_contact,
            goal_distance=m.goal_distance,
            termination_reason=self.state.reason,
            full_task=self.stage in ("pick", "random") and not self.return_start,
            episode_torque_cost=self.episode_torque_cost,
            curriculum_start=self.curriculum_enabled,
            elapsed_seconds=float(self.data.time),
            eased_start=self.eased_start,
            peak_cube_height=self.peak_cube_height,
            release_height=self.release_state.height if self.release_state else None,
            release_table_supported=bool(self.release_state.supported)
            if self.release_state
            else None,
            release_goal_distance=self.release_state.goal_distance
            if self.release_state
            else None,
            cube_speed=m.linear_speed,
            cube_angular_speed=m.angular_speed,
        )

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return self._step(action)

    def step_joint_targets(
        self, target: FloatArray
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """画像方策の6関節目標。PPOと同じ補間・速度制限・物理・終了判定。"""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("target は有限値の6関節目標にしてください。")
        target = np.clip(
            target,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        target[:5] = np.clip(
            target[:5], self.data.qpos[:5] - self.dt, self.data.qpos[:5] + self.dt
        )
        target[5] = np.clip(
            target[5],
            max(-0.12, float(self.data.qpos[5]) - 0.06),
            min(1.0, float(self.data.qpos[5]) + 0.06),
        )
        return self._step(np.zeros(4, dtype=np.float32), target)

    def _step(
        self, action: NDArray[np.float32], joint_target: FloatArray | None = None
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("終了した試行には操作できません。reset()してください。")
        self._step_relocation_compensation = 0.0
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (4,) or not np.isfinite(a).all():
            raise ValueError("action は有限値の4次元配列にしてください。")
        a = np.clip(a, -1, 1)
        approach_cost = self._approach_cost(a)
        control_start = self.data.ctrl.copy()
        if joint_target is None:
            self._control(a)
        else:
            self.data.ctrl[:] = joint_target
        self._protect_release_target(control_start)
        control_end = self.data.ctrl.copy()
        self.last_command_target = control_end.copy()
        torque_integral = 0.0
        force_limits = np.maximum(
            np.abs(self.model.actuator_forcerange).max(axis=1), 1e-6
        )
        substep = 0
        started = self.data.time
        # Verify placement and final home rest with the last motor command held.
        # Retreat and return motion themselves always come from policy actions.
        while (
            substep < self.frame_skip
            or self.state.phase == Phase.SETTLE
            or self.state.home_verifying
        ):
            if self.state.phase != Phase.SETTLE and not self.state.home_verifying:
                self.data.ctrl[:] = control_start + (control_end - control_start) * (
                    (substep + 1) / self.frame_skip
                )
            self._apply_disturbance()
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            torque_integral += (
                float(np.square(self.data.actuator_force / force_limits).mean())
                * self.model.opt.timestep
            )
            measurement = self._measurement()
            old_phase = self.state.phase
            recoveries_before = self.state.recoveries
            self.state.tick(measurement, self.model.opt.timestep)
            if self.state.recoveries != recoveries_before:
                self.release_state = None
            self._maybe_relocate_cube()
            self.peak_cube_height = max(self.peak_cube_height, measurement.height)
            if old_phase != Phase.SETTLE and self.state.phase == Phase.SETTLE:
                self.release_state = measurement
                self.retreat_anchor = self.tip.copy()
            if (
                self.data.time + 1e-9 >= self.max_steps * self.dt
                and not self.state.reason
            ):
                self.state.reason = "timeout"
            if self.state.reason:
                break
            substep += 1
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or any(w.number for w in self.data.warning)
        ):
            raise RuntimeError("MuJoCoの物理計算が不安定になりました。")
        self.steps += 1
        terminated = bool(self.state.reason)
        truncated = False
        potential = 0.0 if terminated else self._potential()
        shaping = (
            GAMMA * potential
            - self.previous_potential
            + self._step_relocation_compensation
        )
        torque_cost = self.torque_weight * torque_integral
        completion_cost = self._completion_cost()
        self.episode_torque_cost += torque_cost
        reward = (
            shaping
            - 0.02 * (self.data.time - started)
            - torque_cost
            - approach_cost
            - completion_cost
            - 0.0002 * float(np.square(a - self.last_action).mean())
        )
        if self.state.reason == "success":
            reward += 10
        elif self.state.reason == "timeout":
            reward -= self.timeout_penalty
        elif terminated:
            reward -= self.failure_penalty
        self.previous_potential = potential
        self.last_action = a.copy()
        self._episode_done = terminated or truncated
        info = self._info()
        info.update(
            torque_cost=torque_cost,
            shaping=shaping,
            approach_cost=approach_cost,
            completion_cost=completion_cost,
        )
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> NDArray[np.uint8]:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera: str | mujoco.MjvCamera = self.camera
        if self.camera == "free":
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.19, 0.01, 0.10]
            camera.distance = 0.8
            camera.azimuth = 135
            camera.elevation = -30
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

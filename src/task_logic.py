"""物理エンジンから独立した配置タスクの契約。時間は秒、距離はm。"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Phase(IntEnum):
    REACH = 0
    LIFT = 1
    TRANSPORT = 2
    LOWER = 3
    RELEASE = 4
    SETTLE = 5
    RETREAT = 6
    RETURN = 7


@dataclass(frozen=True)
class Measurement:
    grasped: bool
    supported: bool
    robot_contact: bool
    opened: bool
    height: float
    goal_xy: float
    goal_distance: float
    clearance: float
    linear_speed: float
    angular_speed: float
    tip_speed: float
    vertical_speed: float
    outside: bool = False
    tip_height: float = 0.0
    retreat_xy: float = float("inf")
    home_distance: float = float("inf")
    home_joint_error: float = float("inf")
    joint_speed: float = float("inf")


@dataclass
class TaskState:
    phase: Phase = Phase.REACH
    # Persistent, observed timers; never reset at action boundaries.
    timers: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(
            ("grasp", "lift", "target", "support", "clear", "fall", "settle"), 0.0
        )
    )
    reason: str = ""
    unsafe_release: bool = False
    allow_recovery: bool = False
    recoveries: int = 0
    grasp_count: int = 0
    last_recovery_reason: str = ""
    return_home: bool = False
    home_timer: float = 0.0
    home_verifying: bool = False

    def recover(self, reason: str) -> None:
        self.phase = Phase.REACH
        self.reason = ""
        self.unsafe_release = False
        self.timers = dict.fromkeys(self.timers, 0.0)
        self.recoveries += 1
        self.last_recovery_reason = reason

    def tick(self, m: Measurement, dt: float) -> None:
        if self.reason:
            return
        if self.return_home and self.phase >= Phase.RETREAT:
            # Placing is irreversible: never reward reaching for the cube again.
            if m.outside:
                self.reason = "outside_workspace"
            elif m.robot_contact:
                self.reason = "robot_contact_after_release"
            elif m.goal_distance >= 0.022 or m.height > 0.025:
                self.reason = "placement_lost_after_release"
            elif self.phase == Phase.RETREAT:
                if m.opened and m.tip_height >= 0.075 and m.retreat_xy < 0.025:
                    self.phase = Phase.RETURN
            else:
                self.home_verifying = (
                    m.supported
                    and m.opened
                    and m.linear_speed < 0.02
                    and m.angular_speed < 0.5
                    and m.home_distance < 0.012
                    and m.home_joint_error < 0.12
                )
                home = (
                    self.home_verifying and m.tip_speed < 0.02 and m.joint_speed < 0.15
                )
                self.home_timer = self.home_timer + dt if home else 0.0
                if self.home_timer + 1e-9 >= 0.3:
                    self.reason = "success"
            return
        conditions = {
            "grasp": m.grasped,
            "lift": m.grasped and m.height >= 0.065,
            "target": m.grasped and m.height >= 0.055 and m.goal_xy < 0.02,
            "support": m.supported
            and m.goal_distance < 0.022
            and m.linear_speed < 0.035,
            "clear": self.phase >= Phase.LIFT and not m.robot_contact,
            "fall": (
                self.phase in (Phase.LIFT, Phase.TRANSPORT) and not m.grasped
                if self.allow_recovery
                else self.phase >= Phase.LIFT
                and not m.robot_contact
                and not m.supported
                and m.vertical_speed < -0.12
            ),
            "settle": self.phase == Phase.SETTLE
            and m.supported
            and m.goal_distance < 0.022
            and m.linear_speed < 0.02
            and m.angular_speed < 0.5
            and not m.robot_contact
            and (m.opened or not self.return_home),
        }
        for name, active in conditions.items():
            self.timers[name] = self.timers[name] + dt if active else 0.0
        # Latch unsupported release; landing later cannot turn a throw into a place.
        if self.phase >= Phase.LIFT and conditions["clear"]:
            self.unsafe_release |= not m.supported and m.height > 0.017
        elif not conditions["clear"]:
            self.unsafe_release = False
        if m.outside:
            self.reason = "outside_workspace"
        elif self.phase == Phase.SETTLE and m.robot_contact:
            self.reason = "robot_contact_after_release"
        elif self.phase >= Phase.LIFT and self.timers["clear"] >= 0.02:
            if self.unsafe_release:
                self.reason = "released_in_air"
            elif m.goal_distance >= 0.022:
                self.reason = "released_outside_goal"
            elif (
                self.phase == Phase.RELEASE
                and m.supported
                and (m.opened or not self.return_home)
            ):
                self.phase = Phase.SETTLE
        if (
            not self.reason
            and self.phase >= Phase.LIFT
            and m.supported
            and not m.robot_contact
            and m.goal_distance >= 0.022
        ):
            self.reason = "released_outside_goal"
        if not self.reason and self.timers["fall"] >= (
            0.12 if self.allow_recovery else 0.06
        ):
            self.reason = "lost_grasp_in_air"
        if self.allow_recovery and self.reason in (
            "released_in_air",
            "released_outside_goal",
            "lost_grasp_in_air",
        ):
            self.recover(self.reason)
            return
        if self.reason:
            return
        if self.phase == Phase.REACH and self.timers["grasp"] >= 0.04:
            self.phase = Phase.LIFT
            self.grasp_count += 1
        elif self.phase == Phase.LIFT and self.timers["lift"] >= 0.06:
            self.phase = Phase.TRANSPORT
        elif self.phase == Phase.TRANSPORT and self.timers["target"] >= 0.04:
            self.phase = Phase.LOWER
        elif self.phase == Phase.LOWER and self.timers["support"] >= 0.06:
            self.phase = Phase.RELEASE
        if self.phase == Phase.SETTLE and self.timers["settle"] >= 0.4:
            if self.return_home:
                self.phase = Phase.RETREAT
            else:
                self.reason = "success"

    def observation(self) -> np.ndarray:
        return np.r_[
            np.eye(8)[self.phase, :6], list(self.timers.values()), self.unsafe_release
        ]

"""新タスクの物理契約と単一方策APIの回帰テスト。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from gymnasium.utils.env_checker import check_env

from pick_place import (
    NARROW_COMMAND,
    NARROW_CUBE_BOUNDS,
    STAGES,
    PickPlaceEnv,
)
from task import require_current_config, scripted_actions
from task_logic import Measurement, Phase, TaskState


def measurement(**changes) -> Measurement:
    return replace(
        Measurement(
            True, False, True, False, 0.07, 0.0, 0.055, 0.011, 0.0, 0.0, 0.0, 0.0
        ),
        **changes,
    )


class TaskLogicTests(unittest.TestCase):
    def tick(self, state, m, seconds):
        for _ in range(round(seconds / 0.002)):
            state.tick(m, 0.002)

    def test_release_requires_support_before_settle(self):
        s = TaskState(Phase.LOWER)
        supported = measurement(supported=True, height=0.015, goal_distance=0.001)
        self.tick(s, supported, 0.08)
        self.assertEqual(s.phase, Phase.RELEASE)
        self.tick(
            s,
            replace(supported, grasped=False, robot_contact=False, opened=True),
            0.024,
        )
        self.assertEqual(s.phase, Phase.SETTLE)
        self.assertFalse(s.reason)

    def test_throw_cannot_be_converted_into_placement_by_landing(self):
        s = TaskState(Phase.LOWER)
        air = measurement(grasped=False, robot_contact=False, opened=True)
        self.tick(s, air, 0.012)
        self.tick(
            s, replace(air, supported=True, height=0.015, goal_distance=0.001), 0.012
        )
        self.assertEqual(s.reason, "released_in_air")

    def test_table_contact_force_flicker_is_not_an_airborne_drop(self):
        s = TaskState(Phase.RELEASE)
        self.tick(
            s,
            measurement(
                grasped=False,
                robot_contact=False,
                opened=False,
                height=0.016,
                goal_distance=0.001,
            ),
            0.024,
        )
        self.assertFalse(s.reason)
        self.assertEqual(s.phase, Phase.RELEASE)
        self.tick(
            s,
            measurement(
                grasped=False,
                robot_contact=False,
                opened=False,
                height=0.015,
                goal_distance=0.001,
                supported=True,
            ),
            0.002,
        )
        self.assertEqual(s.phase, Phase.SETTLE)

    def test_airborne_loss_is_detected_even_before_jaw_opens_fully(self):
        s = TaskState(Phase.LOWER)
        self.tick(
            s, measurement(grasped=False, robot_contact=False, opened=False), 0.024
        )
        self.assertEqual(s.reason, "released_in_air")

    def test_timers_persist_across_actions_and_filter_contact_flicker(self):
        s = TaskState(Phase.TRANSPORT)
        air = measurement(grasped=False, robot_contact=False, opened=True)
        self.tick(s, air, 0.012)
        self.tick(s, measurement(), 0.002)
        self.assertEqual(s.timers["clear"], 0)
        self.tick(s, air, 0.012)
        self.assertFalse(s.reason)
        self.tick(s, air, 0.012)
        self.assertEqual(s.reason, "released_in_air")

    def test_settle_requires_consecutive_stationary_supported_object(self):
        s = TaskState(Phase.SETTLE)
        stable = measurement(
            grasped=False,
            supported=True,
            robot_contact=False,
            opened=True,
            height=0.015,
            goal_distance=0.001,
            clearance=0.045,
        )
        self.tick(s, stable, 0.30)
        self.assertFalse(s.reason)
        self.tick(s, replace(stable, linear_speed=0.1), 0.002)
        self.assertEqual(s.timers["settle"], 0)
        self.tick(s, stable, 0.30)
        self.assertFalse(s.reason)
        self.tick(s, stable, 0.11)
        self.assertEqual(s.reason, "success")

    def test_recontact_fails_without_rewarding_reach_again(self):
        s = TaskState(Phase.SETTLE)
        s.tick(measurement(), 0.002)
        self.assertEqual(s.reason, "robot_contact_after_release")
        self.assertEqual(s.phase, Phase.SETTLE)


class RecoveryLogicTests(unittest.TestCase):
    def test_drop_resets_progress_and_cannot_become_success_by_landing(self):
        state = TaskState(Phase.TRANSPORT, allow_recovery=True, grasp_count=1)
        for _ in range(12):
            state.tick(measurement(grasped=False, robot_contact=False), 0.002)
        self.assertEqual(state.phase, Phase.REACH)
        self.assertEqual(state.recoveries, 1)
        self.assertFalse(state.reason)
        landed = measurement(
            grasped=False,
            robot_contact=False,
            supported=True,
            height=0.015,
            goal_distance=0.001,
        )
        for _ in range(300):
            state.tick(landed, 0.002)
        self.assertEqual(state.phase, Phase.REACH)
        self.assertFalse(state.reason)
        for _ in range(25):
            state.tick(measurement(height=0.015), 0.002)
        self.assertEqual(state.phase, Phase.LIFT)
        self.assertEqual(state.grasp_count, 2)

    def test_single_pad_contact_is_eventually_retried_and_settle_stays_terminal(self):
        state = TaskState(Phase.LIFT, allow_recovery=True, grasp_count=1)
        for _ in range(65):
            state.tick(measurement(grasped=False, robot_contact=True), 0.002)
        self.assertEqual(state.phase, Phase.REACH)
        self.assertEqual(state.recoveries, 1)
        state = TaskState(Phase.SETTLE, allow_recovery=True)
        state.tick(measurement(), 0.002)
        self.assertEqual(state.reason, "robot_contact_after_release")
        state = TaskState(Phase.REACH, allow_recovery=True)
        state.tick(measurement(outside=True), 0.002)
        self.assertEqual(state.reason, "outside_workspace")


class RelocationTests(unittest.TestCase):
    def test_recovery_deadline_finishes_on_the_last_allowed_action(self):
        env = PickPlaceEnv(stage="pick", recovery=True)
        try:
            _, info = env.reset(seed=1)
            done = False
            _step = -1
            for _step in range(env.max_steps):
                _, _, done, _, info = env.step(np.zeros(4, dtype=np.float32))
                if done:
                    break
            self.assertTrue(done)
            self.assertEqual(info["termination_reason"], "timeout")
            self.assertEqual(_step + 1, env.max_steps)
        finally:
            env.close()

    def test_relocation_does_not_rescue_an_episode_already_outside(self):
        env = PickPlaceEnv(stage="transport", recovery=True, cube_relocations=True)
        try:
            env.state.reason = "outside_workspace"
            cube = env.cube.copy()
            env._maybe_relocate_cube()
            np.testing.assert_array_equal(env.cube, cube)
            self.assertEqual(env.state.reason, "outside_workspace")
            self.assertEqual(len(env.relocation_events), 0)
        finally:
            env.close()

    def test_relocation_preserves_robot_and_requires_fresh_grasp(self):
        env = PickPlaceEnv(stage="transport", recovery=True, cube_relocations=True)
        try:
            env.reset(seed=7)
            q = env.data.qpos[:6].copy()
            ctrl = env.data.ctrl.copy()
            time = env.data.time
            before = env.cube.copy()
            env._maybe_relocate_cube()
            np.testing.assert_array_equal(q, env.data.qpos[:6])
            np.testing.assert_array_equal(ctrl, env.data.ctrl)
            self.assertEqual(time, env.data.time)
            self.assertGreater(np.linalg.norm(env.cube[:2] - before[:2]), 0.02)
            self.assertAlmostEqual(env.cube[2], 0.015)
            self.assertEqual(env.state.phase, Phase.REACH)
            self.assertFalse(env._grasped())
            self.assertEqual(env.state.recoveries, 1)
            self.assertEqual(env._info()["relocation_count"], 1)
            compensation = env.episode_relocation_compensation
            self.assertGreater(compensation, 0)
            self.assertLess(compensation, 3)
            env._maybe_relocate_cube()
            self.assertEqual(env.episode_relocation_compensation, compensation)
            self.assertEqual(env._info()["relocation_count"], 1)
            self.assertEqual(len(env._observation()), 69)
        finally:
            env.close()

    def test_relocation_and_strong_physics_are_seeded_and_stable(self):
        env = PickPlaceEnv(
            stage="transport",
            recovery=True,
            cube_relocations=True,
            disturbances=True,
            domain_randomization=True,
            disturbance_strength=3,
            domain_scale=3,
        )
        try:
            for seed in range(10):
                env.reset(seed=seed)
                env._maybe_relocate_cube()
                location = env.cube.copy()
                env.reset(seed=seed)
                env._maybe_relocate_cube()
                np.testing.assert_array_equal(location, env.cube)
                for _ in range(5):
                    env.step(np.zeros(4, dtype=np.float32))
                self.assertTrue(np.isfinite(env.data.qpos).all())
                self.assertFalse(any(w.number for w in env.data.warning))
        finally:
            env.close()


class WorkspaceTests(unittest.TestCase):
    def test_default_scale_keeps_original_command_region(self):
        env = PickPlaceEnv(recovery=True)
        try:
            np.testing.assert_allclose(env.command_bounds, NARROW_COMMAND)
            np.testing.assert_allclose(env.cube_bounds, NARROW_CUBE_BOUNDS)
        finally:
            env.close()

    def test_scale_outside_unit_interval_is_rejected(self):
        for scale in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                PickPlaceEnv(workspace_scale=scale)

    def test_wide_region_is_reached_top_down_without_table_contact(self):
        env = PickPlaceEnv(stage="pick", recovery=True, workspace_scale=1.0)
        try:
            for xy in ([0.12, 0.0], [0.23, 0.18], [0.10, -0.10]):
                env.reset(seed=0)
                for k in range(120):
                    goal = np.r_[xy, 0.03 if k > 50 else 0.07]
                    action = np.r_[np.clip((goal - env.tip) / 0.01, -1, 1), 1.0]
                    _, _, done, _, _ = env.step(action.astype(np.float32))
                    self.assertFalse(done)
                    self.assertFalse(
                        any(
                            {c.geom1, c.geom2} & env.robot_geoms
                            and env.table_geom in (c.geom1, c.geom2)
                            and c.dist < 0
                            for c in env.data.contact
                        )
                    )
                np.testing.assert_allclose(env.tip, [*xy, 0.03], atol=0.003)
                rotation = env.data.site_xmat[env.tip_site].reshape(3, 3)
                self.assertGreater(rotation[2, 2], np.cos(np.radians(3)))
        finally:
            env.close()


class PickPlaceTests(unittest.TestCase):
    def setUp(self):
        self.env = PickPlaceEnv()
        self.addCleanup(self.env.close)

    def test_gym_api_and_seeded_reset(self):
        check_env(self.env, skip_render_check=True)
        self.env.set_stage("random")
        first, _ = self.env.reset(seed=42)
        self.env.step(np.ones(4, dtype=np.float32))
        second, _ = self.env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        third, _ = self.env.reset(seed=43)
        self.assertFalse(np.array_equal(first, third))

    def test_held_starts_are_real_contact_without_weld(self):
        for stage in ("release", "place", "transport", "lift"):
            self.env.set_stage(stage)
            for seed in range(10):
                _, info = self.env.reset(seed=seed)
                self.assertTrue(info["grasped"])
                self.assertFalse(info["is_success"])
                self.assertFalse(info["full_task"])
                self.assertEqual(self.env.model.neq, 0)
                for _ in range(10):
                    _, _, done, _, info = self.env.step(np.zeros(4, np.float32))
                    self.assertFalse(done)
                self.assertTrue(info["grasped"])

    def test_fixed_controller_physically_completes_every_start_stage(self):
        for stage in STAGES:
            for seed in range(3):
                with self.subTest(stage=stage, seed=seed):
                    self.env.set_stage(stage)
                    self.env.reset(seed=seed)
                    info = {}
                    for _, a in scripted_actions(self.env):
                        obs, reward, done, truncated, info = self.env.step(a)
                        self.assertTrue(self.env.observation_space.contains(obs))
                        self.assertTrue(np.isfinite(reward))
                        if done or truncated:
                            break
                    self.assertTrue(info["is_success"], info)
                    self.assertFalse(self.env._measurement().robot_contact)
                    with self.assertRaises(RuntimeError):
                        self.env.step(np.zeros(4, np.float32))

    def test_noop_does_not_count_as_full_task_success(self):
        self.env.set_stage("pick")
        _, info = self.env.reset(seed=0)
        for _ in range(self.env.max_steps):
            _, _, done, timeout, info = self.env.step(np.zeros(4, np.float32))
            if done or timeout:
                break
        self.assertFalse(info["is_success"])
        self.assertTrue(info["full_task"])
        self.assertEqual(info["termination_reason"], "timeout")
        self.assertGreaterEqual(info["episode_torque_cost"], 0)

    def test_curriculum_bridges_transport_but_evaluation_keeps_full_distance(self):
        self.env.set_stage("transport", True)
        distances = []
        for seed in range(30):
            _, info = self.env.reset(seed=seed)
            if info["stage"] == "transport":
                distances.append(self.env._measurement().goal_xy)
                self.assertTrue(info["grasped"])
                self.assertTrue(info["curriculum_start"])
        self.assertLess(min(distances), 0.015)
        self.assertGreater(max(distances), 0.07)
        self.env.set_stage("transport")
        for seed in range(3):
            _, info = self.env.reset(seed=seed)
            self.assertGreater(self.env._measurement().goal_xy, 0.08)
            self.assertFalse(info["curriculum_start"])

    def test_blocked_jaw_command_does_not_wind_up(self):
        self.env.set_stage("transport")
        self.env.reset(seed=0)
        for _ in range(15):
            self.env.step(np.array([0, 0, 0, -1], np.float32))
            self.assertGreaterEqual(
                self.env.data.ctrl[5], self.env.data.qpos[5] - 0.062
            )
        self.assertTrue(self.env._grasped())

    def test_deadline_is_terminal_failure_with_zero_terminal_potential(self):
        self.env.set_stage("pick")
        self.env.reset(seed=0)
        self.env.max_steps = 1
        _, reward, done, truncated, info = self.env.step(np.zeros(4, np.float32))
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "timeout")
        self.assertEqual(self.env.previous_potential, 0)
        self.assertLess(reward, -self.env.timeout_penalty)

    def test_waiting_for_deadline_costs_more_than_failing(self):
        self.assertGreater(
            PickPlaceEnv.timeout_penalty, 2 * PickPlaceEnv.failure_penalty
        )

    def test_release_finishes_policy_and_holds_commands_during_settling(self):
        self.env.set_stage("release")
        _, info = self.env.reset(seed=0)
        original = TaskState.tick
        controls = []

        def track(state, m, dt):
            original(state, m, dt)
            if state.phase == Phase.SETTLE:
                controls.append(self.env.data.ctrl.copy())

        with patch.object(TaskState, "tick", track):
            for _, action in scripted_actions(self.env):
                _, _, done, truncated, info = self.env.step(action)
                if info["phase"] == "settle":
                    self.assertTrue(done)
                if done or truncated:
                    break
        self.assertTrue(info["is_success"])
        self.assertGreaterEqual(len(controls), 200)
        for control in controls:
            np.testing.assert_array_equal(control, controls[0])
        with self.assertRaises(RuntimeError):
            self.env.step(np.ones(4, np.float32))

    def test_approach_cost_rewards_open_approach_and_stops_after_grasp(self):
        self.env.set_stage("pick")
        self.env.reset(seed=0)
        close = np.array([0, 0, 0, -1.0])
        opening = np.array([0, 0, 0, 1.0])
        self.assertGreater(
            self.env._approach_cost(close), self.env._approach_cost(opening)
        )
        self.assertGreater(self.env._approach_cost(opening), 0)
        for phase in (
            Phase.LIFT,
            Phase.TRANSPORT,
            Phase.LOWER,
            Phase.RELEASE,
            Phase.SETTLE,
        ):
            self.env.state.phase = phase
            self.assertEqual(self.env._approach_cost(close), 0)

    def test_invalid_action_is_rejected(self):
        for a in (np.zeros(6, np.float32), np.array([0, 0, 0, np.nan], np.float32)):
            with self.assertRaises(ValueError):
                self.env.step(a)

    def test_legacy_policy_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "新規学習"):
            require_current_config({"task_version": "supported-release-v3"})


class StartDistributionTests(unittest.TestCase):
    def test_expanded_starts_are_reproducible_supported_and_displaced(self):
        env = PickPlaceEnv(stage="random", start_level=5)
        try:
            sources, relative = [], []
            for seed in range(50):
                obs, info = env.reset(seed=seed)
                again, _ = env.reset(seed=seed)
                np.testing.assert_array_equal(obs, again)
                self.assertEqual(info["start_level"], 5)
                self.assertFalse(info["robot_contact"])
                self.assertTrue(env._measurement().supported)
                self.assertEqual(env.state.phase, Phase.REACH)
                self.assertEqual(len(obs), 69)
                sources.append(env.cube.copy())
                relative.append(env.tip - env.cube)
            self.assertLess(np.min(np.array(sources)[:, 0]), 0.21)
            self.assertGreater(np.ptp(np.array(sources)[:, 1]), 0.06)
            self.assertGreater(np.ptp(np.array(relative)[:, 0]), 0.025)
            self.assertGreater(np.ptp(np.array(relative)[:, 2]), 0.025)
        finally:
            env.close()

    def test_level_changes_only_at_reset_and_pick_retains_canonical_start(self):
        env = PickPlaceEnv(stage="random")
        try:
            before, _ = env.reset(seed=17)
            env.set_start_level(4)
            self.assertEqual(env.start_level, 0)
            np.testing.assert_array_equal(before, env._observation())
            env.reset(seed=17)
            self.assertEqual(env.start_level, 4)
            env.set_stage("pick")
            env.reset(seed=17)
            self.assertEqual(env.start_level, 0)
            np.testing.assert_allclose(env.cube[:2], [0.23, 0], atol=1e-5)
            with self.assertRaises(ValueError):
                env.set_start_level(6)
        finally:
            env.close()


class RobustnessTests(unittest.TestCase):
    def test_physics_randomization_is_seeded_bounded_and_non_cumulative(self):
        env = PickPlaceEnv(
            stage="random", start_level=5, domain_randomization=True, disturbances=True
        )
        nominal = PickPlaceEnv(stage="random", start_level=5)
        try:
            snapshots = []
            for seed in range(12):
                obs, info = env.reset(seed=seed)
                physics = env.model.body_mass.copy()
                again, second = env.reset(seed=seed)
                np.testing.assert_array_equal(obs, again)
                np.testing.assert_array_equal(physics, env.model.body_mass)
                self.assertEqual(info, second)
                nominal.reset(seed=seed)
                np.testing.assert_allclose(env.goal, nominal.goal, atol=1e-12)
                np.testing.assert_allclose(env.cube[:2], nominal.cube[:2], atol=1e-5)
                self.assertTrue(env._measurement().supported)
                self.assertFalse(info["robot_contact"])
                self.assertGreaterEqual(info["cube_mass"], 0.02)
                self.assertLessEqual(info["cube_mass"], 0.03)
                snapshots.append(info["cube_mass"])
            self.assertGreater(np.ptp(snapshots), 0.002)
            env.randomization_probability = 0.0
            env.reset(seed=3)
            for name, values in env.nominal_physics.items():
                np.testing.assert_array_equal(getattr(env.model, name), values)
        finally:
            env.close()
            nominal.close()

    def test_table_push_moves_cube_and_reset_clears_applied_forces(self):
        env = PickPlaceEnv(stage="pick", disturbances=True)
        try:
            _, info = env.reset(seed=8)
            start = env.cube.copy()
            for _ in range(25):
                _, _, done, _, info = env.step(np.zeros(4, dtype=np.float32))
                self.assertFalse(done)
            self.assertGreater(info["disturbance_count"], 0)
            self.assertGreater(info["disturbance_impulse"], 0)
            self.assertGreater(np.linalg.norm(env.cube[:2] - start[:2]), 0.001)
            env.state = TaskState(Phase.SETTLE)
            env.data.xfrc_applied[env.cube_body, :3] = 1
            env._apply_disturbance()
            np.testing.assert_array_equal(env.data.xfrc_applied[env.cube_body], 0)
            env.reset(seed=8)
            self.assertEqual(env.disturbance_count, 0)
            np.testing.assert_array_equal(env.data.xfrc_applied, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

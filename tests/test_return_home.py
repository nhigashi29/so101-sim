"""配置・解放・退避・帰還の順序と、既存方策からの移行を検証する。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_place import PickPlaceEnv
from policy_transfer import transfer_to_return
from task import scripted_actions
from task_logic import Measurement, Phase, TaskState


def placed(**changes):
    return replace(
        Measurement(
            grasped=False,
            supported=True,
            robot_contact=False,
            opened=True,
            height=0.015,
            goal_xy=0.001,
            goal_distance=0.001,
            clearance=0.01,
            linear_speed=0,
            angular_speed=0,
            tip_speed=0,
            vertical_speed=0,
            tip_height=0.025,
            retreat_xy=0,
        ),
        **changes,
    )


def tick(state, measurement, seconds):
    for _ in range(round(seconds / 0.002)):
        state.tick(measurement, 0.002)


class ReturnHomeTests(unittest.TestCase):
    def test_no_success_without_opening_retreat_and_stationary_home(self):
        state = TaskState(Phase.RELEASE, return_home=True, allow_recovery=True)
        tick(state, placed(opened=False), 0.6)
        self.assertEqual(state.phase, Phase.RELEASE)
        tick(state, placed(), 0.45)
        self.assertEqual(state.phase, Phase.RETREAT)
        self.assertFalse(state.reason)
        # A lateral exit at table height cannot skip the vertical retreat.
        tick(state, placed(home_distance=0, home_joint_error=0, joint_speed=0), 0.5)
        self.assertEqual(state.phase, Phase.RETREAT)
        state.tick(placed(tip_height=0.08), 0.002)
        self.assertEqual(state.phase, Phase.RETURN)
        tick(state, placed(tip_height=0.08), 0.5)
        self.assertFalse(state.reason)
        home = placed(
            tip_height=0.09, home_distance=0.001, home_joint_error=0.01, joint_speed=0
        )
        tick(state, home, 0.25)
        tick(state, replace(home, joint_speed=0.3), 0.002)
        self.assertEqual(state.home_timer, 0)
        tick(state, replace(home, opened=False), 0.4)
        self.assertFalse(state.reason)
        tick(state, home, 0.3)
        self.assertEqual(state.reason, "success")

    def test_recontact_and_lost_placement_cannot_restart_recovery(self):
        for phase in (Phase.RETREAT, Phase.RETURN):
            for measurement, reason in (
                (placed(robot_contact=True), "robot_contact_after_release"),
                (placed(goal_distance=0.03), "placement_lost_after_release"),
            ):
                state = TaskState(phase, return_home=True, allow_recovery=True)
                state.tick(measurement, 0.002)
                self.assertEqual(state.reason, reason)
                self.assertEqual(state.recoveries, 0)

    def test_full_physical_trajectory_has_policy_steps_after_release(self):
        env = PickPlaceEnv(stage="pick", recovery=True, return_home=True)
        student = PickPlaceEnv(stage="pick", recovery=True, return_home=True)
        try:
            _, info = env.reset(seed=0)
            student.reset(seed=0)
            phases = set()
            retreat_commands = 0
            for _, action in scripted_actions(env):
                if env.state.phase == Phase.RETREAT:
                    retreat_commands += 1
                _, _, done, _, info = env.step(action)
                _, _, student_done, _, _ = student.step_joint_targets(
                    env.last_command_target
                )
                self.assertEqual(done, student_done)
                np.testing.assert_allclose(
                    student.data.qpos, env.data.qpos, rtol=0, atol=1e-10
                )
                phases.add(env.state.phase)
                if done:
                    break
            self.assertTrue(info["is_success"])
            self.assertTrue(info["gripper_opened"])
            self.assertFalse(info["robot_contact"])
            self.assertLess(info["home_distance"], 0.012)
            self.assertGreaterEqual(info["home_timer"], 0.3 - 1e-9)
            self.assertTrue({Phase.RETREAT, Phase.RETURN} <= phases)
            self.assertGreater(retreat_commands, 5)
            self.assertEqual(env.observation_space.shape, (77,))
        finally:
            env.close()
            student.close()

    def test_transfer_preserves_existing_predictions_and_normalization(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            old_env = VecNormalize(DummyVecEnv([lambda: PickPlaceEnv(stage="pick")]))
            new_raw = DummyVecEnv(
                [lambda: PickPlaceEnv(stage="pick", return_home=True)]
            )
            migrated = None
            try:
                old = PPO("MlpPolicy", old_env, n_steps=8, batch_size=8, seed=7)
                old.learn(8)
                old.save(str(source / "policy.zip"))
                old_env.save(str(source / "normalize.pkl"))
                original = (source / "policy.zip").read_bytes()
                migrated, model = transfer_to_return(source, new_raw, 1e-4, 0.02, 0.003)
                assert isinstance(migrated.obs_rms, RunningMeanStd)
                assert isinstance(old_env.obs_rms, RunningMeanStd)
                np.testing.assert_array_equal(
                    migrated.obs_rms.mean[:69], old_env.obs_rms.mean
                )
                np.testing.assert_array_equal(
                    migrated.obs_rms.var[:69], old_env.obs_rms.var
                )
                inputs = (
                    np.random.default_rng(4).normal(size=(5, 69)).astype(np.float32)
                )
                expanded = np.pad(inputs, ((0, 0), (0, 8)), constant_values=1)
                expanded[:, 69:71] = 0
                expected, _ = old.predict(inputs, deterministic=True)
                actual, _ = model.predict(expanded, deterministic=True)
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                for flag in (69, 70):
                    post_release = expanded.copy()
                    post_release[:, flag] = 1
                    neutral, _ = model.predict(post_release, deterministic=True)
                    np.testing.assert_array_equal(neutral, np.zeros((5, 4)))
                self.assertEqual(model.num_timesteps, old.num_timesteps)
                self.assertFalse(model.policy.optimizer.state)
                self.assertEqual((source / "policy.zip").read_bytes(), original)
                model.save(str(source / "expanded.zip"))
                restored = PPO.load(str(source / "expanded.zip"), device="cpu")
                after_load, _ = restored.predict(expanded, deterministic=True)
                np.testing.assert_allclose(after_load, expected, atol=1e-6)
                model.policy.optimizer.zero_grad()
                new_actions, _, _ = model.policy(
                    torch.from_numpy(post_release), deterministic=True
                )
                (new_actions - 0.5).square().mean().backward()
                model.policy.optimizer.step()
                retained, _ = model.predict(expanded, deterministic=True)
                np.testing.assert_allclose(retained, expected, atol=1e-6)
            finally:
                old_env.close()
                if migrated is not None:
                    migrated.close()
                else:
                    new_raw.close()

    def test_return_practice_starts_are_explicit_and_disabled_in_evaluation(self):
        env = PickPlaceEnv(return_home=True, return_start_probability=1)
        evaluation = PickPlaceEnv(return_home=True)
        try:
            _, info = env.reset(seed=2)
            self.assertTrue(info["return_start"])
            self.assertTrue(info["eased_start"])
            self.assertFalse(info["full_task"])
            self.assertFalse(info["robot_contact"])
            _, info = evaluation.reset(seed=2)
            self.assertFalse(info["return_start"])
            self.assertEqual(info["phase"], "lower")
        finally:
            env.close()
            evaluation.close()

    def test_retreat_rejects_downward_lateral_and_closing_motor_commands(self):
        env = PickPlaceEnv(return_home=True, return_start_probability=1)
        try:
            for seed in range(20):
                env.reset(seed=seed)
                if env.state.phase == Phase.RETREAT and env.tip[2] < 0.04:
                    break
            self.assertEqual(env.state.phase, Phase.RETREAT)
            for vertical in (-1, 1):
                import mujoco

                previous = env.data.ctrl.copy()
                jacobian = np.zeros((3, env.model.nv))
                rotation = np.zeros_like(jacobian)
                mujoco.mj_jacSite(env.model, env.data, jacobian, rotation, env.tip_site)
                _, _, done, _, info = env.step(
                    np.array([1, -1, vertical, -1], dtype=np.float32)
                )
                increment = env.last_command_target[:5] - previous[:5]
                self.assertGreaterEqual(float(jacobian[2, :5] @ increment), -1e-10)
                self.assertLess(np.linalg.norm(jacobian[:2, :5] @ increment), 1e-6)
                self.assertLess(np.linalg.norm(rotation[:, :5] @ increment), 1e-6)
                self.assertGreaterEqual(env.last_command_target[5], previous[5])
                self.assertFalse(info["robot_contact"])
                self.assertFalse(done)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

"""画像蒸留の時刻整合・非特権入力・分割・チャンク境界を検証する。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from act_data import CAMERAS, chunk_indices, split_episodes
from act_policy import EpisodeDataset, normalization
from pick_place import PickPlaceEnv
from task import scripted_actions


class ACTDataTests(unittest.TestCase):
    def test_joint_target_replay_matches_teacher_through_relocation_and_success(self):
        teacher = PickPlaceEnv(stage="pick", recovery=True, cube_relocations=True)
        student = PickPlaceEnv(stage="pick", recovery=True, cube_relocations=True)
        try:
            teacher.reset(seed=0)
            student.reset(seed=0)
            actions = iter(scripted_actions(teacher))
            recoveries = 0
            info = {}
            for _ in range(teacher.max_steps):
                _, action = next(actions)
                _, _, done, _, info = teacher.step(action)
                _, _, student_done, _, other = student.step_joint_targets(
                    teacher.last_command_target
                )
                np.testing.assert_allclose(
                    student.data.qpos, teacher.data.qpos, rtol=0, atol=1e-10
                )
                self.assertEqual(done, student_done)
                self.assertEqual(info["phase"], other["phase"])
                if done:
                    break
                if teacher.state.recoveries != recoveries:
                    recoveries = teacher.state.recoveries
                    actions = iter(scripted_actions(teacher))
            self.assertTrue(info["is_success"])
            self.assertEqual(info["relocation_count"], 1)
            self.assertGreaterEqual(info["regrasp_count"], 1)
        finally:
            teacher.close()
            student.close()

    def test_chunks_never_cross_unpredictable_interventions(self):
        boundaries = np.array([True, False, True, False])
        indices, mask = chunk_indices(boundaries, 0, 4)
        np.testing.assert_array_equal(indices, [0, 1, 1, 1])
        np.testing.assert_array_equal(mask, [False, False, True, True])
        indices, mask = chunk_indices(boundaries, 2, 4)
        np.testing.assert_array_equal(indices, [2, 3, 3, 3])
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_episode_split_is_disjoint_and_stratified(self):
        episodes = [
            {"scenario": name, "seed": seed}
            for name in ("standard", "recovery", "strong")
            for seed in range(5)
        ]
        train, val = split_episodes(episodes, 7)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(15)))
        self.assertEqual(
            {episodes[i]["scenario"] for i in val}, {"standard", "recovery", "strong"}
        )
        self.assertEqual((train, val), split_episodes(episodes, 7))

    def test_normalization_uses_training_only_and_student_has_no_privileged_features(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for i, value in enumerate([0.0, 999.0]):
                name = f"episode_{i}.npz"
                np.savez_compressed(
                    root / name,
                    state=np.full((3, 6), value, dtype=np.float32),
                    action=np.full((3, 6), value, dtype=np.float32),
                    wrist=np.zeros((3, 8, 8, 3), dtype=np.uint8),
                    overhead=np.ones((3, 8, 8, 3), dtype=np.uint8),
                    segment_start=np.array([True, False, True]),
                )
                entries.append(dict(file=name, frames=3, scenario="standard"))
            stats = normalization(root, entries, [0], 4)
            np.testing.assert_array_equal(stats["state"]["mean"], np.zeros(6))
            data = EpisodeDataset(root, entries, [0], stats, 4)
            sample = data[0]
            self.assertIsInstance(data.episode(0)["wrist"], np.memmap)
            self.assertEqual(
                set(sample),
                {
                    "observation.state",
                    "action",
                    "action_is_pad",
                    *[f"observation.images.{name}" for name in CAMERAS],
                },
            )
            self.assertEqual(tuple(sample["observation.state"].shape), (6,))
            self.assertEqual(tuple(sample["action"].shape), (4, 6))
            np.testing.assert_array_equal(
                sample["action_is_pad"].numpy(), [False, False, True, True]
            )

    def test_delta_labels_round_trip_to_joint_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(0)
            state = np.cumsum(rng.normal(0, 0.01, (8, 6)), axis=0).astype(np.float32)
            action = (state + rng.normal(0, 0.02, (8, 6))).astype(np.float32)
            np.savez_compressed(
                root / "episode_0.npz",
                state=state,
                action=action,
                wrist=np.zeros((8, 8, 8, 3), dtype=np.uint8),
                overhead=np.zeros((8, 8, 8, 3), dtype=np.uint8),
                segment_start=np.zeros(8, dtype=bool),
            )
            entries = [dict(file="episode_0.npz", frames=8, scenario="standard")]
            stats = normalization(root, entries, [0], 3)
            self.assertEqual(np.array(stats["delta"]["std"]).shape, (3, 6))
            sample = EpisodeDataset(root, entries, [0], stats, 3)[2]
            restored = (
                state[2]
                + sample["action"].numpy() * np.array(stats["delta"]["std"])
                + np.array(stats["delta"]["mean"])
            )
            np.testing.assert_allclose(restored, action[2:5], atol=1e-5)

    def test_colab_cells_are_valid_python_and_select_gpu(self):
        notebook = json.loads(
            (
                Path(__file__).resolve().parents[1] / "notebooks/act_colab.ipynb"
            ).read_text()
        )
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        for number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"colab_cell_{number}", "exec")


if __name__ == "__main__":
    unittest.main()

"""再開時のPPO設定と元のチェックポイントの保持を検証する。"""

import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stable_baselines3 import PPO

from pick_place import TASK_VERSION
from task import start_assignments, start_curriculum_passed, training_stages

ROOT = Path(__file__).resolve().parent.parent


class TrainingTests(unittest.TestCase):
    def test_rehearsal_allocates_steps_with_separate_environments(self) -> None:
        assignments = training_stages("pick", 8)
        self.assertEqual(assignments.count("pick"), 6)
        self.assertEqual(assignments.count("grasp"), 2)
        self.assertEqual(training_stages("pick", 8, True), ["pick"] * 8)
        self.assertEqual(training_stages("release", 8), ["release"] * 8)
        self.assertEqual(training_stages("pick", 1), ["pick"])

    def test_spatial_rehearsal_and_retention_gate(self) -> None:
        assignments = start_assignments("random", 8, 3, True, False)
        self.assertEqual(assignments.count(("random", 3)), 6)
        self.assertEqual(assignments.count(("random", 2)), 2)
        reports = [
            {
                "stage": "random",
                "start_level": 3,
                "deterministic": True,
                "success_rate": 0.9,
            },
            {
                "stage": "random",
                "start_level": 3,
                "deterministic": False,
                "success_rate": 0.8,
            },
            {
                "stage": "random",
                "start_level": 0,
                "deterministic": True,
                "success_rate": 0.5,
            },
        ]
        self.assertFalse(start_curriculum_passed(reports))
        reports[-1]["success_rate"] = 0.8
        self.assertTrue(start_curriculum_passed(reports))

    def test_spatial_resume_keeps_range_and_expansion_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, resumed = (Path(directory) / name for name in ("source", "resumed"))
            command = [
                sys.executable,
                str(ROOT / "src/task.py"),
                "train",
                "--steps",
                "32",
                "--envs",
                "1",
                "--rollout",
                "16",
                "--eval-every",
                "32",
                "--eval-episodes",
                "1",
            ]
            subprocess.run(
                [
                    *command,
                    "--stage",
                    "random",
                    "--expand-starts",
                    "--start-level",
                    "2",
                    "--max-start-level",
                    "4",
                    "--domain-randomization",
                    "--disturbances",
                    "--recovery",
                    "--cube-relocations",
                    "--disturbance-strength",
                    "3",
                    "--domain-scale",
                    "3",
                    "--output",
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            subprocess.run(
                [*command, "--resume", str(source), "--output", str(resumed)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            config = json.loads((resumed / "config.json").read_text())
            self.assertTrue(config["expand_starts"])
            self.assertTrue(config["domain_randomization"])
            self.assertTrue(config["disturbances"])
            self.assertTrue(config["recovery"])
            self.assertTrue(config["cube_relocations"])
            self.assertEqual(config["disturbance_strength"], 3)
            self.assertEqual(config["domain_scale"], 3)
            self.assertEqual(config["start_level"], 2)
            self.assertEqual(config["max_start_level"], 4)
            self.assertEqual(config["training_env_start_levels"], [2])
            self.assertEqual(config["num_timesteps"], 64)
            report = json.loads(
                (resumed / "validation.jsonl").read_text().splitlines()[-1]
            )
            self.assertEqual(
                {item["randomized"] for item in report["evaluations"]}, {False, True}
            )

    def test_validation_and_checkpoint_are_resumable_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src/task.py"),
                    "train",
                    "--steps",
                    "32",
                    "--envs",
                    "1",
                    "--rollout",
                    "16",
                    "--eval-every",
                    "32",
                    "--eval-episodes",
                    "1",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            report = json.loads(
                (output / "validation.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(
                {r["deterministic"] for r in report["evaluations"]}, {True, False}
            )
            for name in ("checkpoints/step_32", "best/release"):
                bundle = output / name
                for filename in ("policy.zip", "normalize.pkl", "config.json"):
                    self.assertTrue((bundle / filename).is_file())
                self.assertEqual(
                    json.loads((bundle / "config.json").read_text())["task_version"],
                    TASK_VERSION,
                )

    def test_interrupt_during_subprocess_rollout_saves_and_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "src/task.py"),
                    "train",
                    "--steps",
                    "1000000",
                    "--envs",
                    "2",
                    "--rollout",
                    "16",
                    "--output",
                    str(output),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                assert process.stdout is not None
                # The first rollout log proves learn() is inside its try/finally.
                for line in process.stdout:
                    if "total_timesteps" in line:
                        break
                process.send_signal(signal.SIGINT)
                text, _ = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, text)
                for filename in ("policy.zip", "normalize.pkl", "config.json"):
                    self.assertTrue((output / filename).is_file())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

    def test_resume_applies_learning_rate_and_kl_to_actual_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            resumed = Path(directory) / "resumed"
            command = [
                sys.executable,
                str(ROOT / "src/task.py"),
                "train",
                "--steps",
                "32",
                "--envs",
                "1",
                "--rollout",
                "16",
            ]
            subprocess.run(
                [
                    *command,
                    "--output",
                    str(source),
                    "--learning-rate",
                    "0.0003",
                    "--target-kl",
                    "0.1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            original = (source / "policy.zip").read_bytes()
            subprocess.run(
                [
                    *command,
                    "--resume",
                    str(source),
                    "--stage",
                    "lift",
                    "--output",
                    str(resumed),
                    "--learning-rate",
                    "0.0001",
                    "--target-kl",
                    "0.02",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            model = PPO.load(str(resumed / "policy.zip"), device="cpu")
            self.assertAlmostEqual(model.lr_schedule(1.0), 1e-4)
            self.assertEqual(model.target_kl, 0.02)
            self.assertEqual(model.num_timesteps, 64)
            for group in model.policy.optimizer.param_groups:
                self.assertAlmostEqual(group["lr"], 1e-4)
            config = json.loads((resumed / "config.json").read_text())
            self.assertEqual(config["learning_rate"], 1e-4)
            self.assertEqual(config["target_kl"], 0.02)
            self.assertEqual(config["task_version"], TASK_VERSION)
            self.assertEqual(config["stage"], "lift")
            self.assertTrue((resumed / "normalize.pkl").is_file())
            self.assertEqual(original, (source / "policy.zip").read_bytes())


if __name__ == "__main__":
    unittest.main()

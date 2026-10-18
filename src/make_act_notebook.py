"""Colabノートブックを生成する。各コードセルは通常のPython。"""

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def main():
    cells = []

    def markdown(text):
        cells.append(
            dict(
                cell_type="markdown",
                metadata={},
                source=dedent(text).strip().splitlines(True),
            )
        )

    def code(text):
        source = dedent(text).strip() + "\n"
        for arguments in (
            ["format", "--stdin-filename", "cell.py", "-"],
            [
                "check",
                "--select",
                "I",
                "--fix",
                "--quiet",
                "--stdin-filename",
                "cell.py",
                "-",
            ],
        ):
            result = subprocess.run(
                [sys.executable, "-m", "ruff", *arguments],
                input=source,
                text=True,
                capture_output=True,
                check=True,
            )
            source = result.stdout
        cells.append(
            dict(
                cell_type="code",
                metadata={},
                execution_count=None,
                outputs=[],
                source=source.rstrip().splitlines(True),
            )
        )

    markdown("""
    # SO-101：手元＋俯瞰の2カメラACTをColabで学習

    PPOを教師として、**画像2枚＋6関節角度 → 6関節目標の動作列**を学ぶ。
    手元で「配置・解放・上方退避・帰還」を追加学習したPPOを教師にする。
    途中移動・物理ランダム化・外力なし。掴み直しは有効。ACTはLeRobot 0.6.1の公式実装を使用する。
    [公式ACTドキュメント](https://huggingface.co/docs/lerobot/act)

    1. 「ランタイム → ランタイムのタイプを変更」でGPUを選ぶ。
    2. 同梱の `act_colab_bundle.zip` を下のセルでアップロード。
    3. セルを上から実行。まず小規模な接続確認、その後に本収集・学習へ進む。

    データとチェックポイントは自分のGoogle Driveに保存する。
    ランタイムが切れたら、GPUを選び直して同じZIPをアップロードし、同じ実験名で再実行する。
    収集は保存済みエピソード、学習は最新チェックポイントから再開する。
    """)
    code("""
    import subprocess, sys, os, json, zipfile, hashlib, shutil
    from pathlib import Path
    subprocess.run(["nvidia-smi", "-L"], check=True)
    # Without the NVIDIA vendor file, EGL silently falls back to CPU rendering (llvmpipe).
    ICD = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if not ICD.exists():
        ICD.parent.mkdir(parents=True, exist_ok=True)
        ICD.write_text('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\\n')
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    """)
    code("""
    from google.colab import files
    uploaded = files.upload()  # act_colab_bundle.zip を選択
    archives = [Path(name) for name in uploaded if name.endswith(".zip")]
    if len(archives) != 1:
        raise ValueError("バンドルZIPを1つ選択してください。")
    destination = Path("/content/so101_act").resolve()
    with zipfile.ZipFile(archives[0]) as archive:
        for member in archive.infolist():
            target = (Path("/content") / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("想定外のZIPパスです。")
        archive.extractall("/content")
    PROJECT = destination
    manifest = json.loads((PROJECT / "bundle.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"バンドルの整合性エラー: {name}")
    print("教師PPO:", manifest.get("teacher_source", "teacher"), manifest["teacher_steps"], "steps")
    """)
    markdown("""
    ## 依存関係を独立環境に入れる

    本プロジェクトとLeRobotではNumPy/Pillowの条件が異なるため、専用環境を作る。
    Python 3.12・LeRobot 0.6.1・PyTorch 2.8.0/CUDA 12.6を使用する。モデルの公開やアカウントのトークン設定は不要。
    """)
    code("""
    import signal, threading, queue

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    uv = shutil.which("uv")
    VENV = Path("/content/so101_act_env")
    if not VENV.exists():
        command = [uv, "venv", str(VENV), "--python", sys.executable if sys.version_info[:2] == (3, 12) else "3.12"]
        subprocess.run(command, check=True)
    PYTHON = str(VENV / "bin/python")
    torch_check = subprocess.run([PYTHON, "-c", "import torch, torchvision; assert torch.__version__.split('+')[0] == '2.8.0'; assert torchvision.__version__.split('+')[0] == '0.23.0'; assert torch.cuda.is_available(); assert torch.version.cuda == '12.6'"], capture_output=True)
    if torch_check.returncode != 0:
        subprocess.run([uv, "pip", "install", "--python", PYTHON,
            "--index-url", "https://download.pytorch.org/whl/cu126",
            "--reinstall-package", "torch", "--reinstall-package", "torchvision",
            "torch==2.8.0", "torchvision==0.23.0"], check=True)
    subprocess.run([uv, "pip", "install", "--python", PYTHON, "-r", str(PROJECT / "requirements/act-colab.txt")], check=True)
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "libegl1", "libgl1", "libosmesa6", "ffmpeg"], check=True)
    subprocess.run([PYTHON, "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"], check=True)

    def run(arguments, idle_timeout=None, script="src/act.py"):
        # Colab does not show a child's inherited stdout, so stream it into the cell.
        process = subprocess.Popen([PYTHON, "-u", script, *map(str, arguments)],
            cwd=PROJECT, env={**os.environ, "PYTHONFAULTHANDLER": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = queue.Queue()
        def reader():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=reader, daemon=True).start()
        while True:
            try:
                line = lines.get(timeout=idle_timeout)
            except queue.Empty:
                process.send_signal(signal.SIGABRT)  # faulthandler prints where it is stuck
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                while (line := lines.get(timeout=5)) is not None:
                    print(line, end="")
                raise TimeoutError(f"{idle_timeout}秒間出力がないため中断しました: {arguments[0]}") from None
            if line is None:
                break
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, arguments)

    renderer = subprocess.run([PYTHON, "-c", "import mujoco; from OpenGL import GL; m = mujoco.MjModel.from_xml_string('<mujoco/>'); r = mujoco.Renderer(m, 8, 8); r.update_scene(mujoco.MjData(m)); r.render(); print(GL.glGetString(GL.GL_RENDERER).decode())"],
        env=os.environ.copy(), capture_output=True, text=True)
    print("GL_RENDERER:", renderer.stdout.strip() or renderer.stderr[-500:])
    if "llvmpipe" in renderer.stdout:
        print("GPU描画になっていません。ランタイムを再起動して上から実行してください。")
    """)
    code("""
    from google.colab import drive
    drive.mount("/content/drive")
    scene_id = manifest["files"]["assets/so101/pick_place.xml"][:8]
    teacher_id = manifest["files"]["teacher/policy.zip"][:8]
    EXPERIMENT = f"ppo_{manifest['teacher_steps']}_{teacher_id}_{scene_id}_return_home_v7"
    SAVE = Path("/content/drive/MyDrive/so101_act") / EXPERIMENT
    SAVE.mkdir(parents=True, exist_ok=True)
    TEACHER = PROJECT / "teacher"
    print("保存先:", SAVE)
    """)
    markdown("""
    ## 手元で追加学習した教師の確認

    帰還まで学習済みのPPOをバンドルに入れてから、Colabへ持ち込む。
    ColabではそのPPOの成功軌跡を収集し、ACTを学習する。
    """)
    code("""
    teacher_config = json.loads((TEACHER / "config.json").read_text())
    if not teacher_config.get("return_home", False):
        raise ValueError("帰還未学習の教師です。手元で--return-homeで追加学習し、バンドルを作り直してください。")
    print("教師:", teacher_config["task_version"], teacher_config["num_timesteps"], "steps")
    print("配置・解放・上方退避・帰還までの軌跡を収集します。")
    """)
    markdown("""
    ## カメラと接続の確認

    キューブと目標マーカーが見えるか確認する。
    小さなモデルを1更新だけ動かし、データ・ACT・保存の接続を先に確認する。
    この小さなモデルは、本学習の初期値には使用しない。
    """)
    code("""
    from IPython.display import display, Image
    PREVIEW = PROJECT / "camera_preview"
    def with_render_fallback(step):
        try:
            step()
        except (subprocess.CalledProcessError, TimeoutError):
            if os.environ.get("MUJOCO_GL") == "osmesa":
                raise
            os.environ["MUJOCO_GL"] = "osmesa"
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            print("描画をOSMesaに切り替えます。ACT学習はCUDAを使用します。")
            step()
    with_render_fallback(lambda: run(["preview", "--output", PREVIEW], idle_timeout=120))
    for name in ("wrist", "overhead"):
        print(name)
        display(Image(filename=str(PREVIEW / f"{name}.png")))
    SMOKE_DATA = Path("/content/act_smoke_data")
    command = ["collect", "--run", TEACHER, "--output", SMOKE_DATA, "--episodes", 6, "--image-size", 64, "--seed", 71000, "--scenarios", "standard"]
    def collect_smoke():
        if (SMOKE_DATA / "index.json").exists() and "--resume" not in command:
            command.append("--resume")
        run(command, idle_timeout=300)
    with_render_fallback(collect_smoke)
    SMOKE_RUN = Path("/content/act_smoke_run")
    if not (SMOKE_RUN / "latest.json").exists():
        run(["train", "--dataset", SMOKE_DATA, "--output", SMOKE_RUN, "--steps", 1,
             "--batch-size", 2, "--chunk-size", 5, "--save-every", 1,
             "--validation-batches", 1, "--workers", 0, "--tiny"])
    """)
    markdown("""
    ## 本番データの収集

    初期設定は、途中移動・物理ランダム化・外力なしの標準条件で収集する。
    位置変更と強い外乱も含めたい場合は、SCENARIOSを["standard", "recovery", "strong"]に変更する。
    成功したエピソードの全軌跡を保存するので、成功に至るまでの回復動作も含まれる。
    画像は同じ物理時刻の2視点。ラベルはPPOの4次元操作ではなく、実際の6関節目標。
    予測できない位置変更・外力開始をまたぐ未来の動作列にはマスクをかける。
    """)
    code("""
    EPISODES = 200
    SCENARIOS = ["standard"]
    DATA_DRIVE = SAVE / "dataset"
    command = ["collect", "--run", TEACHER, "--output", DATA_DRIVE,
               "--episodes", EPISODES, "--image-size", 256, "--seed", 60000,
               "--scenarios", *SCENARIOS]
    if (DATA_DRIVE / "index.json").exists(): command.append("--resume")
    run(command)
    index = json.loads((DATA_DRIVE / "index.json").read_text())
    print("成功軌跡:", len(index["episodes"]), "試行数:", index["attempts"])
    print("掴み直しを含む軌跡:", sum(e["result"]["regrasp_count"] > 0 for e in index["episodes"]))
    """)
    markdown("""
    ## ACTの本学習

    収集数・カメラ設定を変えるときは、上のEXPERIMENT名も変更する。

    エピソード単位で訓練80%・検証20%に分ける。正規化統計は訓練データだけから計算する。
    25ステップ先まで学び、実行時は毎回1ステップだけ使って画像から再計画する。
    中断後は下のセルを再実行すれば、指定した合計更新数まで続行する。
    チェックポイントには正規化・モデル・optimizerを保存する。
    """)
    code("""
    DATA_LOCAL = Path("/content/act_dataset")
    shutil.copytree(DATA_DRIVE, DATA_LOCAL, dirs_exist_ok=True)
    RUN = SAVE / "training_delta"  # 差分ラベル。旧絶対値ラベルの "training" からは再開しない
    TOTAL_STEPS = 30000
    BATCH_SIZE = 8
    command = ["train", "--dataset", DATA_LOCAL, "--output", RUN,
               "--batch-size", BATCH_SIZE, "--chunk-size", 25,
               "--save-every", 5000, "--workers", 2]
    start = 0
    if (RUN / "latest.json").exists():
        latest = json.loads((RUN / "latest.json").read_text())
        start = latest["step"]
        command.extend(["--resume", RUN / latest["checkpoint"]])
    if start < TOTAL_STEPS:
        run([*command, "--steps", TOTAL_STEPS - start])
    else:
        print("指定した更新数まで学習済み:", start)
    """)
    markdown("""
    ## 未知の配置で閉ループ評価

    検証lossだけでは配置成功は確認できないため、別シードのシミュレーターで評価する。
    画像と関節角度からACTだけで操作し、PPOは使わない。
    """)
    code("""
    best = json.loads((RUN / "best.json").read_text())
    CHECKPOINT = RUN / best["checkpoint"]
    for scenario in SCENARIOS:
        run(["evaluate", "--run", CHECKPOINT, "--episodes", 20,
             "--seed", 90000, "--scenario", scenario, "--device", "cuda"])
    """)
    markdown("""
    ## ローカルでwatchするために保存

    `act_checkpoint.zip`をダウンロードし、プロジェクトの`outputs/`に展開する。
    実機を動かす前には、カメラの配置・画角、関節の単位とゼロ点、グリッパの変換を合わせ、
    実画像で追加学習する。ここでの動作確認はMuJoCo内。
    """)
    code("""
    export = SAVE / "act_checkpoint.zip"
    with zipfile.ZipFile(export, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in CHECKPOINT.rglob("*"):
            if file.is_file() and file.name != "optimizer.pt":
                archive.write(file, Path("act_checkpoint") / file.relative_to(CHECKPOINT))
    files.download(str(export))
    print("ローカルでの実行例:")
    print(".venv-act/bin/python src/view.py --act evaluate --run outputs/act_checkpoint --scenario standard --watch")
    """)
    path = Path(__file__).resolve().parent.parent / "notebooks" / "act_colab.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = dict(
        cells=cells,
        metadata=dict(
            accelerator="GPU",
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python"),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    for i, cell in enumerate(cells):
        cell["id"] = f"act-{i:02d}"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

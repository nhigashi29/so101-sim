"""uv管理のPythonでもmacOSのMuJoCo GUIを起動する。"""

import os
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    forwarded = sys.argv[1:]
    script_name = "main.py"
    if "--pick-place" in forwarded:
        forwarded.remove("--pick-place")
        script_name = "task.py"
    elif "--act" in forwarded:
        forwarded.remove("--act")
        script_name = "act.py"
    script = Path(__file__).resolve().with_name(script_name)
    arguments = [sys.executable]
    environment = os.environ.copy()
    if sys.platform == "darwin":
        # uvのPythonの@rpath参照を、mjpythonのアプリ内からも解決する。
        library_dir = sysconfig.get_config_var("LIBDIR") or str(
            Path(sys.base_prefix) / "lib"
        )
        existing = environment.get(
            "DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib"
        )
        environment["DYLD_FALLBACK_LIBRARY_PATH"] = f"{library_dir}:{existing}"
        arguments.append(str(Path(sys.executable).parent / "mjpython"))
    arguments.extend([str(script), *forwarded])
    os.execve(sys.executable, arguments, environment)


if __name__ == "__main__":
    main()

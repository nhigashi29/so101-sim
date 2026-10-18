"""uv管理のPythonでもmacOSのMuJoCo GUIを起動する。"""

import os
from pathlib import Path
import sys
import sysconfig


def main() -> None:
    script = Path(__file__).resolve().with_name("main.py")
    arguments = [sys.executable]
    environment = os.environ.copy()
    if sys.platform == "darwin":
        # uvのPythonの@rpath参照を、mjpythonのアプリ内からも解決する。
        library_dir = sysconfig.get_config_var("LIBDIR") or str(Path(sys.base_prefix) / "lib")
        existing = environment.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
        environment["DYLD_FALLBACK_LIBRARY_PATH"] = f"{library_dir}:{existing}"
        arguments.append(str(Path(sys.executable).parent / "mjpython"))
    arguments.extend([str(script), *sys.argv[1:]])
    os.execve(sys.executable, arguments, environment)


if __name__ == "__main__":
    main()

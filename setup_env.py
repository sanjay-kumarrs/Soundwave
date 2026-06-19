import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQ_FILE = ROOT_DIR / "requirements.txt"
DESIRED_PYTHON_VERSION = "3.10.11"
DESIRED_PYTHON_MAJOR_MINOR = "3.10"


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_capture(cmd):
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def parse_required_packages(requirements_file: Path):
    packages = []
    for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Drop inline comments and environment markers.
        line = line.split("#", 1)[0].strip()
        line = line.split(";", 1)[0].strip()
        if not line:
            continue

        # Extract package name from common requirement forms (e.g., pkg>=1.0, pkg==2.0).
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if match:
            packages.append(match.group(1))

    return packages


def normalize_package_name(name):
    return name.lower().replace("_", "-")


def verify_requirements_installed(venv_python, requirements_file: Path):
    required_packages = parse_required_packages(requirements_file)

    raw_installed = run_capture([venv_python, "-m", "pip", "list", "--format=json"])
    installed_items = json.loads(raw_installed)
    installed_names = {normalize_package_name(item["name"]) for item in installed_items}

    missing = [pkg for pkg in required_packages if normalize_package_name(pkg) not in installed_names]
    if missing:
        raise RuntimeError("Missing required packages: " + ", ".join(sorted(missing)))

    # Validate dependency consistency as well.
    run([venv_python, "-m", "pip", "check"])
    print("[OK] Requirement verification passed. All required packages are installed.")


def requirements_are_satisfied(venv_python, requirements_file: Path):
    try:
        verify_requirements_installed(venv_python, requirements_file)
        return True
    except Exception as exc:
        print(f"[INFO] Requirement verification did not pass yet: {exc}")
        return False


def get_python_version(python_exe):
    version = run_capture([python_exe, "-c", "import sys; print(sys.version.split()[0])"])
    return version


def is_desired_python_version(version):
    return version == DESIRED_PYTHON_VERSION


def try_get_py_launcher_python():
    if not shutil.which("py"):
        return None

    try:
        candidate = run_capture(["py", f"-{DESIRED_PYTHON_MAJOR_MINOR}", "-c", "import sys; print(sys.executable)"])
    except Exception:
        return None

    if not candidate:
        return None

    try:
        version = get_python_version(candidate)
    except Exception:
        return None

    if is_desired_python_version(version):
        return candidate

    return None


def try_get_python_from_path():
    candidates = [
        shutil.which("python3.10"),
        shutil.which("python"),
        shutil.which("python3"),
    ]
    candidates = [c for c in candidates if c]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            version = get_python_version(candidate)
            if is_desired_python_version(version):
                return candidate
        except Exception:
            continue

    return None


def install_python_3_10_11_windows():
    print(f"[INFO] Python {DESIRED_PYTHON_VERSION} not found. Attempting automatic installation on Windows.")

    if shutil.which("winget"):
        try:
            run([
                "winget",
                "install",
                "-e",
                "--id",
                "Python.Python.3.10",
                "--version",
                DESIRED_PYTHON_VERSION,
                "--accept-package-agreements",
                "--accept-source-agreements",
            ])
            return
        except Exception:
            print("[WARN] winget installation failed. Trying direct installer download.")

    installer_url = f"https://www.python.org/ftp/python/{DESIRED_PYTHON_VERSION}/python-{DESIRED_PYTHON_VERSION}-amd64.exe"
    installer_path = Path(tempfile.gettempdir()) / f"python-{DESIRED_PYTHON_VERSION}-amd64.exe"

    print(f"[RUN] Downloading {installer_url} -> {installer_path}")
    urllib.request.urlretrieve(installer_url, installer_path)

    run([
        str(installer_path),
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_test=0",
        "Include_doc=0",
    ])


def resolve_base_python():
    try:
        current_version = get_python_version(sys.executable)
        if is_desired_python_version(current_version):
            return sys.executable
    except Exception:
        pass

    py_candidate = try_get_py_launcher_python()
    if py_candidate:
        return py_candidate

    path_candidate = try_get_python_from_path()
    if path_candidate:
        return path_candidate

    if platform.system().lower().startswith("win"):
        install_python_3_10_11_windows()

        # Re-check after installation.
        py_candidate = try_get_py_launcher_python()
        if py_candidate:
            return py_candidate

        path_candidate = try_get_python_from_path()
        if path_candidate:
            return path_candidate

    raise EnvironmentError(
        f"Python {DESIRED_PYTHON_VERSION} is required but was not found. "
        "Please install it manually from https://www.python.org/downloads/release/python-31011/"
    )


def get_venv_python(venv_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main():
    parser = argparse.ArgumentParser(description="Set up project virtual environment and verify requirements.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that requirements are installed in the existing .venv",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Force reinstall packages from requirements even if already satisfied",
    )
    args = parser.parse_args()

    if not REQ_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQ_FILE}")

    base_python = resolve_base_python()
    print(f"[INFO] Using Python interpreter: {base_python} ({get_python_version(base_python)})")

    if args.verify_only:
        venv_python = str(get_venv_python(VENV_DIR))
        if not Path(venv_python).exists():
            raise FileNotFoundError(f"Virtual env Python not found at {venv_python}")
        verify_requirements_installed(venv_python, REQ_FILE)
        return

    if not VENV_DIR.exists():
        run([base_python, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"[INFO] Virtual environment already exists at {VENV_DIR}")

    venv_python = str(get_venv_python(VENV_DIR))
    if not Path(venv_python).exists():
        raise FileNotFoundError(f"Virtual env Python not found at {venv_python}")

    run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if args.reinstall:
        run([venv_python, "-m", "pip", "install", "-r", str(REQ_FILE)])
        verify_requirements_installed(venv_python, REQ_FILE)
    else:
        if requirements_are_satisfied(venv_python, REQ_FILE):
            print("[INFO] Requirements already satisfied. Skipping reinstall.")
        else:
            run([venv_python, "-m", "pip", "install", "-r", str(REQ_FILE)])
            verify_requirements_installed(venv_python, REQ_FILE)

    print("\n[OK] Environment setup complete.")
    if platform.system().lower().startswith("win"):
        print(r"Activate with: .venv\Scripts\activate")
    else:
        print("Activate with: source .venv/bin/activate")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

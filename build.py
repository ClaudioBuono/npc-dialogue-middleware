import shutil
import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "src"
MAIN_SCRIPT = SRC_DIR / "main.py"
CONFIG_SRC = SRC_DIR / "config"

BUILD_DIR = ROOT / "build"
SPEC_DIR = ROOT
RAW_DIST_DIR = ROOT / "dist"

HURTLEX_SOURCE = SRC_DIR / "tools" / "hurtlex_EN.tsv"


def sep() -> str:
    """Return the correct separator for PyInstaller's --add-data argument."""
    return ";" if sys.platform.startswith("win") else ":"


def check_pyinstaller_installed():
    """Verify that PyInstaller is available in the current environment."""
    try:
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("PyInstaller is not installed in this environment.")
        print(f"Install it with: {sys.executable} -m pip install pyinstaller")
        sys.exit(1)


def run_pyinstaller():
    """Invoke PyInstaller to build the middleware as a single-file executable."""
    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "npc_middleware",
        "--distpath", str(RAW_DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(SPEC_DIR),
        "--paths", str(SRC_DIR),
        "--add-data", f"{CONFIG_SRC / 'settings.yaml'}{sep()}config",
        "--add-data", f"{CONFIG_SRC / 'modelconfigs.json'}{sep()}config",
        "--add-data", f"{HURTLEX_SOURCE}{sep()}.",  # destination "." = bundle root
        str(MAIN_SCRIPT),
    ], check=True, cwd=SRC_DIR)


def assemble_package(package_dir: Path):
    """Move the compiled executable and config templates into the final npc_middleware output folder.

    Creates the config/ and logs/ subfolders if missing, copies the built
    executable from the raw PyInstaller dist output, and copies the default
    config files only if they don't already exist at the destination.
    """
    config_dest = package_dir / "config"
    logs_dest = package_dir / "logs"

    package_dir.mkdir(parents=True, exist_ok=True)
    config_dest.mkdir(parents=True, exist_ok=True)
    logs_dest.mkdir(parents=True, exist_ok=True)

    exe_name = "npc_middleware.exe" if sys.platform.startswith("win") else "npc_middleware"
    built_exe = RAW_DIST_DIR / exe_name
    final_exe = package_dir / exe_name

    shutil.copy(built_exe, final_exe)
    print(f"Copied executable to {final_exe}")

    for filename in ["settings.yaml", "modelconfigs.json"]:
        src = CONFIG_SRC / filename
        dest = config_dest / filename
        if not dest.exists():
            shutil.copy(src, dest)
            print(f"Copied {filename} to {dest}")
        else:
            print(f"{filename} already present, not overwritten")


def cleanup_raw_artifacts():
    """Remove intermediate build artifacts no longer needed after packaging.

    Deletes the raw PyInstaller dist/ and build/ folders and the generated
    .spec file, leaving only the final package folder as build output.
    """
    shutil.rmtree(RAW_DIST_DIR, ignore_errors=True)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    spec_file = SPEC_DIR / "npc_middleware.spec"
    spec_file.unlink(missing_ok=True)


def check_config_files_exist():
    """Verify that the required config template files exist before building.

    Fails fast with a clear list of missing files, instead of letting
    PyInstaller fail later with a less readable "Unable to find" error
    during the --add-data step.
    """
    missing = []
    for filename in ["settings.yaml", "modelconfigs.json"]:
        path = CONFIG_SRC / filename
        if not path.exists():
            missing.append(str(path))

    if missing:
        print("Missing configuration files:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nMake sure they exist inside: {CONFIG_SRC}")
        sys.exit(1)


def main():
    """Run the full build pipeline: validate, compile, and package."""
    parser = argparse.ArgumentParser(description="Build and package npc_middleware.")
    parser.add_argument(
        "-out", "--output",
        type=Path,
        default=ROOT,
        help="Output directory (default: project root)."
    )
    args = parser.parse_args()

    package_dir = (args.output / "npc_middleware").resolve()

    print(f"== Building npc_middleware (Target: {package_dir}) ==")
    check_pyinstaller_installed()
    check_config_files_exist()   # fail fast, before invoking PyInstaller
    run_pyinstaller()

    print("\n== Assembling final package ==")
    assemble_package(package_dir)
    cleanup_raw_artifacts()

    print(f"\nDone. Package ready at: {package_dir}")
    print(f"  {package_dir / ('npc_middleware.exe' if sys.platform.startswith('win') else 'npc_middleware')}")
    print(f"  {package_dir / 'config'}")
    print(f"  {package_dir / 'logs'}")


if __name__ == "__main__":
    main()
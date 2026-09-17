"""Install the donor's ABI-pinned ROCm wheels without replacing ComfyUI's torch."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("cumesh-1.0+rocm10.0", "flex_gemm-1.0.0+rocm10.0",
            "o_voxel-0.0.1+rocm.10.0", "nvdiffrast-0.4.0+rocm10.0")


def wheel_paths(root=ROOT, system=None):
    system = system or platform.system()
    if system not in ("Windows", "Linux"):
        raise RuntimeError("Bundled ROCm wheels support Windows x64 and Linux x86_64 only.")
    tag = "win_amd64" if system == "Windows" else "linux_x86_64"
    return [root / "wheels" / system / "Python3.12" / f"{p}-cp312-cp312-{tag}.whl"
            for p in PACKAGES]


def validate_stack(torch):
    if not torch.version.hip:
        raise RuntimeError("This is not a ROCm PyTorch environment. Install the stack in README.md first.")
    if not str(torch.__version__).startswith("2.13.0+") or not str(torch.version.hip).startswith("10.0"):
        raise RuntimeError(f"Bundled wheels require Torch 2.13.0 / ROCm 10.0, not "
                           f"{torch.__version__} / HIP {torch.version.hip}. Do not mix wheel ABIs.")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot see an AMD GPU. Fix the runtime before installing extensions.")


def installed_constraints():
    lines = []
    for name in ("torch", "torchvision", "torchaudio", "rocm"):
        try:
            lines.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the Python 3.12 plan without modifying or validating the host")
    parser.add_argument("--skip-requirements", action="store_true", help="Install only the four native wheels")
    args = parser.parse_args(argv)
    wheels = wheel_paths()
    pip = [sys.executable, "-m", "pip", "install"]
    if args.dry_run:
        print("Target: CPython 3.12 x64, Torch 2.13.0, ROCm 10.0 (NOT validated by dry-run)")
        print(subprocess.list2cmdline(pip + ["--no-deps", *map(str, wheels)]))
        if not args.skip_requirements:
            print(subprocess.list2cmdline(pip + ["-c", "<installed-torch-constraints>", "-r", str(ROOT / "requirements-amd.txt")]))
        return 0
    if sys.version_info[:2] != (3, 12) or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("Use the ComfyUI CPython 3.12 x64 interpreter; these wheels are cp312.")
    missing = [str(p) for p in wheels if not p.is_file()]
    if missing:
        raise RuntimeError("Missing bundled wheel(s). Clone the full fork, including wheels/:\n" + "\n".join(missing))
    import torch
    validate_stack(torch)
    # --no-deps prevents native extension metadata from replacing the ROCm build.
    # The remaining Python dependencies are installed under the existing stack's constraints.
    with tempfile.TemporaryDirectory(prefix="trellis-rocm-") as tmp:
        constraints = Path(tmp) / "constraints.txt"
        constraints.write_text(installed_constraints(), encoding="utf-8")
        subprocess.check_call(pip + ["--no-deps", *map(str, wheels)])
        if not args.skip_requirements:
            subprocess.check_call(pip + ["-c", str(constraints), "-r", str(ROOT / "requirements-amd.txt")])
    print("Installed. Next: python tools/check_amd.py --smoke")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"AMD installation stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)

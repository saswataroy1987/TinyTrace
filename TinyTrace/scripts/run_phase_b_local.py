from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end local TinyTrace Phase-B launcher. It reuses an existing "
            "virtualenv by default, installs dependencies when needed, ensures "
            "MobileCLIP is available, verifies CUDA visibility, and starts the "
            "ActivityNet Phase-B warm-start run."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local ActivityNet Captions root containing train/val annotations and videos.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "phase_b_activitynet_v1_run",
        help="Writable local run directory.",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=REPO_ROOT / ".venv-phase-b",
        help="Virtualenv directory.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable or "python3",
        help="Python executable used to create the virtualenv if needed.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--bootstrap-checkpoint",
        type=Path,
        default=REPO_ROOT / "final_next_phase_assets" / "phase_a_bootstrap" / "phase_a_v3_best_primary_metric.pt",
        help="Phase-A checkpoint used to warm-start Phase B.",
    )
    parser.add_argument(
        "--persist-output-root",
        type=Path,
        default=None,
        help="Where to copy final artifacts. Defaults to <work-root>/exported_artifacts.",
    )
    parser.add_argument(
        "--skip-feature-cache",
        action="store_true",
        help="Reuse an existing feature cache in the same work root.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation and reuse the existing virtualenv.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare dataset/configs, do not start training.",
    )
    parser.add_argument(
        "--force-recreate-venv",
        action="store_true",
        help="Delete and recreate the virtualenv before installing.",
    )
    return parser.parse_args()


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _venv_pip(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "pip"


def _ensure_venv(args: argparse.Namespace) -> tuple[Path, Path]:
    venv_dir = args.venv.resolve()
    if args.force_recreate_venv and venv_dir.exists():
        shutil.rmtree(venv_dir)
    python_path = _venv_python(venv_dir)
    if not python_path.is_file():
        _run([args.python, "-m", "venv", str(venv_dir)])
    if not python_path.is_file():
        raise FileNotFoundError(f"Virtualenv python not found after creation: {python_path}")
    return python_path, _venv_pip(venv_dir)


def _install_runtime(args: argparse.Namespace, venv_python: Path, venv_pip: Path) -> None:
    _run([str(venv_pip), "install", "--upgrade", "pip"])
    if args.device.startswith("cuda"):
        _run(
            [
                str(venv_pip),
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ]
        )
        _run([str(venv_pip), "install", "-r", str(PROJECT_ROOT / "requirements.txt"), "--no-deps"])
        # MobileCLIP imports timm at module-load time. Install it with its small
        # runtime dependency set while keeping MobileCLIP's optional benchmark
        # stack out of the training environment.
        _run([str(venv_pip), "install", "timm>=0.9.5", "open-clip-torch>=2.20.0"])
    else:
        _run([str(venv_pip), "install", "-r", str(PROJECT_ROOT / "requirements.txt")])
    _run(
        [
            str(venv_python),
            str(PROJECT_ROOT / "scripts" / "setup_mobileclip.py"),
            "--destination",
            str(PROJECT_ROOT / "checkpoints" / "mobileclip_s0.pt"),
        ]
    )


def _verify_cuda(venv_python: Path) -> None:
    snippet = (
        "import torch; "
        "print(f'torch={torch.__version__}'); "
        "print(f'cuda_available={torch.cuda.is_available()}'); "
        "print(f'gpu={torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'gpu=none')"
    )
    _run([str(venv_python), "-c", snippet])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    bootstrap_checkpoint = args.bootstrap_checkpoint.resolve()
    if not bootstrap_checkpoint.is_file():
        raise FileNotFoundError(f"Bootstrap checkpoint does not exist: {bootstrap_checkpoint}")

    work_root = _ensure_directory(args.work_root)
    persist_output_root = _ensure_directory(
        args.persist_output_root if args.persist_output_root is not None else work_root / "exported_artifacts"
    )
    venv_python, venv_pip = _ensure_venv(args)

    print("\nRuntime configuration:", flush=True)
    print(f" dataset_root={dataset_root}", flush=True)
    print(f" work_root={work_root}", flush=True)
    print(f" bootstrap_checkpoint={bootstrap_checkpoint}", flush=True)
    print(f" persist_output_root={persist_output_root}", flush=True)
    print(f" venv={args.venv.resolve()}", flush=True)
    print(f" skip_feature_cache={args.skip_feature_cache}", flush=True)
    print(f" skip_install={args.skip_install}", flush=True)
    print(f" prepare_only={args.prepare_only}", flush=True)
    print(f" force_recreate_venv={args.force_recreate_venv}", flush=True)
    print(flush=True)

    if not args.skip_install:
        _install_runtime(args, venv_python, venv_pip)

    print("Python environment:", flush=True)
    _verify_cuda(venv_python)
    print(flush=True)

    command = [
        str(venv_python),
        str(PROJECT_ROOT / "scripts" / "run_phase_b_activitynet.py"),
        "--dataset-root",
        str(dataset_root),
        "--work-root",
        str(work_root),
        "--device",
        args.device,
        "--bootstrap-checkpoint",
        str(bootstrap_checkpoint),
        "--persist-output-root",
        str(persist_output_root),
    ]
    if args.skip_feature_cache:
        command.append("--skip-feature-cache")
    if args.prepare_only:
        command.append("--prepare-only")

    print("Starting TinyTrace Phase B local run:", flush=True)
    print(" " + " ".join(command), flush=True)
    print(flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    _run(command, env=env)


if __name__ == "__main__":
    main()

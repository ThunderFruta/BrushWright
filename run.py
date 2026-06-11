"""Guided BrushWright synthetic data runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import multiprocessing
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

from Source.PaintTransformerReference.synthesize_samples import (
    DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STROKE_WINDOW,
    SUPPORTED_IMAGE_SUFFIXES,
    build_paint_transformer_sample,
)
from Source.Output.output_archive import prepare_latest_output_root
from Source.Renderer.stroke_schema import DEFAULT_CANVAS_SIZE


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "ThirdParty" / "PaintTransformer" / "model.pth"
DEFAULT_MODEL_URL = "https://drive.google.com/uc?export=download&id=1NDD54BLligyr8tzo8QGI5eihZisXK1nq"
DEFAULT_IMAGE_DIR = ROOT_DIR / "Assets" / "ImageCorpus"
DEFAULT_DEMO_IMAGE = DEFAULT_IMAGE_DIR / "default_input.png"
DEFAULT_VENV = ROOT_DIR / ".venv"
DEFAULT_DEVICE = "cuda"
DEFAULT_RUN_BASE_STROKES = 192
DEFAULT_RUN_FINISHING_STROKES = 64
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_TRAIN_MICRO_BATCH_SIZE = 4
DEFAULT_TRAIN_WORKERS = 4
DEFAULT_VISUAL_DELTA_TRAIN_BATCH_SIZE = 4
DEFAULT_VISUAL_DELTA_TRAIN_WORKERS = 8
DEFAULT_VISUAL_DELTA_TRAIN_EPOCHS = 80
DEFAULT_TRAIN_CHECKPOINT_DIR = ROOT_DIR / "Models" / "Checkpoints" / "StrokePredictorV1TargetGuided"
DEFAULT_TRAIN_OVERFIT_CHECKPOINT_DIR = ROOT_DIR / "Models" / "Checkpoints" / "StrokePredictorV1TargetGuidedOverfit"
DEFAULT_VISUAL_DELTA_CHECKPOINT_DIR = ROOT_DIR / "Models" / "Checkpoints" / "VisualDeltaStrokeCompilerV8UsableV1Large"
DEFAULT_RECURSIVE_VISUAL_DELTA_OUTPUT_ROOT = ROOT_DIR / "Outputs" / "Latest" / "VisualDeltaPredictionsV8UsableV1LargeRecursive"
DEFAULT_TARGET_RETRIEVAL_OUTPUT_ROOT = ROOT_DIR / "Outputs" / "Latest" / "TargetStrokeRetrievalV6Oracle"
DEFAULT_IMAGE_DELTA_OUTPUT_ROOT = ROOT_DIR / "Outputs" / "Latest" / "ImageDeltaStrokeCompilerV1"
DEFAULT_GREEDY_OPTIMIZER_OUTPUT_ROOT = ROOT_DIR / "Outputs" / "Latest" / "GreedyStrokeOptimizerV1"
DEFAULT_WORKERS = 3
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args and _is_interactive():
        return _run_menu()
    return _run_generation_command(raw_args)


def _run_generation_command(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_CUDA_ALLOC_CONF)
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        run_config = _resolve_config(args)
        _ensure_runtime_dependencies(args)
        run_config["resolved_device"] = _resolve_runtime_device(str(run_config["device"]))
        _ensure_default_model(run_config)
        _print_run_summary(run_config)
        if not args.yes and _is_interactive():
            answer = input("Start generation? [Y/n]: ").strip().lower()
            if answer not in ("", "y", "yes"):
                print("Canceled.")
                return 0
        _run(run_config)
    except KeyboardInterrupt:
        print("run interrupted; workers stopped", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_menu() -> int:
    while True:
        print()
        print("BrushWright")
        print("  1. Generate Paint Transformer samples")
        print("  2. Prepare visual-delta Data/Train, Data/Val, Data/Test")
        print("  3. Train retired V1 target-guided stroke predictor")
        print("  4. Overfit retired V1 target-guided stroke predictor")
        print("  5. Export retired V1 test PNGs")
        print("  6. Train visual-delta stroke compiler V8 large model")
        print("  7. Overfit visual-delta V8 large model")
        print("  8. Export visual-delta test PNGs")
        print("  9. Export recursive visual-delta V8 test PNGs")
        print("  10. Export legacy greedy biggest-improving optimizer test PNGs")
        print("  11. Export legacy image-delta stroke compiler test PNGs")
        print("  12. Run tests")
        print("  13. Exit")
        choice = input("Choose an action [1]: ").strip() or "1"
        if choice == "1":
            return _run_generation_command(["--yes"])
        if choice == "2":
            if not _confirm("This rewrites Data/Train, Data/Val, and Data/Test. Continue?"):
                continue
            return _run_subprocess(
                [
                    str(_default_python()),
                    "-m",
                    "Source.Synthetic.prepare_train_val_test",
                    "--clear-existing",
                    "--use-output-detail-pair",
                ]
            )
        if choice == "3":
            return _run_subprocess(_training_command(overfit=False))
        if choice == "4":
            return _run_subprocess(_training_command(overfit=True))
        if choice == "5":
            return _run_subprocess(
                [
                    str(_default_python()),
                    "-m",
                    "Source.Model.export_test_predictions",
                    "--checkpoint",
                    str(DEFAULT_TRAIN_CHECKPOINT_DIR / "best.pt"),
                    "--limit",
                    "4",
                    "--device",
                    "auto",
                ]
            )
        if choice == "6":
            return _run_subprocess(_visual_delta_training_command(overfit=False))
        if choice == "7":
            return _run_subprocess(_visual_delta_training_command(overfit=True))
        if choice == "8":
            return _run_subprocess(
                [
                    str(_default_python()),
                    "-m",
                    "Source.Model.export_visual_delta_predictions",
                    "--limit",
                    "4",
                    "--device",
                    "auto",
                ]
            )
        if choice == "9":
            return _run_subprocess(_recursive_visual_delta_export_command())
        if choice == "10":
            return _run_subprocess(_greedy_optimizer_export_command())
        if choice == "11":
            return _run_subprocess(_image_delta_export_command())
        if choice == "12":
            return _run_subprocess([str(_default_python()), "-m", "unittest", "discover", "-s", "Tests"])
        if choice in ("13", "q", "quit", "exit"):
            print("Canceled.")
            return 0
        print(f"Unknown action: {choice}")


def _default_python() -> Path:
    venv_python = DEFAULT_VENV / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _training_command(overfit: bool) -> list[str]:
    command = [
        str(_default_python()),
        "-m",
        "Source.Model.train_strokes",
        "--batch-size",
        str(DEFAULT_TRAIN_BATCH_SIZE),
        "--micro-batch-size",
        str(DEFAULT_TRAIN_MICRO_BATCH_SIZE),
        "--num-workers",
        str(DEFAULT_TRAIN_WORKERS),
        "--max-base-strokes",
        str(DEFAULT_RUN_BASE_STROKES),
        "--image-input-channels",
        "9",
        "--decoder-query-mode",
        "learned",
        "--output-dir",
        str(DEFAULT_TRAIN_CHECKPOINT_DIR),
        "--device",
        "auto",
    ]
    if overfit:
        command = [
            str(_default_python()),
            "-m",
            "Source.Model.train_strokes",
            "--overfit-samples",
            "1",
            "--epochs",
            "50",
            "--batch-size",
            "8",
            "--micro-batch-size",
            "4",
            "--num-workers",
            str(DEFAULT_TRAIN_WORKERS),
            "--max-base-strokes",
            str(DEFAULT_RUN_BASE_STROKES),
            "--image-input-channels",
            "9",
            "--decoder-query-mode",
            "learned",
            "--device",
            "auto",
            "--train-repeat-factor",
            "100",
            "--visual-validation-samples",
            "1",
            "--visual-validation-interval",
            "1",
            "--output-dir",
            str(DEFAULT_TRAIN_OVERFIT_CHECKPOINT_DIR),
        ]
    resume_path = None if overfit else _default_resume_checkpoint()
    if resume_path is not None and _confirm(f"Resume from {resume_path}?"):
        command.extend(["--resume-checkpoint", str(resume_path)])
    return command


def _visual_delta_training_command(overfit: bool) -> list[str]:
    command = [
        str(_default_python()),
            "-m",
            "Source.Model.train_visual_delta_strokes",
            "--epochs",
            str(DEFAULT_VISUAL_DELTA_TRAIN_EPOCHS),
            "--batch-size",
            str(DEFAULT_VISUAL_DELTA_TRAIN_BATCH_SIZE),
            "--num-workers",
            str(DEFAULT_VISUAL_DELTA_TRAIN_WORKERS),
        "--device",
        "auto",
        "--output-dir",
        str(DEFAULT_VISUAL_DELTA_CHECKPOINT_DIR),
        "--training-renderer",
        "paint-transformer-soft",
        "--require-target-contract",
        "paint_transformer_original_image_target_v1",
    ]
    if overfit:
        command = [
            str(_default_python()),
            "-m",
            "Source.Model.train_visual_delta_strokes",
            "--overfit-samples",
            "1",
            "--epochs",
            "100",
            "--batch-size",
            "8",
            "--num-workers",
            str(DEFAULT_TRAIN_WORKERS),
            "--device",
            "auto",
            "--train-repeat-factor",
            "25",
            "--visual-validation-samples",
            "1",
            "--visual-validation-interval",
            "5",
            "--output-dir",
            str(DEFAULT_VISUAL_DELTA_CHECKPOINT_DIR),
            "--training-renderer",
            "paint-transformer-soft",
            "--require-target-contract",
            "paint_transformer_original_image_target_v1",
        ]
    return command


def _recursive_visual_delta_export_command() -> list[str]:
    return [
        str(_default_python()),
        "-m",
        "Source.Model.export_visual_delta_predictions",
        "--checkpoint",
        str(DEFAULT_VISUAL_DELTA_CHECKPOINT_DIR / "latest.pt"),
        "--output-root",
        str(DEFAULT_RECURSIVE_VISUAL_DELTA_OUTPUT_ROOT),
        "--split",
        "Test",
        "--limit",
        "4",
        "--device",
        "auto",
        "--recursive-passes",
        "6",
        "--strokes-per-pass",
        "512",
    ]


def _target_retrieval_export_command() -> list[str]:
    return [
        str(_default_python()),
        "-m",
        "Source.Model.export_target_stroke_retrieval",
        "--output-root",
        str(DEFAULT_TARGET_RETRIEVAL_OUTPUT_ROOT),
        "--split",
        "Test",
        "--limit",
        "4",
        "--recursive-passes",
        "6",
        "--strokes-per-pass",
        "256",
    ]


def _image_delta_export_command() -> list[str]:
    return [
        str(_default_python()),
        "-m",
        "Source.Model.export_image_delta_strokes",
        "--output-root",
        str(DEFAULT_IMAGE_DELTA_OUTPUT_ROOT),
        "--split",
        "Test",
        "--limit",
        "4",
        "--max-strokes",
        "512",
    ]


def _greedy_optimizer_export_command() -> list[str]:
    return [
        str(_default_python()),
        "-m",
        "Source.Model.export_greedy_stroke_optimizer",
        "--output-root",
        str(DEFAULT_GREEDY_OPTIMIZER_OUTPUT_ROOT),
        "--split",
        "Test",
        "--limit",
        "4",
        "--target-mode",
        "source-image",
    ]


def _default_resume_checkpoint() -> Path | None:
    step_checkpoint = DEFAULT_TRAIN_CHECKPOINT_DIR / "step_latest.pt"
    if step_checkpoint.exists() and _checkpoint_supports_image_conditioning(step_checkpoint):
        return step_checkpoint
    latest_checkpoint = DEFAULT_TRAIN_CHECKPOINT_DIR / "latest.pt"
    if latest_checkpoint.exists() and _checkpoint_supports_image_conditioning(latest_checkpoint):
        return latest_checkpoint
    return None


def _checkpoint_supports_image_conditioning(path: Path) -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False
    has_image_config = checkpoint.get("image_encoder_config") is not None
    if not has_image_config:
        print(f"Skipping old stroke-only checkpoint for resume: {path}", flush=True)
    return has_image_config


def _run_subprocess(command: Sequence[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(list(command), cwd=ROOT_DIR)
    return completed.returncode


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guided runner for Paint Transformer-backed BrushWright synthetic samples."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to a Paint Transformer .pth checkpoint. Default: {DEFAULT_MODEL_PATH}",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", type=Path, default=None, help="Single source image.")
    source_group.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=f"Directory of source images. Default when no source flag is passed: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for generated sample folders. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--base-count",
        type=int,
        default=DEFAULT_RUN_BASE_STROKES,
        help=f"Number of base strokes to keep in each draft. Default: {DEFAULT_RUN_BASE_STROKES}",
    )
    parser.add_argument(
        "--finishing-count",
        type=int,
        default=DEFAULT_RUN_FINISHING_STROKES,
        help=f"Number of withheld finishing strokes. Default: {DEFAULT_RUN_FINISHING_STROKES}",
    )
    parser.add_argument(
        "--stroke-window",
        choices=("start", "detail"),
        default=DEFAULT_STROKE_WINDOW,
        help=f"Which ordered Paint Transformer strokes to export. Default: {DEFAULT_STROKE_WINDOW}",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=DEFAULT_CANVAS_SIZE,
        help=f"Square canvas size for Paint Transformer input. Default: {DEFAULT_CANVAS_SIZE}",
    )
    parser.add_argument(
        "--draft-image-completion-ratio",
        type=float,
        default=DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
        help=f"Native Paint Transformer frame ratio used for draft.png. Default: {DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO}",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default=DEFAULT_DEVICE,
        help=f"Torch inference device. Default: {DEFAULT_DEVICE}",
    )
    parser.add_argument(
        "--no-venv-bootstrap",
        action="store_true",
        help="Do not create/use the repo-local .venv when dependencies are missing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process from --image-dir. Default: all images",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel sample-generation worker processes. Default: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample folders that already contain sample.json, useful for resuming failed runs.",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt.")
    return parser


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    used_default_model_path = args.model_path == DEFAULT_MODEL_PATH
    used_default_image_dir = args.image is None and args.image_dir is None
    model_path = _resolve_path_arg(
        value=args.model_path,
        prompt="Paint Transformer checkpoint path",
        default=DEFAULT_MODEL_PATH,
        must_exist=False,
    )
    image_path = args.image
    image_dir = args.image_dir
    if image_path is None and image_dir is None:
        image_path = DEFAULT_DEMO_IMAGE
        if not image_path.exists():
            _write_default_demo_image(image_path)
    elif image_path is not None:
        image_path = image_path.expanduser()
        if not image_path.exists():
            raise OSError(f"image does not exist: {image_path}")
    elif image_dir is not None:
        image_dir = image_dir.expanduser()
        if not image_dir.exists():
            raise OSError(f"image directory does not exist: {image_dir}")

    output_root = args.output_root
    base_count = _resolve_int_arg(args.base_count, "Base stroke count", DEFAULT_RUN_BASE_STROKES, minimum=1)
    finishing_count = _resolve_int_arg(
        args.finishing_count,
        "Finishing stroke count",
        DEFAULT_RUN_FINISHING_STROKES,
        minimum=1,
    )
    canvas_size = _resolve_int_arg(args.canvas_size, "Canvas size", DEFAULT_CANVAS_SIZE, minimum=1)
    draft_image_completion_ratio = _resolve_float_arg(
        args.draft_image_completion_ratio,
        "Draft image completion ratio",
        DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
        minimum=0.0,
        maximum=1.0,
    )
    workers = _resolve_int_arg(args.workers, "Worker count", DEFAULT_WORKERS, minimum=1)
    limit = _resolve_optional_limit(args.limit)

    image_paths = _collect_image_paths(image_path=image_path, image_dir=image_dir, limit=limit)
    if not image_paths:
        source = image_path or image_dir
        raise ValueError(f"no supported source images found in {source}")

    return {
        "model_path": model_path,
        "used_default_model_path": used_default_model_path,
        "image_paths": image_paths,
        "output_root": output_root.expanduser(),
        "base_count": base_count,
        "finishing_count": finishing_count,
        "stroke_window": args.stroke_window,
        "canvas_size": canvas_size,
        "draft_image_completion_ratio": draft_image_completion_ratio,
        "device": args.device,
        "workers": workers,
        "skip_existing": args.skip_existing,
    }


def _run(config: dict[str, object]) -> None:
    output_root = Path(config["output_root"])
    if not bool(config["skip_existing"]):
        output_root = prepare_latest_output_root(output_root)
    image_paths = list(config["image_paths"])
    total = len(image_paths)
    workers = min(int(config["workers"]), total)
    jobs = []
    skipped = 0
    for index, image_path in enumerate(image_paths, start=1):
        sample_id = f"paint_transformer_{index:06d}"
        output_dir = output_root / sample_id
        if bool(config["skip_existing"]) and (output_dir / "sample.json").exists():
            skipped += 1
            print(f"[{index}/{total}] skip existing {output_dir}", flush=True)
            continue
        jobs.append(
            {
                "index": index,
                "total": total,
                "image_path": str(Path(image_path)),
                "model_path": str(Path(config["model_path"])),
                "output_dir": str(output_dir),
                "sample_id": sample_id,
                "base_count": int(config["base_count"]),
                "finishing_count": int(config["finishing_count"]),
                "stroke_window": str(config["stroke_window"]),
                "canvas_size": int(config["canvas_size"]),
                "draft_image_completion_ratio": float(config["draft_image_completion_ratio"]),
                "device_name": str(config["resolved_device"]),
            }
        )

    if not jobs:
        print(f"No new samples to generate; skipped {skipped} existing sample(s) under {output_root}")
        return

    workers = min(workers, len(jobs))
    if workers == 1:
        for job in jobs:
            print(f"[{job['index']}/{total}] {job['image_path']} -> {job['output_dir']}", flush=True)
            _build_sample_worker(job)
    else:
        print(f"Running with {workers} worker processes", flush=True)
        spawn_context = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=spawn_context)
        try:
            futures = {executor.submit(_build_sample_worker, job): job for job in jobs}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    _terminate_executor_workers(executor)
                    raise RuntimeError(f"worker failed for {job['image_path']}: {exc}") from exc
                completed += 1
                print(f"[{completed}/{total}] done {result['image_path']} -> {result['output_dir']}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted; terminating worker processes...", file=sys.stderr, flush=True)
            _terminate_executor_workers(executor)
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    if skipped:
        print(f"Wrote {len(jobs)} new sample(s), skipped {skipped} existing sample(s) under {output_root}")
    else:
        print(f"Wrote {len(jobs)} sample(s) under {output_root}")


def _terminate_executor_workers(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    processes_by_pid = getattr(executor, "_processes", None) or {}
    processes = list(processes_by_pid.values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=2)


def _build_sample_worker(job: dict[str, object]) -> dict[str, object]:
    build_paint_transformer_sample(
        image_path=Path(str(job["image_path"])),
        model_path=Path(str(job["model_path"])),
        output_dir=Path(str(job["output_dir"])),
        sample_id=str(job["sample_id"]),
        base_count=int(job["base_count"]),
        finishing_count=int(job["finishing_count"]),
        stroke_window=str(job["stroke_window"]),
        draft_image_completion_ratio=float(job["draft_image_completion_ratio"]),
        canvas_size=int(job["canvas_size"]),
        device_name=str(job["device_name"]),
    )
    return {
        "index": int(job["index"]),
        "image_path": str(job["image_path"]),
        "output_dir": str(job["output_dir"]),
    }


def _ensure_runtime_dependencies(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("torch") is not None:
        return
    if args.no_venv_bootstrap:
        _raise_torch_missing()
    if _inside_default_venv():
        _raise_torch_missing()
    if not _is_interactive() and not args.yes:
        raise RuntimeError("PyTorch is missing. Re-run with --yes to allow .venv bootstrap, or use --no-venv-bootstrap.")

    print(f"PyTorch is missing for {sys.executable}; preparing repo-local environment at {DEFAULT_VENV}")
    _ensure_venv(DEFAULT_VENV)
    _install_venv_dependencies(DEFAULT_VENV)
    _reexec_in_venv(DEFAULT_VENV)


def _resolve_runtime_device(requested_device: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required before resolving the inference device") from exc

    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        print("CUDA is not available; falling back to CPU because --device auto was used.")
        return "cpu"
    if requested_device == "cuda":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA GPU: {device_name}")
            return "cuda"
        raise RuntimeError(
            "GPU inference was requested by default, but CUDA is not available to PyTorch. "
            "`nvidia-smi` currently cannot communicate with the NVIDIA driver and /dev/nvidia* is not present. "
            "Fix the NVIDIA driver/device exposure, or run with `--device cpu`."
        )
    return requested_device


def _raise_torch_missing() -> None:
    raise RuntimeError(
        "PyTorch is required for Paint Transformer inference. Install it for this Python first: "
        f"{sys.executable} -m pip install torch"
    )


def _inside_default_venv() -> bool:
    return Path(sys.prefix).resolve() == DEFAULT_VENV.resolve()


def _ensure_venv(venv_path: Path) -> None:
    python_path = venv_path / "bin" / "python"
    if python_path.exists():
        return
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], cwd=ROOT_DIR, check=True)


def _install_venv_dependencies(venv_path: Path) -> None:
    python_path = venv_path / "bin" / "python"
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=ROOT_DIR,
        check=True,
    )
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "torch", "pillow", "numpy"],
        cwd=ROOT_DIR,
        check=True,
    )


def _reexec_in_venv(venv_path: Path) -> None:
    python_path = venv_path / "bin" / "python"
    print(f"Restarting with {python_path}")
    os.execv(str(python_path), [str(python_path), str(Path(__file__).resolve()), *sys.argv[1:]])


def _ensure_default_model(config: dict[str, object]) -> None:
    model_path = Path(config["model_path"])
    if model_path.exists():
        return
    if not config["used_default_model_path"]:
        raise OSError(f"paint transformer checkpoint path does not exist: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Default checkpoint missing; downloading to {model_path}")
    _download_file(DEFAULT_MODEL_URL, model_path)


def _download_file(url: str, output_path: Path) -> None:
    try:
        data, content_type = _read_url(url)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not download default checkpoint from {url}: {exc}") from exc

    if b"Google Drive - Virus scan warning" in data or b"download_warning" in data:
        confirm_url = _extract_google_drive_confirm_url(data)
        if confirm_url is None:
            raise RuntimeError(
                "Google Drive requires a browser confirmation for this checkpoint. Download it manually from "
                "https://drive.google.com/file/d/1NDD54BLligyr8tzo8QGI5eihZisXK1nq/view and save it to "
                f"{output_path}"
            )
        try:
            data, content_type = _read_url(confirm_url)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not download confirmed checkpoint from {confirm_url}: {exc}") from exc
    if "text/html" in content_type and len(data) < 10_000_000:
        raise RuntimeError(
            "checkpoint download returned an HTML page instead of model weights. Download it manually from "
            "https://drive.google.com/file/d/1NDD54BLligyr8tzo8QGI5eihZisXK1nq/view and save it to "
            f"{output_path}"
        )

    output_path.write_bytes(data)
    if output_path.stat().st_size < 1_000_000:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("downloaded checkpoint was unexpectedly small; refusing to use it")


def _read_url(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "BrushWright/0.1"})
    with urllib.request.urlopen(request) as response:
        return response.read(), response.headers.get("Content-Type", "")


def _extract_google_drive_confirm_url(data: bytes) -> str | None:
    html = data.decode("utf-8", errors="replace")
    action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    if action_match is None:
        return None
    action = action_match.group(1).replace("&amp;", "&")
    inputs = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html))
    if not inputs:
        return None
    return action + "?" + urllib.parse.urlencode(inputs)


def _print_run_summary(config: dict[str, object]) -> None:
    image_paths = list(config["image_paths"])
    print("BrushWright Paint Transformer synthetic run")
    print(f"  model: {config['model_path']}")
    print(f"  images: {len(image_paths)}")
    if len(image_paths) <= 5:
        for image_path in image_paths:
            print(f"    - {image_path}")
    else:
        print(f"    - {image_paths[0]}")
        print(f"    - ...")
        print(f"    - {image_paths[-1]}")
    print(f"  output: {config['output_root']}")
    print(f"  base strokes: {config['base_count']}")
    print(f"  finishing strokes: {config['finishing_count']}")
    print(f"  stroke window: {config['stroke_window']}")
    print(f"  canvas size: {config['canvas_size']}")
    print(f"  draft image completion ratio: {config['draft_image_completion_ratio']}")
    print(f"  workers: {config['workers']}")
    print(f"  skip existing: {config['skip_existing']}")
    print(f"  cuda alloc conf: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    print(f"  requested device: {config['device']}")
    print(f"  resolved device: {config['resolved_device']}")


def _resolve_path_arg(value: Path | None, prompt: str, default: Path | None, must_exist: bool) -> Path:
    if value is not None:
        path = value.expanduser()
        if must_exist and not path.exists():
            raise OSError(f"{prompt.lower()} does not exist: {path}")
        return path
    return _prompt_path(prompt, default, must_exist=must_exist)


def _prompt_path(prompt: str, default: Path | None, must_exist: bool) -> Path:
    if not _is_interactive():
        if default is None:
            raise ValueError(f"--{prompt.lower().replace(' ', '-')} is required")
        path = default
    else:
        suffix = f" [{default}]" if default is not None else ""
        raw_value = input(f"{prompt}{suffix}: ").strip()
        if not raw_value and default is None:
            raise ValueError(f"{prompt} is required")
        path = Path(raw_value) if raw_value else default
    if path is None:
        raise ValueError(f"{prompt} is required")
    path = path.expanduser()
    if must_exist and not path.exists():
        raise OSError(f"{prompt.lower()} does not exist: {path}")
    return path


def _write_default_demo_image(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to create the default demo image") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1680, 945
    image = Image.new("RGB", (width, height), (55, 139, 196))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / (height - 1)
        if y < 430:
            red = int(48 + 20 * t)
            green = int(138 + 32 * t)
            blue = int(198 + 34 * t)
        else:
            water_t = (y - 430) / (height - 430)
            red = int(28 + 95 * water_t)
            green = int(130 + 70 * water_t)
            blue = int(158 + 40 * (1 - water_t))
        draw.line((0, y, width, y), fill=(red, green, blue))

    mountains = [
        [(0, 444), (95, 405), (185, 434), (280, 398), (420, 438), (560, 406), (700, 448)],
        [(610, 448), (760, 410), (930, 438), (1070, 402), (1240, 446), (1390, 408), (1680, 432)],
    ]
    for ridge in mountains:
        draw.polygon(ridge + [(ridge[-1][0], 470), (ridge[0][0], 470)], fill=(50, 83, 104))
        snow = [(x, y - 10) for x, y in ridge[1:-1:2]]
        for x, y in snow:
            draw.polygon([(x - 55, y + 20), (x, y), (x + 62, y + 22), (x + 12, y + 14)], fill=(228, 236, 238))

    draw.rectangle((0, 448, width, 470), fill=(25, 90, 120))
    draw.rectangle((0, 470, width, 945), fill=(42, 158, 173))
    for y in range(470, height):
        t = (y - 470) / (height - 470)
        draw.line((0, y, width, y), fill=(int(24 + 112 * t), int(138 + 70 * t), int(162 + 10 * (1 - t))))

    rocks = [
        (360, 625, 280, 110, (178, 186, 184)),
        (690, 535, 160, 80, (166, 176, 176)),
        (870, 520, 190, 105, (96, 111, 116)),
        (995, 605, 180, 80, (184, 190, 185)),
        (1280, 545, 135, 78, (184, 190, 185)),
        (1510, 585, 105, 55, (185, 190, 184)),
        (1200, 885, 760, 280, (83, 93, 91)),
    ]
    for cx, cy, rx, ry, color in rocks:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
        draw.arc((cx - rx, cy - ry, cx + rx, cy + ry), 185, 340, fill=(42, 54, 58), width=max(3, ry // 12))
        draw.ellipse((cx - rx // 3, cy - ry // 2, cx + rx // 4, cy), fill=tuple(min(255, c + 25) for c in color))

    for x, y, r in [(230, 505, 42), (450, 500, 38), (600, 485, 22), (790, 472, 30), (1350, 618, 62)]:
        draw.ellipse((x - r, y - r // 2, x + r, y + r // 2), fill=(150, 158, 154))

    tree_x = 1475
    draw.line((tree_x, 420, tree_x + 8, 280), fill=(72, 55, 38), width=12)
    for cx, cy, rx, ry in [(1460, 315, 38, 74), (1495, 300, 42, 82), (1446, 370, 50, 54), (1514, 365, 45, 58)]:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(42, 105, 74))

    for x, y, w in [(80, 700, 180), (500, 760, 240), (1000, 705, 160), (1325, 680, 190)]:
        draw.arc((x, y, x + w, y + 28), 180, 360, fill=(30, 108, 145), width=4)

    image.save(path)
    print(f"No images found in {path.parent}; wrote default source image {path}")


def _resolve_int_arg(value: int | None, prompt: str, default: int, minimum: int) -> int:
    if value is None and _is_interactive():
        raw_value = input(f"{prompt} [{default}]: ").strip()
        value = int(raw_value) if raw_value else default
    elif value is None:
        value = default
    if value < minimum:
        raise ValueError(f"{prompt.lower()} must be at least {minimum}")
    return value


def _resolve_float_arg(value: float | None, prompt: str, default: float, minimum: float, maximum: float) -> float:
    if value is None and _is_interactive():
        raw_value = input(f"{prompt} [{default}]: ").strip()
        value = float(raw_value) if raw_value else default
    elif value is None:
        value = default
    if value <= minimum or value > maximum:
        raise ValueError(f"{prompt.lower()} must be > {minimum} and <= {maximum}")
    return value


def _resolve_optional_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if limit <= 0:
        raise ValueError("limit must be positive")
    return limit


def _collect_image_paths(image_path: Path | None, image_dir: Path | None, limit: int | None) -> list[Path]:
    if image_path is not None:
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {image_path.suffix}")
        return [image_path]
    if image_dir is None:
        raise ValueError("image or image-dir is required")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def _is_interactive() -> bool:
    return sys.stdin.isatty()


if __name__ == "__main__":
    raise SystemExit(main())

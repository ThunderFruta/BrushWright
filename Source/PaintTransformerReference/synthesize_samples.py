"""Build BrushWright synthetic samples from Paint Transformer inference."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from Source.Renderer.stroke_schema import DEFAULT_CANVAS_SIZE, STROKE_PROGRAM_VERSION, load_stroke_program_json

from Source.Output.output_archive import prepare_latest_output_root
from Source.PaintTransformerReference.strokes import collect_brushwright_strokes


DEFAULT_OUTPUT_ROOT = Path("Outputs/Latest/PaintTransformerSamples")
DEFAULT_PATCH_SIZE = 32
DEFAULT_STROKE_NUM = 8
DEFAULT_HIDDEN_DIM = 256
DEFAULT_BASE_STROKES = 192
DEFAULT_FINISHING_STROKES = 64
DEFAULT_STROKE_WINDOW = "detail"
DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO = 3.0 / 5.0
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate BrushWright samples with a Paint Transformer checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-count", type=int, default=DEFAULT_BASE_STROKES)
    parser.add_argument("--finishing-count", type=int, default=DEFAULT_FINISHING_STROKES)
    parser.add_argument("--stroke-window", choices=("start", "detail"), default=DEFAULT_STROKE_WINDOW)
    parser.add_argument("--draft-image-completion-ratio", type=float, default=DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO)
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    image_paths = _resolve_image_paths(image_path=args.image, image_dir=args.image_dir)
    output_root = prepare_latest_output_root(args.output_root)
    for index, image_path in enumerate(image_paths, start=1):
        sample_id = f"paint_transformer_{index:06d}"
        build_paint_transformer_sample(
            image_path=image_path,
            model_path=args.model_path,
            output_dir=output_root / sample_id,
            sample_id=sample_id,
            base_count=args.base_count,
            finishing_count=args.finishing_count,
            stroke_window=args.stroke_window,
            draft_image_completion_ratio=args.draft_image_completion_ratio,
            canvas_size=args.canvas_size,
            device_name=args.device,
        )
    return 0


def build_paint_transformer_sample(
    image_path: Path,
    model_path: Path,
    output_dir: Path,
    sample_id: str,
    base_count: int = DEFAULT_BASE_STROKES,
    finishing_count: int = DEFAULT_FINISHING_STROKES,
    stroke_window: str = DEFAULT_STROKE_WINDOW,
    draft_image_completion_ratio: float = DEFAULT_DRAFT_IMAGE_COMPLETION_RATIO,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    device_name: str | None = None,
) -> dict[str, Any]:
    if base_count <= 0:
        raise ValueError("base_count must be positive")
    if finishing_count <= 0:
        raise ValueError("finishing_count must be positive")
    if not 0.0 < draft_image_completion_ratio <= 1.0:
        raise ValueError("draft_image_completion_ratio must be between 0 and 1")

    image_path = image_path.resolve()
    model_path = model_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strokes = infer_brushwright_strokes(
        image_path=image_path,
        model_path=model_path,
        canvas_size=canvas_size,
        device_name=device_name,
    )
    requested_base_count = base_count
    requested_finishing_count = finishing_count
    requested_count = requested_base_count + requested_finishing_count
    base_count, finishing_count, adjusted_counts = _resolve_available_stroke_split(
        available_count=len(strokes),
        requested_base_count=requested_base_count,
        requested_finishing_count=requested_finishing_count,
    )
    required_count = base_count + finishing_count
    if adjusted_counts:
        print(
            "Paint Transformer produced "
            f"{len(strokes)} strokes for {image_path.name}; using "
            f"{base_count} base + {finishing_count} finishing instead of "
            f"{requested_base_count} base + {requested_finishing_count} finishing.",
            flush=True,
        )
    selected_strokes, selected_start_index = _select_stroke_window(strokes, required_count, stroke_window)
    native_render = render_source_image_with_paint_transformer(
        image_path=image_path,
        model_path=model_path,
        canvas_size=canvas_size,
        device_name=device_name,
    )
    native_draft_index = _native_frame_index_for_ratio(native_render["frame_count"], draft_image_completion_ratio)
    base_strokes = selected_strokes[:base_count]
    finishing_strokes = selected_strokes[base_count:required_count]

    metadata = {
        "generator": "Source.PaintTransformerReference.synthesize_samples",
        "source_image": str(image_path),
        "source_model": str(model_path),
        "source": "Huage001/PaintTransformer",
        "source_license": "Apache-2.0",
        "stroke_count": required_count,
        "base_count": base_count,
        "finishing_count": finishing_count,
        "requested_stroke_count": requested_count,
        "requested_base_count": requested_base_count,
        "requested_finishing_count": requested_finishing_count,
        "stroke_count_adjusted": adjusted_counts,
        "stroke_window": stroke_window,
        "selected_start_index": selected_start_index,
        "draft_image_completion_ratio": draft_image_completion_ratio,
        "render_source": "paint_transformer_native_inference",
    }
    full_program = _program(canvas_size, metadata, selected_strokes)
    base_program = _program(canvas_size, {**metadata, "split": "base", "stroke_count": base_count}, base_strokes)
    finishing_program = _program(
        canvas_size,
        {**metadata, "split": "finishing", "stroke_count": finishing_count},
        finishing_strokes,
    )
    available_program = _program(
        canvas_size,
        {
            **metadata,
            "split": "available",
            "stroke_count": len(strokes),
            "base_count": None,
            "finishing_count": None,
            "selected_start_index": 0,
            "selected_end_index_exclusive": len(strokes),
        },
        strokes,
    )
    load_stroke_program_json(full_program)
    load_stroke_program_json(base_program)
    load_stroke_program_json(finishing_program)
    load_stroke_program_json(available_program)

    full_program_path = output_dir / "full_program.json"
    base_path = output_dir / "base_strokes.json"
    finishing_path = output_dir / "finishing_strokes.json"
    available_path = output_dir / "available_strokes.json"
    split_manifest_path = output_dir / "split_manifest.json"
    _write_json(full_program_path, full_program)
    _write_json(base_path, base_program)
    _write_json(finishing_path, finishing_program)
    _write_json(available_path, available_program)
    _write_json(
        split_manifest_path,
        {
            "version": 1,
            "method": "paint_transformer_order",
            "source_image": str(image_path),
            "total_available_strokes": len(strokes),
            "total_strokes": required_count,
            "requested_total_strokes": requested_count,
            "requested_base_count": requested_base_count,
            "requested_finishing_count": requested_finishing_count,
            "stroke_count_adjusted": adjusted_counts,
            "stroke_window": stroke_window,
            "selected_start_index": selected_start_index,
            "base_count": base_count,
            "finishing_count": finishing_count,
            "withheld_start_index": base_count,
            "withheld_end_index_exclusive": required_count,
            "available_strokes": _relative_to_sample(output_dir, available_path),
            "draft_image_completion_ratio": draft_image_completion_ratio,
        },
    )

    draft_render_dir = output_dir / "draft_render"
    finished_render_dir = output_dir / "finished_render"
    _write_native_render_artifacts(native_render, draft_render_dir, frame_index=native_draft_index)
    _write_native_render_artifacts(native_render, finished_render_dir, frame_index=native_render["frame_count"] - 1)
    draft_image = output_dir / "draft.png"
    finished_image = output_dir / "finished.png"
    shutil.copyfile(draft_render_dir / "final.png", draft_image)
    shutil.copyfile(finished_render_dir / "final.png", finished_image)

    sample = {
        "version": 1,
        "sample_id": sample_id,
        "source_image": str(image_path),
        "canvas": full_program["canvas"],
        "stroke_count": required_count,
        "base_count": base_count,
        "finishing_count": finishing_count,
        "stroke_window": stroke_window,
        "draft_image_completion_ratio": draft_image_completion_ratio,
        "generator": "paint_transformer_reference",
        "full_program": _relative_to_sample(output_dir, full_program_path),
        "base_strokes": _relative_to_sample(output_dir, base_path),
        "finishing_strokes": _relative_to_sample(output_dir, finishing_path),
        "available_strokes": _relative_to_sample(output_dir, available_path),
        "draft_image": _relative_to_sample(output_dir, draft_image),
        "finished_image": _relative_to_sample(output_dir, finished_image),
        "draft_render_manifest": _relative_to_sample(output_dir, draft_render_dir / "render_manifest.json"),
        "finished_render_manifest": _relative_to_sample(output_dir, finished_render_dir / "render_manifest.json"),
        "split_manifest": _relative_to_sample(output_dir, split_manifest_path),
    }
    _write_json(output_dir / "sample.json", sample)
    return sample


def infer_brushwright_strokes(
    image_path: Path,
    model_path: Path,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    device_name: str | None = None,
) -> list[dict[str, Any]]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Paint Transformer sample generation requires PyTorch") from exc

    from Source.PaintTransformerReference.model import Painter, SignWithSigmoidGrad
    from Source.PaintTransformerReference.rendering import crop, load_meta_brushes, pad, param2img_parallel, read_img

    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if not model_path.exists():
        raise OSError(f"Paint Transformer checkpoint does not exist: {model_path}")

    device = _resolve_torch_device(torch, device_name)
    net_g = Painter(5, DEFAULT_STROKE_NUM, DEFAULT_HIDDEN_DIM, 8, 3, 3).to(device)
    net_g.load_state_dict(torch.load(model_path, map_location=device))
    net_g.eval()
    for param in net_g.parameters():
        param.requires_grad = False

    meta_brushes = load_meta_brushes(device)
    collected_strokes: list[dict[str, Any]] = []
    with torch.no_grad():
        original_img = read_img(image_path, "RGB", canvas_size, canvas_size).to(device)
        original_h, original_w = original_img.shape[-2:]
        level_count = max(math.ceil(math.log2(max(original_h, original_w) / DEFAULT_PATCH_SIZE)), 0)
        padded_size = DEFAULT_PATCH_SIZE * (2 ** level_count)
        original_img_pad = pad(original_img, padded_size, padded_size)
        final_result = torch.zeros_like(original_img_pad).to(device)
        last_patch_num = 1

        for layer in range(0, level_count + 1):
            layer_size = DEFAULT_PATCH_SIZE * (2 ** layer)
            img = F.interpolate(original_img_pad, (layer_size, layer_size))
            result = F.interpolate(final_result, (layer_size, layer_size))
            img_patch = F.unfold(img, (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE), stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE))
            result_patch = F.unfold(
                result,
                (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
                stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
            )
            patch_num = (layer_size - DEFAULT_PATCH_SIZE) // DEFAULT_PATCH_SIZE + 1
            last_patch_num = patch_num
            img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE)
            result_patch = result_patch.permute(0, 2, 1).contiguous().view(
                -1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE
            )
            shape_param, stroke_decision = net_g(img_patch, result_patch)
            stroke_decision = SignWithSigmoidGrad.apply(stroke_decision)
            stroke_param = _attach_sampled_colors(torch, F, img_patch, shape_param)
            param = stroke_param.view(1, patch_num, patch_num, DEFAULT_STROKE_NUM, 8).contiguous()
            decision = stroke_decision.view(1, patch_num, patch_num, DEFAULT_STROKE_NUM).contiguous().bool()
            param[..., :2] = param[..., :2] / 2 + 0.25
            param[..., 2:4] = param[..., 2:4] / 2
            collected_strokes.extend(collect_brushwright_strokes(param, decision, patch_count=patch_num))
            final_result = param2img_parallel(param, decision, meta_brushes, final_result)

        border_size = padded_size // (2 * last_patch_num)
        img = F.interpolate(original_img_pad, (DEFAULT_PATCH_SIZE * (2 ** layer), DEFAULT_PATCH_SIZE * (2 ** layer)))
        result = F.interpolate(final_result, (DEFAULT_PATCH_SIZE * (2 ** layer), DEFAULT_PATCH_SIZE * (2 ** layer)))
        img = F.pad(img, [DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2])
        result = F.pad(
            result,
            [DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2],
        )
        img_patch = F.unfold(img, (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE), stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE))
        result_patch = F.unfold(
            result,
            (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
            stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
        )
        final_result = F.pad(final_result, [border_size, border_size, border_size, border_size, 0, 0, 0, 0])
        patch_h = (img.shape[2] - DEFAULT_PATCH_SIZE) // DEFAULT_PATCH_SIZE + 1
        patch_w = (img.shape[3] - DEFAULT_PATCH_SIZE) // DEFAULT_PATCH_SIZE + 1
        img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE)
        result_patch = result_patch.permute(0, 2, 1).contiguous().view(-1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE)
        shape_param, stroke_decision = net_g(img_patch, result_patch)
        stroke_decision = SignWithSigmoidGrad.apply(stroke_decision)
        stroke_param = _attach_sampled_colors(torch, F, img_patch, shape_param)
        param = stroke_param.view(1, patch_h, patch_w, DEFAULT_STROKE_NUM, 8).contiguous()
        decision = stroke_decision.view(1, patch_h, patch_w, DEFAULT_STROKE_NUM).contiguous().bool()
        param[..., :2] = param[..., :2] / 2 + 0.25
        param[..., 2:4] = param[..., 2:4] / 2
        collected_strokes.extend(collect_brushwright_strokes(param, decision, patch_count=last_patch_num, offset_x=-0.5, offset_y=-0.5))
        final_result = param2img_parallel(param, decision, meta_brushes, final_result)
        final_result = final_result[:, :, border_size:-border_size, border_size:-border_size]
        crop(final_result, original_h, original_w)

    return collected_strokes


def render_source_image_with_paint_transformer(
    image_path: Path,
    model_path: Path,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    device_name: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Paint Transformer rendering requires PyTorch") from exc

    from Source.PaintTransformerReference.model import Painter, SignWithSigmoidGrad
    from Source.PaintTransformerReference.rendering import crop, load_meta_brushes, pad, param2img_parallel, read_img

    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if not model_path.exists():
        raise OSError(f"Paint Transformer checkpoint does not exist: {model_path}")

    device = _resolve_torch_device(torch, device_name)
    net_g = Painter(5, DEFAULT_STROKE_NUM, DEFAULT_HIDDEN_DIM, 8, 3, 3).to(device)
    net_g.load_state_dict(torch.load(model_path, map_location=device))
    net_g.eval()
    for param in net_g.parameters():
        param.requires_grad = False

    meta_brushes = load_meta_brushes(device)
    frames = []
    with torch.no_grad():
        original_img = read_img(image_path, "RGB", canvas_size, canvas_size).to(device)
        original_h, original_w = original_img.shape[-2:]
        level_count = max(math.ceil(math.log2(max(original_h, original_w) / DEFAULT_PATCH_SIZE)), 0)
        padded_size = DEFAULT_PATCH_SIZE * (2 ** level_count)
        original_img_pad = pad(original_img, padded_size, padded_size)
        final_result = torch.zeros_like(original_img_pad).to(device)
        last_patch_num = 1

        for layer in range(0, level_count + 1):
            layer_size = DEFAULT_PATCH_SIZE * (2 ** layer)
            img = F.interpolate(original_img_pad, (layer_size, layer_size))
            result = F.interpolate(final_result, (layer_size, layer_size))
            img_patch = F.unfold(img, (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE), stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE))
            result_patch = F.unfold(
                result,
                (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
                stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
            )
            patch_num = (layer_size - DEFAULT_PATCH_SIZE) // DEFAULT_PATCH_SIZE + 1
            last_patch_num = patch_num
            img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE)
            result_patch = result_patch.permute(0, 2, 1).contiguous().view(
                -1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE
            )
            shape_param, stroke_decision = net_g(img_patch, result_patch)
            stroke_decision = SignWithSigmoidGrad.apply(stroke_decision)
            stroke_param = _attach_sampled_colors(torch, F, img_patch, shape_param)
            param = stroke_param.view(1, patch_num, patch_num, DEFAULT_STROKE_NUM, 8).contiguous()
            decision = stroke_decision.view(1, patch_num, patch_num, DEFAULT_STROKE_NUM).contiguous().bool()
            param[..., :2] = param[..., :2] / 2 + 0.25
            param[..., 2:4] = param[..., 2:4] / 2
            final_result = param2img_parallel(param, decision, meta_brushes, final_result)
            frames.append(crop(final_result, original_h, original_w).detach().cpu())

        border_size = padded_size // (2 * last_patch_num)
        img = F.interpolate(original_img_pad, (DEFAULT_PATCH_SIZE * (2 ** layer), DEFAULT_PATCH_SIZE * (2 ** layer)))
        result = F.interpolate(final_result, (DEFAULT_PATCH_SIZE * (2 ** layer), DEFAULT_PATCH_SIZE * (2 ** layer)))
        img = F.pad(img, [DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2])
        result = F.pad(
            result,
            [DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2, DEFAULT_PATCH_SIZE // 2],
        )
        img_patch = F.unfold(img, (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE), stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE))
        result_patch = F.unfold(
            result,
            (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
            stride=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
        )
        final_result = F.pad(final_result, [border_size, border_size, border_size, border_size, 0, 0, 0, 0])
        patch_h = (img.shape[2] - DEFAULT_PATCH_SIZE) // DEFAULT_PATCH_SIZE + 1
        patch_w = (img.shape[3] - DEFAULT_PATCH_SIZE) // DEFAULT_PATCH_SIZE + 1
        img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE)
        result_patch = result_patch.permute(0, 2, 1).contiguous().view(-1, 3, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE)
        shape_param, stroke_decision = net_g(img_patch, result_patch)
        stroke_decision = SignWithSigmoidGrad.apply(stroke_decision)
        stroke_param = _attach_sampled_colors(torch, F, img_patch, shape_param)
        param = stroke_param.view(1, patch_h, patch_w, DEFAULT_STROKE_NUM, 8).contiguous()
        decision = stroke_decision.view(1, patch_h, patch_w, DEFAULT_STROKE_NUM).contiguous().bool()
        param[..., :2] = param[..., :2] / 2 + 0.25
        param[..., 2:4] = param[..., 2:4] / 2
        final_result = param2img_parallel(param, decision, meta_brushes, final_result)
        final_result = final_result[:, :, border_size:-border_size, border_size:-border_size]
        frames.append(crop(final_result, original_h, original_w).detach().cpu())

    return {
        "frames": frames,
        "frame_count": len(frames),
        "canvas": {"width": original_w, "height": original_h},
        "source_image": str(image_path),
        "renderer": "paint_transformer_native_inference",
    }


def _native_frame_index_for_exported_count(frame_count: int, base_count: int, total_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("native render produced no frames")
    ratio = base_count / total_count
    return _native_frame_index_for_ratio(frame_count, ratio)


def _native_frame_index_for_ratio(frame_count: int, ratio: float) -> int:
    if frame_count <= 0:
        raise ValueError("native render produced no frames")
    return max(0, min(frame_count - 1, math.ceil(frame_count * ratio) - 1))


def _write_native_render_artifacts(native_render: dict[str, Any], output_dir: Path, frame_index: int) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frames_dir.glob("frame_*.png"):
        stale_frame.unlink()

    frame_paths = []
    frames = native_render["frames"]
    for index, frame in enumerate(frames[: frame_index + 1], start=1):
        frame_path = frames_dir / f"frame_{index:06d}.png"
        _save_tensor_image(frame[0], frame_path, Image)
        frame_paths.append(frame_path)

    final_path = output_dir / "final.png"
    _save_tensor_image(frames[frame_index][0], final_path, Image)
    replay_path = output_dir / "replay.gif"
    _write_replay_gif(frame_paths, replay_path, Image)
    manifest = {
        "renderer": native_render["renderer"],
        "source_image": native_render["source_image"],
        "canvas": native_render["canvas"],
        "stroke_count": frame_index + 1,
        "frame_count": len(frame_paths),
        "native_frame_index": frame_index,
        "native_frame_count": native_render["frame_count"],
        "final_image": str(final_path),
        "replay_gif": str(replay_path),
        "frames_dir": str(frames_dir),
        "frames": [str(frame_path) for frame_path in frame_paths],
    }
    _write_json(output_dir / "render_manifest.json", manifest)


def render_program_with_paint_transformer(stroke_program_path: Path, output_dir: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Paint Transformer rendering requires PyTorch") from exc

    from PIL import Image

    from Source.PaintTransformerReference.rendering import load_meta_brushes, param2stroke

    stroke_program_path = stroke_program_path.resolve()
    output_dir = output_dir.resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frames_dir.glob("frame_*.png"):
        frame_path.unlink()

    raw_program = json.loads(stroke_program_path.read_text(encoding="utf-8"))
    stroke_program = load_stroke_program_json(raw_program)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_brushes = load_meta_brushes(device)
    canvas = torch.zeros(
        1,
        3,
        stroke_program.canvas.height,
        stroke_program.canvas.width,
        device=device,
    )
    frame_paths: list[Path] = []
    for index, stroke in enumerate(stroke_program.strokes, start=1):
        param = _stroke_to_paint_transformer_param(torch, stroke).to(device)
        foreground, alpha = param2stroke(param, stroke_program.canvas.height, stroke_program.canvas.width, meta_brushes)
        opacity = torch.tensor(stroke.opacity, device=device).view(1, 1, 1, 1)
        canvas = foreground * alpha * opacity + canvas * (1 - alpha * opacity)
        frame_path = frames_dir / f"frame_{index:06d}.png"
        _save_tensor_image(canvas[0], frame_path, Image)
        frame_paths.append(frame_path)

    final_path = output_dir / "final.png"
    _save_tensor_image(canvas[0], final_path, Image)
    replay_path = output_dir / "replay.gif"
    _write_replay_gif(frame_paths, replay_path, Image)
    manifest = {
        "renderer": "paint_transformer_reference",
        "input": str(stroke_program_path),
        "canvas": stroke_program.canvas.to_json(),
        "stroke_count": len(stroke_program.strokes),
        "frame_count": len(frame_paths),
        "final_image": str(final_path),
        "replay_gif": str(replay_path),
        "frames_dir": str(frames_dir),
        "frames": [str(frame_path) for frame_path in frame_paths],
    }
    _write_json(output_dir / "render_manifest.json", manifest)


def render_program_final_with_paint_transformer(
    stroke_program_path: Path,
    output_path: Path,
    background_path: Path | None = None,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Paint Transformer rendering requires PyTorch") from exc

    from PIL import Image

    from Source.PaintTransformerReference.rendering import load_meta_brushes, param2stroke

    raw_program = json.loads(stroke_program_path.read_text(encoding="utf-8"))
    stroke_program = load_stroke_program_json(raw_program)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_brushes = load_meta_brushes(device)
    if background_path is None:
        canvas = torch.zeros(
            1,
            3,
            stroke_program.canvas.height,
            stroke_program.canvas.width,
            device=device,
        )
    else:
        canvas = _load_background_canvas(
            background_path=background_path,
            height=stroke_program.canvas.height,
            width=stroke_program.canvas.width,
            torch_module=torch,
            image_module=Image,
            device=device,
        )
    for stroke in stroke_program.strokes:
        param = _stroke_to_paint_transformer_param(torch, stroke).to(device)
        foreground, alpha = param2stroke(param, stroke_program.canvas.height, stroke_program.canvas.width, meta_brushes)
        opacity = torch.tensor(stroke.opacity, device=device).view(1, 1, 1, 1)
        canvas = foreground * alpha * opacity + canvas * (1 - alpha * opacity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_tensor_image(canvas[0], output_path, Image)


def _load_background_canvas(
    background_path: Path,
    height: int,
    width: int,
    torch_module,
    image_module,
    device,
):
    import numpy as np

    image = image_module.open(background_path).convert("RGB").resize((width, height))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch_module.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch_module.float32)


def _stroke_to_paint_transformer_param(torch_module, stroke) :
    # Paint Transformer theta is normalized over pi, while BrushWright angle is full turns.
    # BrushWright stores length as the major axis and width as the minor axis.
    # Paint Transformer's width parameter is the axis oriented by theta, so keep
    # the major axis in that slot when replaying collected strokes.
    theta = (stroke.angle * 2.0) % 1.0
    return torch_module.tensor(
        [
            [
                stroke.x,
                stroke.y,
                max(stroke.length, MIN_RENDER_SIZE),
                max(stroke.width, MIN_RENDER_SIZE),
                theta,
                stroke.color[0],
                stroke.color[1],
                stroke.color[2],
            ]
        ],
        dtype=torch_module.float32,
    )


MIN_RENDER_SIZE = 0.001


def _save_tensor_image(image_tensor, output_path: Path, image_module) -> None:
    image_array = (
        image_tensor.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
        .transpose((1, 2, 0))
        * 255
    ).astype("uint8")
    image_module.fromarray(image_array).save(output_path)


def _write_replay_gif(frame_paths: list[Path], output_path: Path, image_module) -> None:
    if not frame_paths:
        return
    images = [image_module.open(frame_path).convert("P", palette=image_module.Palette.ADAPTIVE) for frame_path in frame_paths]
    try:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=80, loop=0)
    finally:
        for image in images:
            image.close()


def _resolve_torch_device(torch_module, device_name: str | None):
    requested_device = device_name or "auto"
    if requested_device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA inference was requested, but PyTorch cannot see a CUDA GPU. "
            "Check `nvidia-smi`, NVIDIA driver status, and whether /dev/nvidia* devices are exposed."
        )
    return torch_module.device(requested_device)


def _resolve_available_stroke_split(
    available_count: int,
    requested_base_count: int,
    requested_finishing_count: int,
) -> tuple[int, int, bool]:
    requested_count = requested_base_count + requested_finishing_count
    if available_count >= requested_count:
        return requested_base_count, requested_finishing_count, False
    if available_count < 2:
        raise ValueError(f"Paint Transformer produced {available_count} strokes; need at least 2")
    if available_count > requested_finishing_count:
        return available_count - requested_finishing_count, requested_finishing_count, True

    finishing_ratio = requested_finishing_count / requested_count
    finishing_count = max(1, round(available_count * finishing_ratio))
    finishing_count = min(finishing_count, available_count - 1)
    base_count = available_count - finishing_count
    return base_count, finishing_count, True


def _select_stroke_window(
    strokes: list[dict[str, Any]],
    required_count: int,
    stroke_window: str,
) -> tuple[list[dict[str, Any]], int]:
    if stroke_window == "start":
        return strokes[:required_count], 0
    if stroke_window == "detail":
        selected_start_index = max(0, len(strokes) - required_count)
        return strokes[selected_start_index:selected_start_index + required_count], selected_start_index
    raise ValueError(f"unknown stroke_window: {stroke_window}")


def _attach_sampled_colors(torch_module, functional_module, img_patch, shape_param):
    grid = shape_param[:, :, :2].view(img_patch.shape[0] * DEFAULT_STROKE_NUM, 1, 1, 2).contiguous()
    img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, DEFAULT_STROKE_NUM, 1, 1, 1).view(
        img_patch.shape[0] * DEFAULT_STROKE_NUM,
        3,
        DEFAULT_PATCH_SIZE,
        DEFAULT_PATCH_SIZE,
    )
    color = functional_module.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(
        img_patch.shape[0],
        DEFAULT_STROKE_NUM,
        3,
    )
    return torch_module.cat([shape_param, color], dim=-1)


def _resolve_image_paths(image_path: Path | None, image_dir: Path | None) -> list[Path]:
    if image_path is None and image_dir is None:
        raise ValueError("provide --image or --image-dir")
    if image_path is not None and image_dir is not None:
        raise ValueError("provide only one of --image or --image-dir")
    if image_path is not None:
        return [image_path]
    if image_dir is None:
        raise ValueError("image_dir cannot be None")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def _program(canvas_size: int, metadata: dict[str, Any], strokes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": STROKE_PROGRAM_VERSION,
        "canvas": {"width": canvas_size, "height": canvas_size},
        "metadata": metadata,
        "strokes": strokes,
    }


def _relative_to_sample(sample_dir: Path, path: Path) -> str:
    return str(path.relative_to(sample_dir))


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

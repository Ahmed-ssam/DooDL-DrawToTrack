"""SAM 2.1 video predictor backend.

Written against the public API of facebookresearch/sam2 (``build_sam2_video_predictor``,
``init_state``, ``add_new_points_or_box``, ``add_new_mask``, ``propagate_in_video``).
``torch`` and ``sam2`` are imported lazily so everything else in DooDL (and the unit
tests, which inject a stand-in predictor) works without a GPU stack.
"""

from __future__ import annotations

import contextlib
import logging
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import Settings
from ..errors import BackendError, PromptError
from ..prompts import ObjectPrompt, PromptStrategy, build_sam_prompt
from .base import BackendInfo, Masks, SegmentationBackend

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sam2Model:
    file: str
    config: str
    hf_repo: str


# File names and config paths as documented in the SAM 2 repository (SAM 2.1 release).
SAM2_MODELS: dict[str, Sam2Model] = {
    "tiny": Sam2Model(
        "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml", "facebook/sam2.1-hiera-tiny"
    ),
    "small": Sam2Model(
        "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml", "facebook/sam2.1-hiera-small"
    ),
    "base_plus": Sam2Model(
        "sam2.1_hiera_base_plus.pt",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "facebook/sam2.1-hiera-base-plus",
    ),
    "large": Sam2Model(
        "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml", "facebook/sam2.1-hiera-large"
    ),
}

_MIN_CHECKPOINT_BYTES = 50 * 1024 * 1024


def checkpoint_urls(model: Sam2Model) -> list[str]:
    return [
        f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{model.file}",
        f"https://huggingface.co/{model.hf_repo}/resolve/main/{model.file}",
    ]


def ensure_checkpoint(model_name: str, directory: Path) -> Path:
    """Return the checkpoint path, downloading it on first use (never committed to git)."""
    model = SAM2_MODELS[model_name]
    target = directory / model.file
    if target.exists() and target.stat().st_size >= _MIN_CHECKPOINT_BYTES:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in checkpoint_urls(model):
        partial = target.with_suffix(".part")
        try:
            log.info("Downloading %s", url)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            if partial.stat().st_size < _MIN_CHECKPOINT_BYTES:
                raise BackendError("downloaded file is too small to be a checkpoint")
            partial.replace(target)
            return target
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            partial.unlink(missing_ok=True)
    raise BackendError(
        "Could not download the SAM 2.1 checkpoint. Tried:\n  "
        + "\n  ".join(errors)
        + f"\nDownload {model.file} manually into {directory}."
    )


@dataclass
class _Handle:
    state: Any
    frames_dir: Path
    num_frames: int
    prompted: dict[int, set[int]] = field(default_factory=dict)


class Sam2Backend(SegmentationBackend):
    def __init__(
        self,
        predictor: Any,
        *,
        model_name: str = "tiny",
        device: str = "cpu",
        torch_module: Any = None,
        autocast_dtype: Any = None,
        strategy: PromptStrategy = PromptStrategy.BOX_POINTS,
        positive_points: int = 4,
        negative_points: int = 4,
        offload_state_frames: int = 150,
    ) -> None:
        self.predictor = predictor
        self.model_name = model_name
        self.device = device
        self._torch = torch_module
        self._dtype = autocast_dtype
        self.strategy = strategy
        self.positive_points = positive_points
        self.negative_points = negative_points
        self.offload_state_frames = offload_state_frames
        label = f"SAM 2.1 {model_name.replace('_', ' ')}"
        self.info = BackendInfo(name="sam2", label=label, device=device)

    # ----------------------------------------------------------------- creation

    @classmethod
    def from_settings(cls, settings: Settings) -> Sam2Backend:
        try:
            import torch
        except ImportError as exc:
            raise BackendError(
                "PyTorch is not installed. SAM 2.1 needs it (preinstalled on Colab); "
                "see the README for install steps."
            ) from exc
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise BackendError(
                "SAM 2 is not installed. Run: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from exc

        device = _resolve_device(settings.device, torch)
        dtype = _resolve_dtype(settings.precision, device, torch)
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere or newer: TF32 is a free speed-up
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        spec = SAM2_MODELS[settings.sam2_model]
        checkpoint = ensure_checkpoint(settings.sam2_model, settings.checkpoint_dir)
        log.info("Loading SAM 2.1 %s on %s (autocast=%s)", settings.sam2_model, device, dtype)
        predictor = build_sam2_video_predictor(spec.config, str(checkpoint), device=device)
        return cls(
            predictor,
            model_name=settings.sam2_model,
            device=device,
            torch_module=torch,
            autocast_dtype=dtype,
            strategy=PromptStrategy(settings.prompt_strategy),
            positive_points=settings.positive_points,
            negative_points=settings.negative_points,
            offload_state_frames=settings.offload_state_frames,
        )

    # ------------------------------------------------------------------ helpers

    def _inference(self) -> AbstractContextManager[Any]:
        torch = self._torch
        if torch is None:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        if self._dtype is not None:
            stack.enter_context(torch.autocast(device_type=self.device, dtype=self._dtype))
        return stack

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    def _model_mask(self, mask: np.ndarray) -> Any:
        if self._torch is not None:
            return self._torch.from_numpy(mask.astype(bool))
        return mask.astype(bool)

    def _extract(self, obj_ids: Sequence[int], logits: Any, obj_id: int) -> np.ndarray:
        ids = [int(i) for i in obj_ids]
        if obj_id not in ids:
            raise BackendError("The model did not return a mask for this object.")
        arr = self._to_numpy(logits[ids.index(obj_id)])
        if arr.ndim == 3:
            arr = arr[0]
        return np.asarray(arr > 0.0)

    # ------------------------------------------------------------------ contract

    def open_session(self, frames_dir: Path, num_frames: int) -> _Handle:
        with self._inference():
            state = self.predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=num_frames > self.offload_state_frames,
            )
        return _Handle(state=state, frames_dir=frames_dir, num_frames=num_frames)

    def set_prompt(
        self, handle: _Handle, frame_idx: int, obj_id: int, prompt: ObjectPrompt
    ) -> np.ndarray:
        frame = cv2.imread(str(handle.frames_dir / f"{frame_idx:05d}.jpg"))
        if frame is None:
            raise PromptError(f"Frame {frame_idx} could not be read.")
        sam = build_sam_prompt(
            prompt,
            frame,
            self.strategy,
            positive_points=self.positive_points,
            negative_points=self.negative_points,
        )
        with self._inference():
            if sam.mask is not None:
                _, ids, logits = self.predictor.add_new_mask(
                    handle.state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=self._model_mask(sam.mask),
                )
            else:
                has_points = len(sam.points) > 0
                _, ids, logits = self.predictor.add_new_points_or_box(
                    handle.state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=sam.points if has_points else None,
                    labels=sam.labels if has_points else None,
                    clear_old_points=True,
                    normalize_coords=True,
                    box=sam.box,
                )
            mask = self._extract(ids, logits, obj_id)
        handle.prompted.setdefault(obj_id, set()).add(frame_idx)
        return mask

    def propagate(
        self, handle: _Handle, *, start_frame: int, reverse: bool, obj_ids: Sequence[int]
    ) -> Iterator[tuple[int, Masks]]:
        wanted = {int(i) for i in obj_ids}
        generator = self.predictor.propagate_in_video(
            handle.state, start_frame_idx=start_frame, reverse=reverse
        )
        try:
            while True:
                # The inference context wraps each step, not the whole generator, so it
                # is always entered and exited on the thread that is consuming it.
                with self._inference():
                    try:
                        frame_idx, ids, logits = next(generator)
                    except StopIteration:
                        return
                    masks = {
                        int(obj): self._extract(ids, logits, int(obj))
                        for obj in ids
                        if int(obj) in wanted
                    }
                yield int(frame_idx), masks
        finally:
            generator.close()

    def remove_object(self, handle: _Handle, obj_id: int) -> None:
        handle.prompted.pop(obj_id, None)
        with self._inference():
            try:
                self.predictor.remove_object(handle.state, obj_id, need_output=False)
            except TypeError:  # older sam2 builds have no need_output argument
                self.predictor.remove_object(handle.state, obj_id)
            except (KeyError, ValueError):
                log.debug("object %s was not in the SAM 2 state", obj_id)

    def close_session(self, handle: _Handle) -> None:
        with contextlib.suppress(Exception), self._inference():
            self.predictor.reset_state(handle.state)
        handle.prompted.clear()
        handle.state = None
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.empty_cache()

    def reset_stats(self) -> None:
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()

    def stats(self) -> dict[str, float]:
        if self._torch is not None and self.device == "cuda":
            return {"peak_gpu_gb": round(self._torch.cuda.max_memory_allocated() / 1e9, 2)}
        return {}


def _resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(precision: str, device: str, torch: Any) -> Any:
    """Autocast dtype, or None for plain fp32.

    "auto" picks bf16 only where the GPU supports it natively (Ampere and newer). On a
    T4 that means fp32. fp16 is available as an opt-in because it is faster there but
    has not been validated for SAM 2's memory attention: benchmark before trusting it.
    """
    if device != "cuda":
        return None
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else None

import torch
from typing import Iterable, Optional, Sequence


def _fmt(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return "None"
    try:
        return f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"
    except Exception:
        return "<tensor>"


def _fail(ctx: str, msg: str, *tensors: Optional[torch.Tensor]) -> None:
    details = ", ".join(_fmt(t) for t in tensors if isinstance(t, torch.Tensor))
    raise ValueError(f"[{ctx}] {msg}. {('Tensors: ' + details) if details else ''}")


def assert_is_long(ctx: str, t: Optional[torch.Tensor], name: str) -> None:
    if t is None:
        _fail(ctx, f"{name} must not be None")
    if t.dtype != torch.long:
        _fail(ctx, f"{name} must have dtype torch.long, but got {t.dtype}", t)


def assert_is_float(ctx: str, t: Optional[torch.Tensor], name: str,
                    allowed=(torch.float32, torch.float16, torch.bfloat16)) -> None:
    if t is None:
        _fail(ctx, f"{name} must not be None")
    if t.dtype not in allowed:
        _fail(ctx, f"{name} must be float tensor (one of {allowed}), but got {t.dtype}", t)


def assert_ndim(ctx: str, t: Optional[torch.Tensor], name: str, expected_ndim: int) -> None:
    if t is None:
        _fail(ctx, f"{name} must not be None")
    if t.ndim != expected_ndim:
        _fail(ctx, f"{name} must have {expected_ndim} dims, but got {t.ndim}", t)


def assert_shape(ctx: str, t: Optional[torch.Tensor], name: str, expected: Sequence[Optional[int]]) -> None:
    if t is None:
        _fail(ctx, f"{name} must not be None")
    if t.ndim != len(expected):
        _fail(ctx, f"{name} must have {len(expected)} dims, but got {t.ndim}", t)
    for i, e in enumerate(expected):
        if e is None:
            continue
        if t.shape[i] != e:
            _fail(ctx, f"{name} dim {i} must be {e}, but got {t.shape[i]}", t)


def assert_same_device(ctx: str, tensors: Iterable[Optional[torch.Tensor]], allow_none: bool = True) -> None:
    devices = [t.device for t in tensors if (t is not None or not allow_none)]
    if not devices:
        return
    first = devices[0]
    for d in devices[1:]:
        if d != first:
            _fail(ctx, f"All tensors must be on the same device ({first} != {d})")


def assert_mask01(ctx: str, mask: Optional[torch.Tensor], name: str) -> None:
    if mask is None:
        _fail(ctx, f"{name} must not be None")
    if mask.dtype == torch.bool:
        _fail(ctx, f"{name} must be integer 0/1 mask, not bool", mask)
    if not torch.all((mask == 0) | (mask == 1)):
        bad = mask[~((mask == 0) | (mask == 1))]
        _fail(ctx, f"{name} must contain only 0/1 values; found {bad.unique().tolist()[:4]}", mask)


def assert_monotonic_starts(ctx: str, starts: Optional[torch.Tensor], total_length: int, name: str) -> None:
    if starts is None:
        _fail(ctx, f"{name} must not be None")
    if starts.ndim != 1:
        _fail(ctx, f"{name} must be 1-D, but got {starts.ndim}", starts)
    if starts.numel() == 0:
        return
    if starts.dtype != torch.long:
        _fail(ctx, f"{name} must be torch.long", starts)
    if torch.any(starts < 0):
        _fail(ctx, f"{name} must be non-negative", starts)
    if torch.any(starts >= total_length):
        _fail(ctx, f"{name} must be < total_length={total_length}", starts)
    # strictly increasing
    if torch.any(starts[1:] <= starts[:-1]):
        _fail(ctx, f"{name} must be strictly increasing", starts)


def count_token_occurrences(input_ids: torch.Tensor, token_id: int) -> int:
    if input_ids is None:
        return 0
    return int((input_ids == token_id).sum().item())

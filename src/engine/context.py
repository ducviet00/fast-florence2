from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    k_cache: list[torch.Tensor | None] = field(default_factory=list)
    v_cache: list[torch.Tensor | None] = field(default_factory=list)
    cross_k_cache: list[torch.Tensor | None] = field(default_factory=list)
    cross_v_cache: list[torch.Tensor | None] = field(default_factory=list)


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill,
    k_cache=[],
    v_cache=[],
    cross_k_cache=[],
    cross_v_cache=[],
) -> Context:
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        k_cache,
        v_cache,
        cross_k_cache,
        cross_v_cache,
    )
    return _CONTEXT


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

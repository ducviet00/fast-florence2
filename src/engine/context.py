from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    cross_block_tables: torch.Tensor | None = None
    cross_slot_mapping: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill,
    context_lens=None,
    block_tables=None,
    slot_mapping=None,
    cross_block_tables=None,
    cross_slot_mapping=None,
) -> Context:
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        context_lens=context_lens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        cross_block_tables=cross_block_tables,
        cross_slot_mapping=cross_slot_mapping,
    )
    return _CONTEXT


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

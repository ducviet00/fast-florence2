import torch
from transformers import BartConfig
from transformers import BartForConditionalGeneration as HFBart

from engine.context import set_context
from models.bart import BartForConditionalGeneration
from utils.loader import load_model

MODEL_NAME = "bart-large-cnn"
BATCH_SIZE = 8
ENC_SEQ_LEN = 256
DEC_SEQ_LEN = 64
VOCAB_SIZE = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_token_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Return a (batch_size, seq_len) tensor of random token IDs in [0, VOCAB_SIZE)."""
    return torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE)


def make_positions(batch_size: int, seq_len: int) -> torch.Tensor:
    """Return a (batch_size, seq_len) tensor where each row is 0..seq_len-1."""
    return torch.arange(seq_len, device=DEVICE).expand(batch_size, seq_len).contiguous()


def build_custom_model() -> tuple[BartForConditionalGeneration, BartConfig]:
    config = BartConfig.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration(config)
    load_model(model, MODEL_NAME)
    return model.to(DEVICE), config


def build_reference_model() -> HFBart:
    return HFBart.from_pretrained(MODEL_NAME).to(DEVICE)


def test_logits_match_transformers() -> None:
    model, config = build_custom_model()
    set_context(
        is_prefill=True,
        k_cache=[None for _ in range(config.decoder_layers)],
        v_cache=[None for _ in range(config.decoder_layers)],
        cross_k_cache=[None for _ in range(config.encoder_layers)],
        cross_v_cache=[None for _ in range(config.encoder_layers)],
    )
    reference = build_reference_model()

    encoder_input_ids = make_token_ids(BATCH_SIZE, ENC_SEQ_LEN)
    encoder_positions = make_positions(BATCH_SIZE, ENC_SEQ_LEN)
    decoder_input_ids = make_token_ids(BATCH_SIZE, DEC_SEQ_LEN)
    decoder_positions = make_positions(BATCH_SIZE, DEC_SEQ_LEN)

    with torch.inference_mode():
        expected = reference(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
        ).logits

        hidden = model(
            decoder_input_ids=decoder_input_ids,
            decoder_positions=decoder_positions,
            encoder_input_ids=encoder_input_ids,
            encoder_positions=encoder_positions,
        )
        logits = model.compute_logits(hidden)

    torch.testing.assert_close(logits, expected)


if __name__ == "__main__":
    test_logits_match_transformers()

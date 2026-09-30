import torch
from transformers import AutoConfig
from transformers import BartForConditionalGeneration as HFBart

from src.models.bart import BartForConditionalGeneration
from src.utils.loader import load_model

MODEL_NAME = "bart-large-cnn"
BATCH_SIZE = 4
ENC_SEQ_LEN = 256
DEC_SEQ_LEN = 512
VOCAB_SIZE = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_token_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Return a (batch_size, seq_len) tensor of random token IDs in [0, VOCAB_SIZE)."""
    return torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE)


def make_positions(batch_size: int, seq_len: int) -> torch.Tensor:
    """Return a (batch_size, seq_len) tensor where each row is 0..seq_len-1."""
    return torch.arange(seq_len, device=DEVICE).expand(batch_size, seq_len).contiguous()


def build_custom_model() -> BartForConditionalGeneration:
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration(config)
    load_model(model, MODEL_NAME)
    return model.to(DEVICE)


def build_reference_model() -> HFBart:
    return HFBart.from_pretrained(MODEL_NAME).to(DEVICE)


def test_logits_match_transformers() -> None:
    model = build_custom_model()
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

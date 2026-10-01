import torch
from transformers import BartConfig
from transformers import BartForConditionalGeneration as HFBart

from engine.context import set_context
from models.bart import BartForConditionalGeneration
from utils.loader import load_model

MODEL_NAME = "bart-large-cnn"
BATCH_SIZE = 1
ENC_SEQ_LEN = 256
DEC_SEQ_LEN = 500
VOCAB_SIZE = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16


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
    return model.to(DEVICE, DTYPE), config


def build_reference_model() -> HFBart:
    return HFBart.from_pretrained(MODEL_NAME).to(DEVICE, DTYPE)


def test_logits_match_transformers() -> None:
    model, _ = build_custom_model()
    set_context(is_prefill=True)
    reference = build_reference_model()

    encoder_input_ids = make_token_ids(BATCH_SIZE, ENC_SEQ_LEN)
    encoder_positions = make_positions(BATCH_SIZE, ENC_SEQ_LEN)
    decoder_input_ids = make_token_ids(BATCH_SIZE, DEC_SEQ_LEN)
    decoder_positions = make_positions(BATCH_SIZE, DEC_SEQ_LEN)

    with torch.inference_mode():
        expected_logits = reference(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
        ).logits

        expected_token_ids = torch.softmax(expected_logits.float(), dim=-1).argmax(dim=-1)


        actual_logits = model.compute_logits(model(
            decoder_input_ids=decoder_input_ids,
            decoder_positions=decoder_positions,
            encoder_input_ids=encoder_input_ids,
            encoder_positions=encoder_positions,
        ))
        actual_token_ids = torch.softmax(actual_logits.float(), dim=-1).argmax(dim=-1)


    torch.testing.assert_close(actual_token_ids, expected_token_ids)


if __name__ == "__main__":
    test_logits_match_transformers()

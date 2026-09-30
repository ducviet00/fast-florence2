import torch
from transformers import BartConfig
from transformers import BartForConditionalGeneration as HFBart

from engine.context import set_context, get_context
from models.bart import BartForConditionalGeneration
from utils.loader import load_model

MODEL_NAME = "bart-large-cnn"
BATCH_SIZE = 4
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 64
DEC_N_STEPS = 16
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


def test_decode_loop() -> None:
    model, config = build_custom_model()
    reference = build_reference_model()
    dt_c, dt_r = next(model.parameters()).dtype, next(reference.parameters()).dtype
    print(f"dtype custom={dt_c}  reference={dt_r}")

    encoder_input_ids = make_token_ids(BATCH_SIZE, ENC_SEQ_LEN)
    encoder_positions = make_positions(BATCH_SIZE, ENC_SEQ_LEN)
    decoder_input_ids = make_token_ids(BATCH_SIZE, DEC_SEQ_LEN)
    decoder_positions = make_positions(BATCH_SIZE, DEC_SEQ_LEN)

    with torch.inference_mode():
        logits = reference(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
        ).logits
        next_input_ids = torch.softmax(logits[:, -1], dim=-1).argmax(dim=-1)
        next_input_ids.unsqueeze_(1)
        next_input_ids = torch.concat([decoder_input_ids, next_input_ids], dim=-1)
        for _ in range(DEC_N_STEPS):
            logits = reference(
                input_ids=encoder_input_ids,
                decoder_input_ids=next_input_ids,
            ).logits
            new_input_ids = torch.softmax(logits[:, -1], dim=-1).argmax(dim=-1)
            new_input_ids.unsqueeze_(1)
            next_input_ids = torch.concat([next_input_ids, new_input_ids], dim=-1)
        print(decoder_input_ids)
        print(next_input_ids)
        final_logits = logits
        print(final_logits.shape)

    with torch.inference_mode():
        context = set_context(
            is_prefill=True,
            k_cache=[None for _ in range(config.decoder_layers)],
            v_cache=[None for _ in range(config.decoder_layers)],
            cross_k_cache=[None for _ in range(config.encoder_layers)],
            cross_v_cache=[None for _ in range(config.encoder_layers)],
        )
        context.is_prefill = True
        hidden = model(
            decoder_input_ids=decoder_input_ids,
            decoder_positions=decoder_positions,
            encoder_input_ids=encoder_input_ids,
            encoder_positions=encoder_positions,
        )
        logits = model.compute_logits(hidden)
        next_input_ids = torch.softmax(logits[:, -1], dim=-1).argmax(dim=-1)
        next_input_ids.unsqueeze_(1)
        next_positions = decoder_positions[:, -1].unsqueeze(1) + 1
        print(logits.shape)
        print(final_logits[:,:logits.shape[1]].shape)
        torch.testing.assert_close(logits.cpu().float(), final_logits[:,:logits.shape[1]].cpu().float(), atol=1e-3, rtol=1e-3)

        all_logits = [logits]
        context.is_prefill = False
        pos = DEC_SEQ_LEN
        for _ in range(DEC_N_STEPS):
            hidden = model(
                decoder_input_ids=next_input_ids,
                decoder_positions=next_positions,
            )
            logits = model.compute_logits(hidden)
            print(pos)
            print(logits.cpu().squeeze(1).float())
            print(final_logits[:,pos,:].cpu().float())
            is_close = torch.allclose(logits.cpu().squeeze(1).float(), final_logits[:,pos,:].cpu().float(), atol=1e-3, rtol=1e-3)
            # is_close = torch.allclose(logits.cpu().squeeze(1).float(), final_logits[:,pos,:].cpu().float(), atol=1e-3, rtol=1e-3)
            pos+=1
            print(is_close)
            next_input_ids = torch.softmax(logits, dim=-1).argmax(dim=-1)
            next_positions = next_positions + 1
            all_logits.append(logits)

        all_logits_tensor = torch.concat(all_logits, dim=1)
        print(all_logits_tensor.shape)



    output_ids = torch.softmax(all_logits_tensor[:, :-DEC_N_STEPS], dim=-1).argmax(dim=-1)
    expected_output_ids = torch.softmax(final_logits[:, :-DEC_N_STEPS], dim=-1).argmax(dim=-1)
    torch.testing.assert_close(output_ids, expected_output_ids) # OK
    torch.testing.assert_close(all_logits_tensor, final_logits)
    # Mismatched elements: 4219780 / 16084480 (26.2%)
    # Greatest absolute difference: 21.95160675048828 at index (3, 69, 0) (up to 1e-05 allowed)
    # Greatest relative difference: 1443570.125 at index (1, 69, 3590) (up to 1.3e-06 allowed)

if __name__ == "__main__":
    test_decode_loop()

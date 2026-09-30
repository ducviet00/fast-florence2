import torch
from transformers import AutoConfig
from transformers import BartForConditionalGeneration as TransformersBart

from src.models.bart import BartForConditionalGeneration
from src.utils.loader import load_model

config = AutoConfig.from_pretrained("bart-large-cnn")
model = BartForConditionalGeneration(config)
transformers_model = TransformersBart.from_pretrained("bart-large-cnn")
load_model(model, "bart-large-cnn")
encoder_input_ids = torch.arange(0, 64).reshape(2, 32)
encoder_positions = torch.arange(0, 32).repeat(2).reshape(2, 32)
decoder_input_ids = torch.arange(0, 10).repeat(2).reshape(2, 10)
decoder_positions = torch.arange(0, 10).repeat(2).reshape(2, 10)

with torch.inference_mode():
    transformers_out = transformers_model.forward(
        input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
    )

    expected = transformers_out.logits

    out = model(
        decoder_input_ids=decoder_input_ids,
        decoder_positions=decoder_positions,
        encoder_input_ids=encoder_input_ids,
        encoder_positions=encoder_positions,
    )
    logits = model.compute_logits(out)
    torch.testing.assert_close(logits, expected)

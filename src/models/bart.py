import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import get_kernel
from transformers import BartConfig

from engine.context import get_context

activation = get_kernel("kernels-community/activation", version=1)
ACT2FN = {
    "gelu": activation.layers.Gelu(),
}


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input: torch.Tensor):
        return super().forward(input + self.offset)


class BartScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input: torch.Tensor):
        return super().forward(input) * self.embed_scale


class BartEncoderSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: BartConfig | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # get QKV proj
        # batch_size, num_heads, seq_len, head_dim
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # batch_size, num_heads, seq_len, head_dim
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # batch_size, num_heads, seq_len, head_dim
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class BartDecoderSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: BartConfig | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # get QKV proj
        # batch_size, num_heads, seq_len, head_dim
        q_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()
        k_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()
        v_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()

        if context.k_cache[self.layer_idx] is not None and context.v_cache[self.layer_idx] is not None:
            k_states = torch.concat([context.k_cache[self.layer_idx], k_states], dim=2)
            v_states = torch.concat([context.v_cache[self.layer_idx], v_states], dim=2)

        context.k_cache[self.layer_idx] = k_states.clone()
        context.v_cache[self.layer_idx] = v_states.clone()

        # print("="*100)
        # print("SelfAttn:", self.layer_idx)
        # print(q_states.shape)
        # print(k_states.shape)
        # print(v_states.shape)
        attn_output = F.scaled_dot_product_attention(
            q_states,
            k_states,
            v_states,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class BartDecoderCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: BartConfig | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = get_context()
        input_shape = hidden_states.shape[:-1]
        # batch_size, seq_len, num_heads, head_dim
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)

        # batch_size, num_heads, dec_seq_len, head_dim
        q_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if encoder_hidden_states is not None:
            # batch_size, num_heads, enc_seq_len, head_dim
            kv_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)
            enc_k_states = self.k_proj(encoder_hidden_states).view(kv_shape).transpose(1, 2).contiguous()
            enc_v_states = self.v_proj(encoder_hidden_states).view(kv_shape).transpose(1, 2).contiguous()
            context.cross_k_cache[self.layer_idx] = enc_k_states
            context.cross_v_cache[self.layer_idx] = enc_v_states
        else:
            assert context.cross_k_cache is not None
            assert context.cross_v_cache is not None
            enc_k_states = context.cross_k_cache[self.layer_idx]
            enc_v_states = context.cross_v_cache[self.layer_idx]

        # print("="*100)
        # print("CrossAttn:", self.layer_idx)
        # print(q_states.shape)
        # print(enc_k_states.shape)
        # print(enc_v_states.shape)
        attn_output = F.scaled_dot_product_attention(
            q_states,
            enc_k_states,
            enc_v_states,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig, layer_idx: int | None = None):
        super().__init__()
        assert config.d_model
        assert config.encoder_ffn_dim
        assert config.encoder_attention_heads
        self.embed_dim = config.d_model

        self.self_attn = BartEncoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN.get(config.activation_function, F.gelu)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig, layer_idx: int | None = None):
        super().__init__()
        assert config.d_model
        assert config.decoder_ffn_dim
        assert config.decoder_attention_heads
        self.embed_dim = config.d_model

        self.self_attn = BartDecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN.get(config.activation_function, F.gelu)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartDecoderCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
            layer_idx=layer_idx,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Self Attention
        residual = decoder_hidden_states
        hidden_states = self.self_attn(decoder_hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross Attention
        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartEncoder(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        assert config.d_model
        assert config.pad_token_id
        assert config.encoder_layers

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
        )

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [
                BartEncoderLayer(config, layer_idx=i)
                for i in range(config.encoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
    ) -> torch.Tensor:

        inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class BartDecoder(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        assert config.d_model
        assert config.pad_token_id
        assert config.decoder_layers

        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [
                BartDecoderLayer(config, layer_idx=i)
                for i in range(config.decoder_layers)
            ]
        )

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
    ) -> torch.Tensor:

        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        return hidden_states


class BartModel(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        assert config.pad_token_id
        assert config.d_model
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = BartScaledWordEmbedding(
            vocab_size, config.d_model, padding_idx, embed_scale=embed_scale
        )

        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)

        if config.tie_word_embeddings:
            self.decoder.embed_tokens.weight.data = self.shared.weight.data
            self.encoder.embed_tokens.weight.data = self.shared.weight.data

    def forward(
        self,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_positions: torch.LongTensor | None = None,
        encoder_input_ids: torch.LongTensor | None = None,
        encoder_positions: torch.LongTensor | None = None,
    ) -> tuple:

        encoder_outputs = None
        if encoder_input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=encoder_input_ids,
                positions=encoder_positions,
            )

        return self.decoder(
            input_ids=decoder_input_ids,
            positions=decoder_positions,
            encoder_hidden_states=encoder_outputs,
        )


class BartForConditionalGeneration(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.model = BartModel(config)
        self.final_logits_bias = nn.Buffer(
            torch.zeros((1, self.model.shared.num_embeddings))
        )
        assert config.d_model
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )

        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.shared.weight.data

    def forward(
        self,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_positions: torch.LongTensor | None = None,
        encoder_input_ids: torch.LongTensor | None = None,
        encoder_positions: torch.LongTensor | None = None,
    ) -> torch.Tensor:

        return self.model(
            decoder_input_ids=decoder_input_ids,
            decoder_positions=decoder_positions,
            encoder_input_ids=encoder_input_ids,
            encoder_positions=encoder_positions,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

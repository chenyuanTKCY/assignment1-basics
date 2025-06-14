from .layers import *
import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    """ Basic Transformer block."""

    def __init__(self, d_model: int, num_heads:int, d_ff: int, rope:nn.Module = None, **kwargs):
        """Initialize a Transformer block.

        d_model: dimension of the model
        num_heads: number of attention heads
        d_ff: dimension of the feed-forward network
        rope:RoPE module, should be pre-initialized
        kwargs:device, dtype, ..
        """

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.ln1 = RMSNorm(d_model, **kwargs)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, **kwargs)
        self.ln2 = RMSNorm(d_model, **kwargs)
        self.ffn = SwiGLU(d_model,d_ff,**kwargs)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Forward pass of the Transformer block. Computes x + FFN(RMSNorm(x + attn(RMSNorm(x)))).

        x: input tensor
        """

        # it's so beautiful that my layers are all properly batched!
        h = x + self.attn(self.ln1(x))

        # the x add is already implicit in h
        h = h + self.ffn(self.ln2(h))
        return h

"""Ablation Study: removing RMSNorm"""
class TransformerBlockNoRMSNorm(TransformerBlock):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, **kwargs):
        super().__init__(d_model, num_heads, d_ff, rope, **kwargs)

    # we only need to override the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(x)
        h = h + self.ffn(h)
        return h

class TransformerLM(nn.Module):
    """Full Transformer language model."""

    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        """Initialize a Transformer language model.

        d_model: dimension of the model
        vocab_size: size of the vocabulary
        context_length: context length
        num_layers: number of layers
        rope_theta: RoPE theta
        num_heads: number of attention heads
        d_ff: hidden dimension of the feed-forward network
        kwargs: device, dtype, etc.
        """

        super().__init__()
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, **kwargs)
        self.embedding = Embedding(vocab_size, d_model, **kwargs)
        self.layers = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, **kwargs)
        # note that the weights are stored out_features, in_features
        self.lm_head = Linear(d_model, vocab_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.layers(h)
        h = self.ln_final(h)
        h = self.lm_head(h)
        return h
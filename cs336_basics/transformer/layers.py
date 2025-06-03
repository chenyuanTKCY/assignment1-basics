import torch
import torch.nn as nn
import einops
import math
from jaxtyping import Float

class Linear(nn.Module):
    """ Linear Layer"""

    def __init__(self, in_features: int , out_features: int, device :torch.device | None = None, dtype : torch.dtype|None = None) -> None:
        # super().__init__(*args, **kwargs)
        """
        Initialize the linear layer.
        in_features: int stands for the input features.
        out_features: int stands for the output features.
        dtype: data type of the weight matirx
        """
        super().__init__()
        self.weight =  nn.Parameter(torch.zeros(out_features,in_features, device = device, dtype = dtype))
        stdev = (2/(in_features+out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, 0, stdev, -3 * stdev, 3 * stdev )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        """
        Initalize the RMSNorm layer.
        d_model: dimension of the model
        eps: epsilon for numerical stability
        device: device to store tha gain on
        dtype: data type of the gain
        """
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size, sequence_length, d_model
        in_type = x.dtype
        x = x.to(torch.float32)

        result = None
        squared_sums = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt((1/self.d_model)* squared_sums + self.eps)
        # gain_values = self.gain.unsqueeze(0)  # shape: (1, d_model)
        gain_values = self.gain.view(1, 1, -1)
        result = (x/rms)* gain_values
        return result.to(in_type)

def silu(x:torch.Tensor)->torch.Tensor:
    """SiLU activation function."""
    return x* torch.sigmoid(x)
class SiLU(nn.Module):
    """SiLU activation function"""
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """Initialize SiLU feed-forward network.

        d_model: dimension of the model
        d_ff: dimension of hidder layer
        device: device to store the weights on
        dtype: data type of the weights
        """
        super().__init__()

        self.W1 = nn.Parameter(torch.zeros(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.zeros(d_model, d_ff, device=device, dtype=dtype))
        stdev = (2/(d_ff+d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.W1, mean=0, std=stdev, a=-3*stdev, b=3*stdev)
        torch.nn.init.trunc_normal_(self.W2, mean=0, std=stdev, a=-3*stdev, b=3*stdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einops.einsum(x,self.W1, " ... d_model, d_ff d_model -> ... d_ff")
        z = silu(w1x)
        result = einops.einsum(z, self.W2, " ... d_ff, d_model d_ff -> ... d_model")
        return result

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network"""
    def __init__(self, d_model, d_ff = None, device = None, dtype = None):
        """Initialize SwigLU feed-forward network
        d_model: the dimension of the model
        d_ff: dimension of hidder layer
        device: device to store the weights on
        dtype: data type of the weights
        """
        super().__init__()

        if d_ff is None:
            d_ff = (8/3)* d_model
            d_ff = 64 * math.cel(d_ff/64)

        self.W1 = nn.Parameter(torch.zeros(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.zeros(d_model,d_ff,device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.zeros(d_ff,d_model,device = device, dtype = dtype))
        stdev = (2/(d_ff+d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.W1, mean=0, std=stdev, a = -3*stdev, b= 3*stdev)
        torch.nn.init.trunc_normal_(self.W2, mean=0, std=stdev, a=-3*stdev, b=3*stdev)
        torch.nn.init.trunc_normal_(self.W3, mean=0, std=stdev, a=-3*stdev, b=3*stdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))"""
        w1x = einops.einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        w3x = einops.einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        z = silu(w1x)
        z = z * w3x

        results = einops.einsum(z, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return results

class RoPE(nn.Module):

    def __init__(self, theta: float, d_k:int, max_seq_len: int, device = None, dtype = None):

        """Initialize RoPe
        theta: theta for RoPE
        d_k: RoPE dimension
        max_seq_len: maximun sequence length
        device: device to store te values on
        dtype: data type of the values
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        idx = torch.arange(0, max_seq_len, device=device, dtype=dtype)
        denom = theta ** (torch.arange(0,d_k,2,device=device, dtype=dtype)/d_k)
        theta_i_k = idx.unsqueeze(1) /denom.unsqueeze(0)

        # # step 1 calculate frequency vector
        # dim_pair = torch.arange(0,d_k,2)
        # inv_freq = 1.0 / (theta ** (dim_pair / d_k))

        # pos = torch.arange(max_seq_len).unsqueeze(1)
        # freqs = pos * inv_freq
        cos_cache = torch.cos(theta_i_k)
        sin_cache = torch.sin(theta_i_k)

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)


    def forward(self, x:torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        assert d_k % 2 == 0
        assert d_k == self.d_k

        if token_positions is not None:
            cos_values = self.cos_cache[token_positions, :]
            sin_values = self.sin_cache[token_positions, :]
        else:
            cos_values = self.cos_cache[:seq_len, :]
            sin_values = self.sin_cache[:seq_len, :]

        x_split = einops.rearrange(x,"... seq_len (d_split pair) -> ... seq_len d_split pair", d_split = self.d_k //2, pair = 2)
        even_x = x_split[..., 0]
        odd_x = x_split[..., 1]

        x_rotate_even = even_x * cos_values - odd_x * sin_values
        x_rotate_odd = even_x * sin_values + odd_x * cos_values

        x_rotated = torch.stack([x_rotate_even, x_rotate_odd], dim = -1)

        x_rotated = einops.rearrange(x_rotated, " ... seq_len d_split pair -> ... seq_len (d_split pair)")

        return x_rotated

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute softmax over a given dimension"""

    offset = torch.max(x, dim=dim, keepdim=True)[0]

    numerator = x-offset
    numerator = torch.exp(numerator)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)

    return numerator/denominator


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:

    """Compute scaled dot product attention.
    Q: queries, shape (..., seq_len_q, d_k)
    K: keys, shape (..., seq_len_k, d_k)
    V: values, shape(..., seq_len_k, d_v)
    mask: mask to apply to the attention weights, shape(..., sep_len_q, seq_len_k)
    """

    d_k  = Q.shape[-1]

    attn_weights = einops.einsum(Q,K,"... seq_len_q d_k, ... seq_len_k  d_k -> ... seq_len_q seq_len_k")
    attn_weights /= math.sqrt(d_k)

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

    smax = softmax(attn_weights, dim=-1)
    result = einops.einsum(smax, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return result


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, nums_heads: int, rope: nn.Module = None, device = None, dtype = None):
        """Initialize multi-head self-attention layer.
        d_model: dimension of the model
        num_heads: number of attention heads
        rope: Rope module
        device
        dtype
        """

        super().__init__()
        self.num_heads = nums_heads
        self.d_model = d_model
        self.rope = rope

        self.WQ = nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.WK = nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.WV = nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))

        self.WO = nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))

        stdev = (2/(d_model+d_model)) ** 0.5

        torch.nn.init.trunc_normal_(self.WQ, mean = 0, std = stdev, a = -3*stdev, b = 3*stdev)
        torch.nn.init.trunc_normal_(self.WK, mean = 0, std = stdev, a = -3*stdev, b = 3*stdev)
        torch.nn.init.trunc_normal_(self.WV, mean = 0, std = stdev, a = -3*stdev, b = 3*stdev)
        torch.nn.init.trunc_normal_(self.WO, mean = 0, std = stdev, a = -3*stdev, b = 3*stdev)
        # torch.nn.init.trunc_normal_(self.WQ, mean = 0, std = stdev, a = -3*stdev, b = 3*stdev)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        seq_len = x.shape[-2]
        casual_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype))

        Q = einops.rearrange(self.WQ, "(h dk) d_model -> h dk d_model", h = self.num_heads)
        Q = einops.einsum(Q,x,"h dk d_model, ... seq_len d_model -> ... h seq_len dk")
        K = einops.rearrange(self.WK, "(h dk) d_model -> h dk d_model",h=self.num_heads)
        K = einops.einsum(K,x,"h dk d_model, ... seq_len d_model -> ... h seq_len dk")
        V = einops.rearrange(self.WV, "(h dk) d_model -> h dk d_model",h=self.num_heads)
        V = einops.einsum(V,x,"h dk d_model, ... seq_len d_model -> ... h seq_len dk")

        if self.rope:
            Q = self.rope(Q)
            K = self.rope(K)

        attn = scaled_dot_product_attention(Q,K,V,mask=casual_mask)
        attn = einops.rearrange(attn, "... h seq_len dv -> ... seq_len (h dv)", h = self.num_heads)

        output = einops.einsum(attn, self.WO, "... seq_len hdv, d_model hdv -> ... seq_len d_model")

        return output
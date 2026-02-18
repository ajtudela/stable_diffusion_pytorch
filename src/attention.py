import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention module.

    Allows every position in a sequence to attend to every other
    position, capturing global dependencies. The embedding is split
    into n_heads independent heads so each head can specialize in
    different types of relationships.
    """

    def __init__(
            self,
            n_heads: int,
            d_embed: int,
            in_proj_bias: bool = True,
            out_proj_bias: bool = True
    ) -> None:
        """
        Initialize the self-attention module.

        Parameters
        ----------
        n_heads : int
            Number of attention heads. The embedding dimension is split
            equally across heads.
        d_embed : int
            Total embedding dimension of the input and output.
        in_proj_bias : bool
            Whether to include a bias term in the input projection.
        out_proj_bias : bool
            Whether to include a bias term in the output projection.
        """
        super().__init__()

        # Single linear layer that projects the input into queries,
        # keys, and values simultaneously (3 * d_embed output).
        # Using one fused projection is more efficient than three
        # separate layers; we split the result into Q, K, V later.
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # Projects the concatenated multi-head output back to d_embed,
        # mixing information gathered by all heads into a single vector.
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        # Store the number of heads for use in forward().
        self.n_heads = n_heads

        # Dimension of each individual attention head.
        # Each head operates on a d_head-dimensional subspace so the
        # total computation stays equivalent to one d_embed attention.
        self.d_head = d_embed // n_heads

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = False
    ) -> torch.Tensor:
        """
        Compute multi-head self-attention over the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, SeqLen, d_embed).
        causal_mask : bool
            If True, prevents each position from attending to future
            positions (autoregressive / causal masking).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (Batch, SeqLen, d_embed) where
            each position contains a weighted sum of value vectors.
        """
        # Save the original shape to restore it after attention.
        # (Batch, SeqLen, d_embed)
        input_shape = x.shape

        # Unpack dimensions for clarity and to build the reshape target.
        batch_size, sequence_length, d_embed = input_shape

        # Target shape after splitting d_embed into n_heads * d_head.
        # We will transpose to move the head dimension before SeqLen
        # so that batched matmul treats each head independently.
        interim_shape = (
            batch_size, sequence_length, self.n_heads, self.d_head
        )

        # Project input to Q, K, V in a single linear pass, then split
        # along the last dimension into three equal tensors.
        # in_proj: (Batch, SeqLen, d_embed) -> (Batch, SeqLen, 3*d_embed)
        # chunk:   -> 3 x (Batch, SeqLen, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Reshape each of Q, K, V from a flat embedding into n_heads
        # separate head vectors, then transpose so the head axis is
        # second: (Batch, n_heads, SeqLen, d_head).
        # This lets PyTorch's batched matmul (@) run all heads in
        # parallel without an explicit loop.
        # (Batch, SeqLen, d_embed) -> (Batch, n_heads, SeqLen, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute raw attention scores: dot product of every query with
        # every key. k is transposed on the last two dims so that the
        # result is (Batch, n_heads, SeqLen_q, SeqLen_k).
        # Higher dot-product = higher compatibility between positions.
        # (Batch, n_heads, SeqLen, SeqLen)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Build an upper-triangular boolean mask (True above the
            # main diagonal). Position i should not attend to position
            # j > i in autoregressive generation.
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Replace masked positions with -inf so softmax maps them
            # to 0, effectively ignoring future tokens.
            weight.masked_fill_(mask, -torch.inf)

        # Scale scores by 1/sqrt(d_head) to prevent dot products from
        # growing too large as d_head increases, which would push
        # softmax into regions with near-zero gradients.
        weight /= math.sqrt(self.d_head)

        # Convert raw scores to a probability distribution over keys.
        # Each query position now has a set of non-negative weights
        # summing to 1 that determines how much to attend to each key.
        # (Batch, n_heads, SeqLen, SeqLen)
        weight = F.softmax(weight, dim=-1)

        # Weighted sum of value vectors: each query position receives
        # a blend of value vectors weighted by the attention scores.
        # (Batch, n_heads, SeqLen, SeqLen) @
        # (Batch, n_heads, SeqLen, d_head)
        # -> (Batch, n_heads, SeqLen, d_head)
        output = weight @ v

        # Transpose back to (Batch, SeqLen, n_heads, d_head) so the
        # head and d_head dimensions are adjacent and can be merged.
        output = output.transpose(1, 2)

        # Merge the n_heads and d_head dimensions back into d_embed,
        # restoring the original (Batch, SeqLen, d_embed) shape.
        # reshape is safe here because the data is contiguous after
        # the transpose.
        output = output.reshape(input_shape)

        # Final linear projection mixes the head outputs together,
        # allowing the model to learn how to combine information from
        # all heads into a coherent representation.
        # (Batch, SeqLen, d_embed) -> (Batch, SeqLen, d_embed)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention module.

    Queries come from one modality (e.g. image latents) while keys and
    values come from another (e.g. text embeddings).  This is the
    mechanism that injects conditioning information (the CLIP text
    encoding) into the U-Net feature maps at every attention layer.
    """

    def __init__(
            self,
            n_heads: int,
            d_embed: int,
            d_cross: int,
            in_proj_bias: bool = True,
            out_proj_bias: bool = True
    ) -> None:
        """
        Initialize the cross-attention module.

        Parameters
        ----------
        n_heads : int
            Number of attention heads.
        d_embed : int
            Embedding dimension of the query source (latent features).
        d_cross : int
            Embedding dimension of the key/value source (e.g. 768 for
            CLIP text embeddings).
        in_proj_bias : bool
            Whether to include bias in the Q, K, V projections.
        out_proj_bias : bool
            Whether to include bias in the output projection.
        """
        super().__init__()

        # Project queries from the latent feature space (d_embed) to
        # d_embed.  Queries represent "what information is each spatial
        # position looking for?"
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)

        # Project keys from the conditioning context (d_cross) into the
        # same d_embed space as queries, so dot-product compatibility
        # scores can be computed.  Keys represent "what information does
        # each text token offer?"
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        # Project values from the conditioning context (d_cross) to
        # d_embed.  Values carry the actual content that will be read
        # out when a position attends to a text token.
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        # Output projection: mixes the information gathered by all
        # heads back into a single d_embed-dimensional vector.
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        # Store head count and per-head dimension for reshaping in forward().
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head cross-attention.

        Parameters
        ----------
        x : torch.Tensor
            Query source (latent features) of shape
            (Batch, SeqLen_Q, d_embed).
        y : torch.Tensor
            Key/value source (conditioning context) of shape
            (Batch, SeqLen_KV, d_cross).  For CLIP this is
            (Batch, 77, 768).

        Returns
        -------
        torch.Tensor
            Attended output of shape (Batch, SeqLen_Q, d_embed).
        """
        # Save the query shape to restore it at the end.
        # (Batch, SeqLen_Q, d_embed)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Target shape for splitting d_embed into n_heads * d_head.
        # -1 lets PyTorch infer SeqLen (works for both Q and KV lengths).
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Project x into queries: each spatial position formulates a
        # query describing what conditioning signal it needs.
        # (Batch, SeqLen_Q, d_embed) -> (Batch, SeqLen_Q, d_embed)
        q = self.q_proj(x)

        # Project the context y into keys and values.
        # Keys and values share the same source (text embeddings) but
        # are projected independently so the model can decouple "what
        # to match on" (keys) from "what to read out" (values).
        # (Batch, SeqLen_KV, d_cross) -> (Batch, SeqLen_KV, d_embed)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Reshape each projection from (Batch, SeqLen, d_embed) into
        # (Batch, SeqLen, n_heads, d_head) and transpose to
        # (Batch, n_heads, SeqLen, d_head) so batched matmul handles
        # every head in parallel.
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute raw attention scores: dot product between every query
        # position and every key position.
        # (Batch, n_heads, SeqLen_Q, d_head) @
        # (Batch, n_heads, d_head, SeqLen_KV)
        # -> (Batch, n_heads, SeqLen_Q, SeqLen_KV)
        weight = q @ k.transpose(-1, -2)

        # Scale by 1/sqrt(d_head) to keep the variance of the logits
        # stable regardless of d_head magnitude.
        weight /= math.sqrt(self.d_head)

        # Softmax converts raw scores into a probability distribution
        # over the key/value positions for each query.
        # No causal mask is needed: every spatial position is allowed
        # to attend to every text token.
        weight = F.softmax(weight, dim=-1)

        # Weighted sum of values: each query position receives a blend
        # of value vectors weighted by the attention probabilities.
        # (Batch, n_heads, SeqLen_Q, SeqLen_KV) @
        # (Batch, n_heads, SeqLen_KV, d_head)
        # -> (Batch, n_heads, SeqLen_Q, d_head)
        output = weight @ v

        # Transpose back to (Batch, SeqLen_Q, n_heads, d_head) and
        # call contiguous() because the subsequent view/reshape requires
        # memory-contiguous storage.
        output = output.transpose(1, 2).contiguous()

        # Merge the head and d_head dimensions back into d_embed,
        # restoring the original query shape (Batch, SeqLen_Q, d_embed).
        output = output.view(input_shape)

        # Final output projection: linearly combines the per-head
        # results, allowing the model to learn how to fuse information
        # gathered by different heads.
        # (Batch, SeqLen_Q, d_embed) -> (Batch, SeqLen_Q, d_embed)
        output = self.out_proj(output)

        return output

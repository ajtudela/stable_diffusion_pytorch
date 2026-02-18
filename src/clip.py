import torch
from torch import nn
from .attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """
    Combines token embeddings with learned positional embeddings to
    produce the input representation for the CLIP text transformer.
    """

    def __init__(self, n_vocab: int, n_embd: int, n_token: int) -> None:
        """
        Initialize the CLIP embedding layer.

        Parameters
        ----------
        n_vocab : int
            Size of the BPE vocabulary (49 408 for default CLIP).
        n_embd : int
            Dimensionality of each token embedding vector.
        n_token : int
            Maximum sequence length (number of positional embeddings).
        """
        super().__init__()

        # Lookup table that maps each token id to a
        # dense vector  of size n_embd.
        # Vocabulary size is 49 408 for the default CLIP tokenizer.
        self.token_embedding = nn.Embedding(n_vocab, n_embd)

        # Learnable positional embedding:
        # one vector per token position (max 77).
        # Initialised to zeros; the model learns absolute position information
        # during training so the transformer can distinguish token order.
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Embed tokens and add positional information.

        Parameters
        ----------
        tokens : torch.LongTensor
            Token ids of shape (Batch, Seq_len).

        Returns
        -------
        torch.Tensor
            Embedded representation of shape (Batch, Seq_len, Dim).
        """
        # Convert discrete token ids into continuous embedding vectors.
        # (Batch, Seq_len) -> (Batch, Seq_len, Dim)
        x = self.token_embedding(tokens)

        # Add position embedding (broadcast over the batch dimension).
        # This injects absolute positional information into every token.
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    """
    A single transformer block used in the CLIP text encoder.

    Follows a Pre-LN (pre-normalisation) architecture:
    LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual.
    """

    def __init__(self, n_head: int, n_embd: int) -> None:
        """
        Initialize a single CLIP transformer block.

        Parameters
        ----------
        n_head : int
            Number of attention heads.
        n_embd : int
            Embedding dimension (split equally across heads).
        """
        super().__init__()

        # Layer normalisation applied before self-attention (Pre-LN variant).
        # Stabilises training and allows deeper stacking of layers.
        self.layernorm_1 = nn.LayerNorm(n_embd)

        # Multi-head causal self-attention.
        # Each head attends over Dim // n_head dimensions, allowing the model
        # to capture different relationship
        # patterns in parallel.
        self.attention = SelfAttention(n_head, n_embd)

        # Second layer norm applied before the feed-forward network.
        self.layernorm_2 = nn.LayerNorm(n_embd)

        # Two-layer feed-forward network (FFN) with an inner expansion factor
        # of 4x. This gives the model a wider non-linear transformation
        # capacity at each position independently.
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, Seq_len, Dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (Batch, Seq_len, Dim).
        """
        # --- Self-Attention sub-block ---

        # Save input for the residual (skip) connection.
        # (Batch, Seq_len, Dim)
        residue = x

        # Pre-norm: normalise before attention to improve gradient flow.
        x = self.layernorm_1(x)

        # Causal (masked) self-attention: each token can only attend to
        # itself and previous tokens.  This is required because CLIP's text
        # encoder is auto-regressive during pre-training.
        x = self.attention(x, causal_mask=True)

        # Residual connection: adds the original input back, preventing
        # information loss and easing optimisation.
        x += residue

        # --- Feed-Forward sub-block ---

        # Save input for the second residual connection.
        residue = x

        # Pre-norm before the feed-forward network.
        x = self.layernorm_2(x)

        # Project up to 4 * Dim (expand). Increases representation capacity.
        x = self.linear_1(x)

        # QuickGELU activation: x * σ(1.702 * x).
        # An efficient approximation of GELU used in OpenAI's original CLIP.
        # The constant 1.702 was chosen so that QuickGELU closely matches
        # the exact GELU curve.
        x = x * torch.sigmoid(1.702 * x)

        # Project back down to Dim (contract).
        x = self.linear_2(x)

        # Second residual connection.
        x += residue

        return x


class CLIP(nn.Module):
    """
    Full CLIP text encoder.

    Consists of a token + position embedding layer, 12 stacked
    transformer blocks, and a final layer normalisation.
    Encodes a token sequence of length 77 into contextual embeddings
    of dimension 768, which condition the diffusion model.
    """

    def __init__(self) -> None:
        """
        Initialize the full CLIP text encoder.

        Uses hard-coded hyper-parameters matching the original
        OpenAI CLIP ViT-L/14 text encoder: 49 408 vocab size,
        768 embedding dim, 77 max tokens, and 12 transformer layers
        with 12 attention heads each.
        """
        super().__init__()

        # Token + positional embedding layer.
        # 49 408 = CLIP BPE vocab size, 768 = hidden dim,
        # 77 = max sequence length.
        self.embedding = CLIPEmbedding(49408, 768, 77)

        # Stack of 12 identical transformer layers (Pre-LN variant).
        # 12 attention heads, each of dimension 768 / 12 = 64.
        # nn.ModuleList registers every layer so their parameters are visible
        # to the optimiser and correctly handled by .to(device),
        # state_dict, etc.
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        # Final layer normalisation applied after the last transformer block.
        # Normalises the output embeddings before they are used as conditioning
        # signals for the diffusion U-Net.
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Encode a batch of token sequences into contextual embeddings.

        Parameters
        ----------
        tokens : torch.LongTensor
            Token ids of shape (Batch, 77).

        Returns
        -------
        torch.FloatTensor
            Contextual embeddings of shape (Batch, 77, 768).
        """
        # Ensure token dtype is long (int64), required by nn.Embedding.
        tokens = tokens.type(torch.long)  # type: ignore[assignment]

        # Embed tokens and add positional information.
        # (Batch, Seq_len) -> (Batch, Seq_len, 768)
        state = self.embedding(tokens)

        # Pass through each transformer layer sequentially.
        # Each layer refines the contextual representation by attending
        # to other positions and applying a non-linear transformation.
        for layer in self.layers:
            state = layer(state)

        # Final layer normalisation produces the output embeddings.
        # (Batch, Seq_len, 768)
        output = self.layernorm(state)

        return output

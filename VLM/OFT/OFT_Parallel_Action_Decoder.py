import torch
import torch.nn as nn


class ActionQueryBuilder(nn.Module):
    """
    Build placeholder action query tokens for parallel action decoding.
    In OFT, these act like empty slots that ask the model to predict a future action chunk.
    """

    def __init__(self, llm_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.query_tokens = nn.Parameter(torch.zeros(1, chunk_size, llm_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.query_tokens.expand(batch_size, -1, -1)


class ParallelActionDecoderBlock(nn.Module):
    """
    A small bidirectional transformer block for action-chunk decoding.
    This is a teaching version of the OFT idea: action queries are decoded in parallel,
    not token-by-token autoregressively.
    """

    def __init__(
        self,
        llm_dim: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, llm_dim),
        )
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Bidirectional attention: all action-query tokens can see one another.
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights


class OFTParallelActionDecoder(nn.Module):
    """
    A simplified OFT-style parallel decoder:
    multimodal context -> append action query slots -> predict a full action chunk in one pass.
    """

    def __init__(
        self,
        llm_dim: int = 1024,
        chunk_size: int = 8,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.llm_dim = llm_dim
        self.chunk_size = chunk_size
        self.query_builder = ActionQueryBuilder(llm_dim=llm_dim, chunk_size=chunk_size)
        self.layers = nn.ModuleList(
            [
                ParallelActionDecoderBlock(
                    llm_dim=llm_dim,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(llm_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        context_tokens: (B, context_len, llm_dim)

        Returns:
        - decoder_input: context tokens + action query tokens
        - action_hidden_states: final hidden states for the action-query region
        """
        batch_size = context_tokens.size(0)
        action_queries = self.query_builder(batch_size=batch_size)

        # OFT idea: insert empty action slots, then decode all future actions together.
        decoder_input = torch.cat([context_tokens, action_queries], dim=1)

        x = decoder_input
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)

        x = self.final_norm(x)
        action_hidden_states = x[:, -self.chunk_size :, :]

        return {
            "decoder_input": decoder_input,
            "full_hidden_states": x,
            "action_hidden_states": action_hidden_states,
            "attn_weights": all_attn_weights,
        }


def demo_parallel_decoder_shapes() -> None:
    batch_size = 2
    context_len = 260
    llm_dim = 512
    chunk_size = 8

    context_tokens = torch.randn(batch_size, context_len, llm_dim)

    decoder = OFTParallelActionDecoder(
        llm_dim=llm_dim,
        chunk_size=chunk_size,
        num_heads=8,
        num_layers=2,
        d_ff=1024,
        dropout=0.0,
    )

    outputs = decoder(context_tokens)

    print("context tokens shape      :", context_tokens.shape)
    print("decoder input shape       :", outputs["decoder_input"].shape)
    print("full hidden states shape  :", outputs["full_hidden_states"].shape)
    print("action hidden states shape:", outputs["action_hidden_states"].shape)
    print("layer1 attn shape         :", outputs["attn_weights"][0].shape)
    print()
    print("shape flow:")
    print(f"(B, {context_len}, {llm_dim}) [multimodal context]")
    print(f"-> (B, {context_len + chunk_size}, {llm_dim}) [append action queries]")
    print(f"-> (B, {chunk_size}, {llm_dim}) [select action chunk region]")


if __name__ == "__main__":
    demo_parallel_decoder_shapes()

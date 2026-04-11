import torch
import torch.nn as nn


class ActionUnnormalizer(nn.Module):
    """
    Map normalized continuous actions back to robot action ranges.
    """

    def __init__(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("action_mean", action_mean.view(1, 1, -1))
        self.register_buffer("action_std", action_std.view(1, 1, -1))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std + self.action_mean


class ContinuousActionHead(nn.Module):
    """
    Teaching version of the OFT continuous action head.
    It reads action hidden states from the decoder and predicts a full continuous action chunk.
    """

    def __init__(
        self,
        llm_dim: int,
        action_dim: int,
        hidden_dim: int = 2048,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()

        self.output_activation = output_activation
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        action_hidden_states: (B, chunk_size, llm_dim)
        returns:             (B, chunk_size, action_dim)
        """
        actions = self.mlp(action_hidden_states)

        # Many robot policies predict normalized actions and then unnormalize outside the head.
        if self.output_activation == "tanh":
            actions = torch.tanh(actions)

        return actions


def demo_continuous_action_head_shapes() -> None:
    batch_size = 2
    chunk_size = 8
    llm_dim = 512
    action_dim = 7

    action_hidden_states = torch.randn(batch_size, chunk_size, llm_dim)

    action_head = ContinuousActionHead(
        llm_dim=llm_dim,
        action_dim=action_dim,
        hidden_dim=1024,
        output_activation="tanh",
    )

    pred_actions = action_head(action_hidden_states)

    print("action hidden states shape :", action_hidden_states.shape)
    print("predicted actions shape    :", pred_actions.shape)
    print()
    print("shape flow:")
    print(f"(B, {chunk_size}, {llm_dim}) [action hidden states]")
    print(f"-> (B, {chunk_size}, {action_dim}) [continuous action chunk]")


if __name__ == "__main__":
    demo_continuous_action_head_shapes()

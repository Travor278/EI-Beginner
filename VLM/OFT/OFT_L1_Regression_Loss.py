import torch
import torch.nn as nn


class MaskedL1ActionLoss(nn.Module):
    """
    L1 regression loss for continuous action chunks.
    A mask is useful when some time steps in the chunk are padding or invalid.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        pred_actions:   (B, chunk_size, action_dim)
        target_actions: (B, chunk_size, action_dim)
        action_mask:    (B, chunk_size), optional
        """
        l1 = torch.abs(pred_actions - target_actions)

        if action_mask is not None:
            mask = action_mask.unsqueeze(-1).to(dtype=l1.dtype)
            l1 = l1 * mask
            denom = mask.sum().clamp_min(1.0) * pred_actions.size(-1)
            return l1.sum() / denom

        if self.reduction == "mean":
            return l1.mean()
        if self.reduction == "sum":
            return l1.sum()
        return l1


class OFTL1RegressionStep(nn.Module):
    """
    A tiny teaching wrapper:
    action hidden states -> continuous action head -> masked L1 loss.

    In the real OFT codebase, this would be wired into the full multimodal forward pass.
    Diffusion training would need a different target construction and iterative denoising path.
    """

    def __init__(
        self,
        action_head: nn.Module,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.action_head = action_head
        self.loss_fn = MaskedL1ActionLoss(reduction=reduction)

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_actions = self.action_head(action_hidden_states)
        loss = self.loss_fn(pred_actions, target_actions, action_mask)
        return {
            "pred_actions": pred_actions,
            "loss": loss,
        }


def demo_l1_regression_shapes() -> None:
    batch_size = 2
    chunk_size = 8
    llm_dim = 512
    action_dim = 7

    action_hidden_states = torch.randn(batch_size, chunk_size, llm_dim)
    target_actions = torch.randn(batch_size, chunk_size, action_dim)
    action_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    action_head = nn.Sequential(
        nn.Linear(llm_dim, 1024),
        nn.GELU(),
        nn.Linear(1024, action_dim),
    )

    train_step = OFTL1RegressionStep(action_head=action_head)
    outputs = train_step(
        action_hidden_states=action_hidden_states,
        target_actions=target_actions,
        action_mask=action_mask,
    )

    print("action hidden states shape :", action_hidden_states.shape)
    print("target actions shape       :", target_actions.shape)
    print("predicted actions shape    :", outputs["pred_actions"].shape)
    print("l1 loss                    :", outputs["loss"].item())
    print()
    print("shape flow:")
    print(f"(B, {chunk_size}, {llm_dim}) -> (B, {chunk_size}, {action_dim}) -> scalar loss")


if __name__ == "__main__":
    demo_l1_regression_shapes()

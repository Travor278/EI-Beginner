"""
Teaching implementation of the core pi0 flow-matching action path.

This file is intentionally aligned with the official open-source
`Physical-Intelligence/openpi` implementation, but simplified:

- It assumes image/language prefix tokens are already embedded.
- It keeps a small pedagogical transformer instead of the full Gemma /
  PaliGemma stack used in the official repo.
- It mirrors the official pi0 time convention used in code:
  t = 1.0 means pure noise, t = 0.0 means the target action distribution.
- It mirrors the official loss construction:
  x_t = t * noise + (1 - t) * actions
  u_t = noise - actions

Official references checked against openpi commit:
e4429ad35ec380842dc72b4074735cf3e8a503c2
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """
    Mirror the official openpi attention-mask construction.

    input_mask: (B, N) bool
    mask_ar:    (N,) or (B, N) bool/int

    Semantics:
    - False means "same attention block as the previous token"
    - True means "start a new block that previous tokens cannot attend to"

    Returned mask follows nn.MultiheadAttention convention:
    True means "do not attend".
    """

    if input_mask.dim() != 2:
        raise ValueError("Expected input_mask shape: (B, N).")
    if mask_ar.dim() not in (1, 2):
        raise ValueError("Expected mask_ar shape: (N,) or (B, N).")

    if mask_ar.dim() == 1:
        mask_ar = mask_ar.unsqueeze(0).expand(input_mask.size(0), -1)
    if mask_ar.shape != input_mask.shape:
        raise ValueError("mask_ar must broadcast to input_mask shape.")

    cumsum = torch.cumsum(mask_ar.to(dtype=torch.int64), dim=1)
    can_attend = cumsum[:, None, :] <= cumsum[:, :, None]
    valid = input_mask[:, None, :] & input_mask[:, :, None]
    full_mask = can_attend & valid

    # nn.MultiheadAttention expects True where attention is blocked.
    # We use the same mask for every item in the batch when batch_first=True by
    # requiring that all rows are identical. For this teaching file, we enforce
    # equal masks across the batch.
    if not torch.equal(full_mask, full_mask[:1].expand_as(full_mask)):
        raise ValueError("This teaching implementation expects identical attention masks across the batch.")

    return ~full_mask[0]


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """
    Match the role of openpi.models.pi0.posemb_sincos.
    """

    if pos.dim() != 1:
        raise ValueError("Expected pos shape: (B,).")
    if embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be divisible by 2.")

    fraction = torch.linspace(
        0.0,
        1.0,
        embedding_dim // 2,
        device=pos.device,
        dtype=pos.dtype,
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = pos.unsqueeze(1) / period.unsqueeze(0)
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixedExpertTransformerBlock(nn.Module):
    """
    Small pedagogical analogue of the official "shared attention + separate expert params" idea.

    - All tokens interact through the same self-attention layer.
    - Prefix tokens use one FFN branch.
    - Robot-specific suffix tokens use another FFN branch.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prefix_ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.suffix_ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        suffix_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        prefix_out = self.prefix_ffn(x)
        suffix_out = self.suffix_ffn(x)
        routed = torch.where(suffix_token_mask.view(1, -1, 1), suffix_out, prefix_out)

        x = self.norm2(x + self.dropout2(routed))
        return x


class PiZeroFlowActionExpert(nn.Module):
    """
    Teaching model that mirrors the official pi0 prefix/suffix decomposition.

    Prefix:
    - already-embedded context tokens standing in for image + language tokens

    Suffix:
    - one state token
    - one action chunk, where every action token gets action + timestep information

    Output:
    - predicted vector field for the action chunk
    """

    def __init__(
        self,
        context_dim: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        self.context_proj = nn.Linear(context_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_in_proj = nn.Linear(action_dim, d_model)
        self.action_time_mlp_in = nn.Linear(2 * d_model, d_model)
        self.action_time_mlp_out = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                MixedExpertTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.action_out_proj = nn.Linear(d_model, action_dim)

    def embed_prefix(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context_tokens.dim() != 3:
            raise ValueError("Expected context_tokens shape: (B, context_len, context_dim).")

        prefix_tokens = self.context_proj(context_tokens)
        batch_size, context_len, _ = prefix_tokens.shape
        if context_mask is None:
            context_mask = torch.ones(
                batch_size,
                context_len,
                dtype=torch.bool,
                device=context_tokens.device,
            )
        prefix_ar_mask = torch.zeros(context_len, dtype=torch.bool, device=context_tokens.device)
        return prefix_tokens, context_mask, prefix_ar_mask

    def embed_suffix(
        self,
        robot_state: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if robot_state.dim() != 2:
            raise ValueError("Expected robot_state shape: (B, state_dim).")
        if noisy_actions.dim() != 3:
            raise ValueError("Expected noisy_actions shape: (B, action_horizon, action_dim).")
        if noisy_actions.size(1) != self.action_horizon:
            raise ValueError("noisy_actions horizon must match action_horizon.")
        if time.dim() != 1:
            raise ValueError("Expected time shape: (B,).")

        state_token = self.state_proj(robot_state).unsqueeze(1)
        action_tokens = self.action_in_proj(noisy_actions)

        time_emb = posemb_sincos(
            time,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        time_tokens = time_emb.unsqueeze(1).expand(-1, self.action_horizon, -1)
        action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = F.silu(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)

        suffix_tokens = torch.cat([state_token, action_time_tokens], dim=1)
        suffix_mask = torch.ones(
            suffix_tokens.size(0),
            suffix_tokens.size(1),
            dtype=torch.bool,
            device=suffix_tokens.device,
        )

        # Match the official pi0 suffix grouping:
        # - one new block for the state token
        # - one new block for the first action token
        # - remaining action tokens stay in the same block
        suffix_ar_mask = torch.tensor(
            [True] + [True] + [False] * (self.action_horizon - 1),
            dtype=torch.bool,
            device=suffix_tokens.device,
        )
        return suffix_tokens, suffix_mask, suffix_ar_mask

    def forward_vector_field(
        self,
        context_tokens: torch.Tensor,
        robot_state: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(context_tokens, context_mask)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(robot_state, noisy_actions, time)

        tokens = torch.cat([prefix_tokens, suffix_tokens], dim=1)
        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        mask_ar = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, mask_ar)

        suffix_token_mask = torch.zeros(tokens.size(1), dtype=torch.bool, device=tokens.device)
        suffix_token_mask[prefix_tokens.size(1) :] = True

        x = tokens
        for block in self.blocks:
            x = block(x=x, attn_mask=attn_mask, suffix_token_mask=suffix_token_mask)

        x = self.final_norm(x)
        action_hidden = x[:, -self.action_horizon :, :]
        pred_vector_field = self.action_out_proj(action_hidden)

        return {
            "tokens": x,
            "input_mask": input_mask,
            "mask_ar": mask_ar,
            "attn_mask": attn_mask,
            "action_hidden": action_hidden,
            "pred_vector_field": pred_vector_field,
        }


def sample_time(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Match the official openpi time sampling:
    Beta(1.5, 1.0) scaled into (0.001, 1.0].
    """

    base = torch.distributions.Beta(1.5, 1.0).sample((batch_size,)).to(device=device, dtype=dtype)
    return base * 0.999 + 0.001


def build_flow_targets(
    clean_actions: torch.Tensor,
    time: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match the official openpi pi0 training target:
    x_t = t * noise + (1 - t) * actions
    u_t = noise - actions
    """

    if clean_actions.dim() != 3:
        raise ValueError("Expected clean_actions shape: (B, action_horizon, action_dim).")
    if time.dim() != 1:
        raise ValueError("Expected time shape: (B,).")

    noise = torch.randn_like(clean_actions) if noise is None else noise
    time_expanded = time.view(-1, 1, 1)
    x_t = time_expanded * noise + (1.0 - time_expanded) * clean_actions
    u_t = noise - clean_actions
    return noise, x_t, u_t


class PiZeroFlowMatchingStep(nn.Module):
    """
    Training wrapper aligned with the official pi0 compute_loss path.
    """

    def __init__(self, model: PiZeroFlowActionExpert) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        context_tokens: torch.Tensor,
        robot_state: torch.Tensor,
        clean_actions: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = clean_actions.size(0)
        time = sample_time(batch_size, clean_actions.device, clean_actions.dtype) if time is None else time
        noise, x_t, u_t = build_flow_targets(clean_actions, time, noise)

        outputs = self.model.forward_vector_field(
            context_tokens=context_tokens,
            robot_state=robot_state,
            noisy_actions=x_t,
            time=time,
            context_mask=context_mask,
        )
        v_t = outputs["pred_vector_field"]
        loss_per_timestep = torch.mean((v_t - u_t) ** 2, dim=-1)
        loss = loss_per_timestep.mean()

        return {
            "time": time,
            "noise": noise,
            "x_t": x_t,
            "u_t": u_t,
            "v_t": v_t,
            "loss_per_timestep": loss_per_timestep,
            "loss": loss,
            **outputs,
        }


@torch.no_grad()
def sample_action_chunk(
    model: PiZeroFlowActionExpert,
    context_tokens: torch.Tensor,
    robot_state: torch.Tensor,
    *,
    context_mask: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    num_steps: int = 10,
) -> torch.Tensor:
    """
    Match the official openpi sampling convention used in code:
    - start at t = 1 with Gaussian noise
    - integrate toward t = 0 with Euler updates

    The official implementation caches the prefix KV state.
    This teaching file recomputes the full forward pass each step for simplicity.
    """

    batch_size = context_tokens.size(0)
    device = context_tokens.device
    dtype = context_tokens.dtype

    if noise is None:
        noise = torch.randn(
            batch_size,
            model.action_horizon,
            model.action_dim,
            device=device,
            dtype=dtype,
        )

    x_t = noise
    dt = -1.0 / num_steps
    time = torch.ones(batch_size, device=device, dtype=dtype)

    for _ in range(num_steps):
        v_t = model.forward_vector_field(
            context_tokens=context_tokens,
            robot_state=robot_state,
            noisy_actions=x_t,
            time=time,
            context_mask=context_mask,
        )["pred_vector_field"]
        x_t = x_t + dt * v_t
        time = time + dt

    return x_t


def demo_pi0_flow_action_expert() -> None:
    batch_size = 2
    context_len = 196
    context_dim = 768
    state_dim = 18
    action_horizon = 50
    action_dim = 18

    context_tokens = torch.randn(batch_size, context_len, context_dim)
    context_mask = torch.ones(batch_size, context_len, dtype=torch.bool)
    robot_state = torch.randn(batch_size, state_dim)
    clean_actions = torch.randn(batch_size, action_horizon, action_dim)

    model = PiZeroFlowActionExpert(
        context_dim=context_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        d_model=256,
        d_ff=1024,
        num_heads=8,
        num_layers=3,
        dropout=0.0,
    )
    train_step = PiZeroFlowMatchingStep(model=model)

    train_outputs = train_step(
        context_tokens=context_tokens,
        context_mask=context_mask,
        robot_state=robot_state,
        clean_actions=clean_actions,
    )
    sampled_actions = sample_action_chunk(
        model=model,
        context_tokens=context_tokens,
        context_mask=context_mask,
        robot_state=robot_state,
        num_steps=10,
    )

    print("context tokens shape     :", context_tokens.shape)
    print("context mask shape       :", context_mask.shape)
    print("robot state shape        :", robot_state.shape)
    print("clean actions shape      :", clean_actions.shape)
    print("x_t shape                :", train_outputs["x_t"].shape)
    print("u_t shape                :", train_outputs["u_t"].shape)
    print("v_t shape                :", train_outputs["v_t"].shape)
    print("loss/timestep shape      :", train_outputs["loss_per_timestep"].shape)
    print("attention mask shape     :", train_outputs["attn_mask"].shape)
    print("sampled actions shape    :", sampled_actions.shape)
    print("mean loss                :", train_outputs["loss"].item())
    print()
    print("shape flow:")
    print(f"(B, {context_len}, {context_dim}) [embedded image/language prefix]")
    print(f"+ (B, {state_dim}) [robot state]")
    print(f"+ (B, {action_horizon}, {action_dim}) [clean action chunk]")
    print("-> sample t and Gaussian noise")
    print("-> x_t = t * noise + (1 - t) * actions")
    print("-> predict v_t with prefix/suffix attention")
    print("-> supervise against u_t = noise - actions")
    print("-> integrate from t=1 to t=0 during sampling")


if __name__ == "__main__":
    demo_pi0_flow_action_expert()

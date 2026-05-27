from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import math
import torch


class DelayAwareEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, act_hist_dim, delay_dim, features_dim=256):
        super().__init__(observation_space, features_dim)

        flat_obs_dim = observation_space.shape[0]
        self.obs_dim = flat_obs_dim - act_hist_dim - delay_dim
        self.act_hist_dim = act_hist_dim
        self.delay_dim = delay_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        fused_input_dim = 64 + self.act_hist_dim + self.delay_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_input_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),

        )

    def forward(self, obs):
        """
        Expected obs = [o_t, a_hist, delay]
        where obs is a concatenated tensor already prepared in env wrapper.
        """

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        o_t = obs[..., : self.obs_dim]
        a_hist = obs[..., self.obs_dim : self.obs_dim + self.act_hist_dim]
        delay = obs[..., -self.delay_dim :]

        o_emb = self.obs_encoder(o_t)
        fused = th.cat([o_emb, a_hist, delay], dim=-1)
        features = self.fusion(fused)

        if features.shape[0] == 1:
            features = features.squeeze(0)
        return features




class DelayAwareGRUEncoder(BaseFeaturesExtractor):
    """
    obs = [o_t, action_history (T * act_dim), delay]
    We reshape action_history -> (batch, T, act_dim) and process with GRU.
    """
    def __init__(self, observation_space, act_dim, hist_len, delay_dim,
                 features_dim=256, gru_hidden=64):
        super().__init__(observation_space, features_dim)

        flat_obs = observation_space.shape[0]

        # full action-history dimension in flattened obs
        self.act_hist_dim = act_dim * hist_len
        self.obs_dim = flat_obs - self.act_hist_dim - delay_dim

        self.act_dim = act_dim
        self.hist_len = hist_len
        self.delay_dim = delay_dim

        # -------------------------------
        # 1) Encode observation (o_t)
        # -------------------------------
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # -------------------------------
        # 2) GRU for action history
        # -------------------------------
        self.gru = nn.GRU(
            input_size=act_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # -------------------------------
        # 3) Fusion MLP
        # -------------------------------
        fused_dim = 64 + gru_hidden + delay_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )


    def forward(self, obs):
        """
        obs = [o_t, a_hist_flat, delay]
        a_hist_flat has shape (batch, hist_len * act_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        B = obs.size(0)

        o_t = obs[:, :self.obs_dim]

        a_hist_flat = obs[:, self.obs_dim : self.obs_dim + self.act_hist_dim]
        delay = obs[:, -self.delay_dim:]

        # ---- Encode obs ----
        o_emb = self.obs_encoder(o_t)

        # ---- Prepare action history ----
        # reshape: (B, hist_len, act_dim)
        a_hist = a_hist_flat.view(B, self.hist_len, self.act_dim)

        # ---- GRU ----
        _, h_last = self.gru(a_hist)   # h_last: (1, B, hidden)
        act_emb = h_last.squeeze(0)    # (B, hidden)

        # ---- Fuse ----
        fused = th.cat([o_emb, act_emb, delay], dim=-1)
        features = self.fusion(fused)

        return features


class GRUEncoder(BaseFeaturesExtractor):
    """
    GRU encoder for no-delay scenarios.
    Processes entity observations as a sequence using GRU.
    Handles both 2D observation space (31, 3) and flattened observations.
    """
    def __init__(self, observation_space, entity_seq_len, entity_feat_dim,
                 features_dim=256, gru_hidden=64):
        super().__init__(observation_space, features_dim)

        self.entity_seq_len = entity_seq_len
        self.entity_feat_dim = entity_feat_dim
        
        # Check observation space - could be 2D (31, 3) or flattened (93,)
        # Stable-baselines3 may flatten 2D Box spaces when passing to feature extractor
        obs_shape = observation_space.shape
        if len(obs_shape) == 2:
            # 2D: (entity_seq_len, entity_feat_dim)
            if obs_shape[0] != entity_seq_len or obs_shape[1] != entity_feat_dim:
                raise ValueError(
                    f"GRUEncoder expected obs shape ({entity_seq_len}, {entity_feat_dim}), "
                    f"got {obs_shape}"
                )
            self.expected_flat_dim = entity_seq_len * entity_feat_dim
        elif len(obs_shape) == 1:
            # Already flattened
            if obs_shape[0] != entity_seq_len * entity_feat_dim:
                raise ValueError(
                    f"GRUEncoder expected flat obs dim {entity_seq_len * entity_feat_dim}, "
                    f"got {obs_shape[0]}"
                )
            self.expected_flat_dim = obs_shape[0]
        else:
            raise ValueError(
                f"GRUEncoder expected 1D or 2D observation space, "
                f"got shape {obs_shape}"
            )

        # -------------------------------
        # GRU for entity sequence
        # -------------------------------
        self.gru = nn.GRU(
            input_size=entity_feat_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # -------------------------------
        # Output projection
        # -------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        """
        obs: Can be:
        - 1D flattened: (batch, entity_seq_len * entity_feat_dim) or (entity_seq_len * entity_feat_dim,)
        - 2D: (batch, entity_seq_len, entity_feat_dim) or (entity_seq_len, entity_feat_dim)
        Stable-baselines3 typically flattens 2D Box spaces, so we expect 1D input.
        """
        # Handle single sample (no batch dimension)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        B = obs.size(0)
        
        # Reshape to (B, entity_seq_len, entity_feat_dim)
        if obs.dim() == 2:
            # Flattened input: (B, entity_seq_len * entity_feat_dim)
            entity_seq = obs.view(B, self.entity_seq_len, self.entity_feat_dim)
        elif obs.dim() == 3:
            # Already in sequence format: (B, entity_seq_len, entity_feat_dim)
            entity_seq = obs
        else:
            raise ValueError(f"Unexpected obs dimension: {obs.dim()}")

        # ---- GRU ----
        _, h_last = self.gru(entity_seq)   # h_last: (1, B, hidden)
        seq_emb = h_last.squeeze(0)        # (B, hidden)

        # ---- Project to features ----
        features = self.fusion(seq_emb)

        return features


# -----------------------------
# RMSNorm (better for tiny models)
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.size(-1)))
        return (x / (rms + self.eps)) * self.weight


# -----------------------------
# Multi-head attention (tiny friendly)
# -----------------------------
class TinySelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.size()

        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape into heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention weights
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = torch.softmax(att, dim=-1)

        out = att @ v  # (B, heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# -----------------------------
# Transformer Block (tiny)
# -----------------------------
class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = TinySelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x):
        # attention
        x = x + self.attn(self.norm1(x))
        # feedforward
        x = x + self.ffn(self.norm2(x))
        return x


# -----------------------------
# Final TinyTransformer Model
# -----------------------------
class TinyTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=32,
        n_heads=1,
        ffn_hidden=64,
        n_layers=1,
        max_len=256,
        use_pos_emb=True,
    ):
        super().__init__()

        self.embed = nn.Linear(input_dim, d_model)

        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.layers = nn.ModuleList([
            TinyTransformerBlock(d_model, n_heads, ffn_hidden)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        B, T, _ = x.shape

        x = self.embed(x)

        if self.use_pos_emb:
            x = x + self.pos_emb[:, :T, :]

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)




class TinyTransformerEncoder(BaseFeaturesExtractor):
    """
    SB3-compatible encoder that uses TinyTransformer for entity tokens while
    processing action history and delay with lightweight MLPs before fusion.
    """

    def __init__(
        self,
        observation_space,
        entity_seq_len,
        entity_feat_dim,
        act_hist_len,
        act_dim,
        delay_dim=1,
        d_model=32,
        n_heads=1,
        ffn_hidden=64,
        n_layers=1,
        features_dim=128,
    ):
        super().__init__(observation_space, features_dim)

        self.entity_seq_len = entity_seq_len
        self.entity_feat_dim = entity_feat_dim
        self.act_hist_len = act_hist_len
        self.act_dim = act_dim
        self.delay_dim = delay_dim

        self.entity_flat_dim = entity_seq_len * entity_feat_dim
        self.act_hist_flat_dim = act_hist_len * act_dim

        expected_dim = self.entity_flat_dim + self.act_hist_flat_dim + delay_dim
        if observation_space.shape[0] != expected_dim:
            raise ValueError(
                f"TinyTransformerEncoder expected flat obs dim {expected_dim}, "
                f"got {observation_space.shape[0]}"
            )

        self.transformer = TinyTransformer(
            input_dim=entity_feat_dim,
            d_model=d_model,
            n_heads=n_heads,
            ffn_hidden=ffn_hidden,
            n_layers=n_layers,
            max_len=entity_seq_len,
        )

        # Simple encoders for action history and delay scalar.
        self.action_encoder = nn.Sequential(
            nn.Linear(self.act_hist_flat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.delay_encoder = nn.Sequential(
            nn.Linear(self.delay_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        fused_input = d_model * 3  # entity summary + action emb + delay emb
        self.out = nn.Sequential(
            nn.Linear(fused_input, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        """
        obs shape: (B, flat_dim) where flat_dim = entities + action hist + delay.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        B = obs.shape[0]
        entities_flat = obs[:, : self.entity_flat_dim]
        act_hist_flat = obs[
            :, self.entity_flat_dim : self.entity_flat_dim + self.act_hist_flat_dim
        ]
        delay = obs[:, -self.delay_dim :]

        entity_tokens = entities_flat.view(B, self.entity_seq_len, self.entity_feat_dim)
        entity_features = self.transformer(entity_tokens)
        entity_summary = entity_features[:, -1, :]

        action_emb = self.action_encoder(act_hist_flat)
        delay_emb = self.delay_encoder(delay)

        fused = th.cat([entity_summary, action_emb, delay_emb], dim=-1)
        return self.out(fused)


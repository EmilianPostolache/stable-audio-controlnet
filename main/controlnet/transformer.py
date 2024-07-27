import torch
from stable_audio_tools.models.transformer import checkpoint, ScaledSinusoidalEmbedding, RotaryEmbedding, \
    AbsolutePositionalEmbedding, TransformerBlock
from torch import nn


class ContinuousTransformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            *,
            dim_in=None,
            dim_out=None,
            dim_heads=64,
            cross_attend=False,
            cond_token_dim=None,
            global_cond_dim=None,
            causal=False,
            rotary_pos_emb=True,
            zero_init_branch_outputs=True,
            conformer=False,
            use_sinusoidal_emb=False,
            use_abs_pos_emb=False,
            abs_pos_emb_max_length=10000,
            **kwargs
    ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    cross_attend=cross_attend,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )

    def forward(
            self,
            x,
            mask=None,
            prepend_embeds=None,
            prepend_mask=None,
            global_cond=None,
            return_info=False,
            controlnet_embeds=None,
            **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim=-2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device=device, dtype=torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length),
                                                                                        device=device, dtype=torch.bool)

                mask = torch.cat((prepend_mask, mask), dim=-1)

        # Attention layers

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        # Iterate over the transformer layers
        for i, layer in enumerate(self.layers):
            # x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
            if controlnet_embeds and i < len(controlnet_embeds):
                x = x + controlnet_embeds[i]
            x = checkpoint(layer, x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

        x = self.project_out(x)

        if return_info:
            return x, info

        return x

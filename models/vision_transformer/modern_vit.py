"""
Modern Vision Transformer with xformers memory-efficient attention.
Supports sequence packing, register tokens, and masking for DINOv2.

Optional looped (weight-tied recurrent-depth) backbone variant: when
`looped_T_max > 0`, the standard stack of `depth` unique blocks is replaced
by a shared stack of `looped_L` blocks applied `looped_T_max` times. See
`shared_stack.SharedStack` and `LOOPED_DINOV2.md` for details.
"""

import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchDropout, trunc_normal_

import xformers.ops as xops
from xformers.ops import fmha

from timm.models._manipulate import checkpoint_seq

from .shared_stack import SharedStack

# Cache for attention bias to avoid recomputation
attn_bias_cache = {}


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Per-sample stochastic depth. No-op in eval or when drop_prob=0."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth (drops whole residual branch)."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_attn_bias_and_cat(x_list, branges=None):
    """
    Pack multiple sequences and create block-diagonal attention mask.
    
    Args:
        x_list: List of tensors, each [B,N,D], N can differ
        branges: Optional batch ranges for stochastic depth
        
    Returns:
        attn_bias: Block diagonal matrix (BlockDiagonalMask)
        cat_tensors: Packed tensor [1, total_tokens, D]
    """
    batch_sizes = [x.shape[0] for x in x_list]
    
    # Create tuple of shapes for caching
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    
    if all_shapes not in attn_bias_cache:
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
    
    # Concatenate all tensors
    tensors_bs1 = tuple(x.reshape(1, -1, x.shape[-1]) for x in x_list)
    cat_tensors = torch.cat(tensors_bs1, dim=1)
    
    return attn_bias_cache[all_shapes], cat_tensors


class PatchEmbed(nn.Module):
    """Convert image into patch embeddings with optional dual normalization."""
    def __init__(
        self, 
        img_size, 
        patch_size, 
        in_channels=3, 
        embed_dim=768, 
        dual_norm=False, 
        norm_layer=None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size)**2
        self.dual_norm = dual_norm
        
        self.project = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        if dual_norm and norm_layer is not None:
            self.pre_norm = norm_layer(in_channels)
            self.post_norm = norm_layer(embed_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.dual_norm:
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = self.pre_norm(x_flat)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.project(x)
        x = x.flatten(2).transpose(1, 2)
        
        if self.dual_norm:
            x = self.post_norm(x)
            
        return x


class SwiGLUFFN(nn.Module):
    """Optimized SwiGLU implementation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU with optimized hidden dimension sizing."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class Mlp(nn.Module):
    """Standard MLP FFN (Linear -> act -> Linear). DINOv2 ssl_default uses this
    in place of SwiGLU. Drop-in for `mlp_layer`: same call signature as
    SwiGLUFFNFused (in_features / hidden_features / out_features / drop / bias);
    act_layer defaults to GELU since the TransformerBlock call does not pass it."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with xformers attention and SwiGLU MLP."""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=None,
        mlp_bias=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, bias=False)
            self.k_norm = nn.LayerNorm(self.head_dim, bias=False)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        
        mlp_layer = mlp_layer or SwiGLUFFNFused
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=proj_drop,
            bias=mlp_bias,
        )
        
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x, attn_bias=None):
        """
        Args:
            x: Either [B,N,D] for regular or [1,total_tokens,D] for packed
            attn_bias: None for regular, BlockDiagonalMask for packed
        """
        shortcut = x
        x = self.norm1(x)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        
        x = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            scale=self.scale
        )
        
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.gamma_1 is not None:
            x = shortcut + self.drop_path(self.gamma_1 * x)
        else:
            x = shortcut + self.drop_path(x)
        
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class VisionTransformer(nn.Module):
    """
    Modern Vision Transformer with:
    - XFormers memory-efficient attention
    - SwiGLU MLP activation
    - Dynamic position embeddings
    - Register tokens
    - Mask token for DINOv2 training
    - Sequence packing support
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        mlp_bias=True,
        qk_norm=False,
        dual_norm=False,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        drop_path_uniform=False,
        weight_init="",
        norm_layer=None,
        act_layer=None,
        block_fn=TransformerBlock,
        mlp_layer=SwiGLUFFNFused,
        num_register_tokens=4,
        looped_T_max: int = 0,
        looped_L: Optional[int] = None,
        layerscale_init=None,
    ):
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0 
        self.no_embed_class = no_embed_class
        self.numregisters = num_register_tokens

        # Mask token for DINOv2
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
            dual_norm=dual_norm,
            norm_layer=norm_layer,
        )

        self.grad_checkpointing = False
        
        if isinstance(img_size, tuple):
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            num_patches = (img_size // patch_size) ** 2

        # Token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        # Position embeddings
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        
        # Dropouts
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.patch_drop = PatchDropout(
            patch_drop_rate,
            num_prefix_tokens=self.num_prefix_tokens,
        ) if patch_drop_rate > 0 else nn.Identity()
        
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Looped (weight-tied) backbone: the unique block count is `looped_L`
        # (default = `depth`) and the shared stack is applied `looped_T_max`
        # times per forward. When `looped_T_max == 0`, behavior is unchanged.
        self.looped_T_max = int(looped_T_max) if looped_T_max else 0
        if self.looped_T_max > 0:
            self.looped_L = int(looped_L) if looped_L is not None else int(depth)
            block_count = self.looped_L
        else:
            self.looped_L = None
            block_count = int(depth)

        # Stochastic depth (one rate per unique block; the shared stack reuses
        # the same per-block drop_path across recursion steps, which is
        # equivalent to per-recursion-step stochastic depth at the rate of the
        # underlying block — see __doc__ at top of file).
        if drop_path_uniform:
            dpr = [drop_path_rate] * block_count   # canonical DINOv2: flat across depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, block_count)]  # CaiT ramp

        # LayerScale initialization (effective depth still drives the choice
        # of init values in looped mode, since T_max * L is what the backbone
        # represents in operation count).
        effective_depth = self.looped_T_max * block_count if self.looped_T_max > 0 else block_count
        if layerscale_init is not None:
            layer_init_values = [layerscale_init] * block_count
        else:
            layer_init_values = []
            for _ in range(block_count):
                if effective_depth < 18:
                    layer_init_values.append(0.1)
                elif effective_depth < 24:
                    layer_init_values.append(1e-5)
                else:
                    layer_init_values.append(1e-6)

        # Transformer blocks. Use ModuleList in looped mode so SharedStack
        # owns the iteration order; nn.Sequential is fine in the standard
        # mode where the stack is just a forward pass through unique blocks.
        block_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                mlp_bias=mlp_bias,
                qk_norm=qk_norm,
                init_values=layer_init_values[i],
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(block_count)
        ]
        if self.looped_T_max > 0:
            self.blocks = nn.ModuleList(block_list)
            self.shared_stack = SharedStack(
                blocks=self.blocks,
                embed_dim=embed_dim,
                T_max=self.looped_T_max,
                norm_layer=norm_layer,
            )
        else:
            self.blocks = nn.Sequential(*block_list)
            self.shared_stack = None

        # Final normalization
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        if weight_init != "skip":
            self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        nn.init.normal_(self.mask_token, std=1e-6)
        
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "register_tokens", "mask_token"}
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def interpolate_pos_embed(self, x, h, w):
        """Interpolate position embeddings for different image sizes."""
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if num_patches == N:
            return self.pos_embed
            
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        
        w0, h0 = w0 + 0.1, h0 + 0.1
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens_with_masks(self, x, token_masks=None):
        """Prepare tokens with optional masking for DINOv2."""
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        if token_masks is not None:
            x = torch.where(token_masks.unsqueeze(-1), 
                            self.mask_token.to(x.dtype).unsqueeze(0), 
                            x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_embed(x, H, W)
        
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((x[:, 0:1], reg_tokens, x[:, 1:]), dim=1)
        
        return self.pos_drop(x)

    def prepare_tokens(self, x):
        """Prepare tokens with register tokens and position embeddings."""
        B, C, H, W = x.shape 
        
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1) if self.cls_token is not None else torch.zeros(B, 0, self.embed_dim, device=x.device)
        x = torch.cat((cls_token, x), dim=1) if self.cls_token is not None else x
        
        pos_embed = self.interpolate_pos_embed(x, H, W)
        x = x + pos_embed
        
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        if self.cls_token is not None:
            x = torch.cat((x[:, 0:1], reg_tokens, x[:, 1:]), dim=1)
        else:
            x = torch.cat((reg_tokens, x), dim=1)
        
        return self.pos_drop(x)

    def get_intermediate_layers(self, x):
        """Extract features at specified points in the network."""
        if self.looped_T_max > 0:
            # Quartile extraction by unique-block index doesn't carry the
            # same meaning under weight tying — `i in extraction_points` runs
            # only over the L unique blocks of one recursion step, not over
            # the T_max * L effective depth. For looped models, callers
            # should pull intermediate states from the shared stack's
            # per-recursion-step outputs instead.
            raise NotImplementedError(
                "get_intermediate_layers is not defined for looped backbones "
                "(looped_T_max > 0). Use shared_stack(z0, return_per_step=True) "
                "and pick the recursion-step indices you need."
            )

        x = self.prepare_tokens(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        features = []
        total_blocks = len(self.blocks)
        extraction_points = [
            (total_blocks // 4) - 1,
            (total_blocks // 2) - 1,
            (3 * total_blocks // 4) - 1,
            total_blocks - 1
        ]

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in extraction_points:
                features.append(x)

        return features
    
    def forward_features(self, x):
        """Forward pass through features."""
        x = self.prepare_tokens(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if self.looped_T_max > 0:
            # Looped backbone: run T_max recursion steps through the shared
            # stack, return only the final-step state. Used by inference and
            # by the EMA teacher (which always runs to T_max).
            x = self.shared_stack(
                x,
                attn_bias=None,
                grad_checkpoint=self.grad_checkpointing,
                return_per_step=False,
            )
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def forward_features_list(self, x, masks_list):
        """
        Process multiple crops using sequence packing.

        Args:
            x: list of image tensors
            masks_list: list of mask tensors (can be None)

        Returns:
            List of output dictionaries, one per crop type. In looped mode
            this returns only the final-step (z_T_max) outputs, preserving the
            non-looped contract; use `forward_features_list_per_step` to get
            per-recursion-step outputs.
        """
        x_packed = self._prepare_packed(x, masks_list)

        if self.looped_T_max > 0:
            # Pack once to a single tensor so the shared stack runs T_max
            # times on the full packed sequence with the same attn_bias.
            attn_bias, x_cat = get_attn_bias_and_cat(x_packed)
            x_cat = self.shared_stack(
                x_cat,
                attn_bias=attn_bias,
                grad_checkpoint=self.grad_checkpointing,
                return_per_step=False,
            )
            x_packed = attn_bias.split(x_cat)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            # Checkpoint-friendly version
            for blk in self.blocks:
                attn_bias, x_cat = get_attn_bias_and_cat(x_packed)
                x_cat = torch.utils.checkpoint.checkpoint(
                    blk,
                    x_cat,
                    attn_bias,
                    use_reentrant=False
                )
                x_packed = attn_bias.split(x_cat)
        else:
            for blk in self.blocks:
                attn_bias, x_cat = get_attn_bias_and_cat(x_packed)
                x_cat = blk(x_cat, attn_bias=attn_bias)
                x_packed = attn_bias.split(x_cat)

        outputs = []
        for x, masks in zip(x_packed, masks_list):
            x_norm = self.norm(x)
            outputs.append({
                "clstoken": x_norm[:, 0],
                "regtokens": x_norm[:, 1:self.numregisters+1],
                "patchtokens": x_norm[:, self.numregisters+1:],
                "masks": masks
            })

        return outputs

    def _prepare_packed(self, x, masks_list):
        """Run patch embedding + token preparation for each crop in the list."""
        x_packed = []
        for img, masks in zip(x, masks_list):
            if masks is not None:
                x_prep = self.prepare_tokens_with_masks(img, masks)
            else:
                x_prep = self.prepare_tokens(img)
            x_packed.append(x_prep)
        return x_packed

    def forward_features_list_per_step(self, x, masks_list):
        """
        Looped-mode forward returning per-recursion-step outputs.

        Only valid when `looped_T_max > 0`. All crops traverse all T_max
        recursion steps in lockstep; the return is a list of length T_max
        whose t-th entry is exactly the same shape (list of per-crop dicts)
        that `forward_features_list` would return.

        Args:
            x: list of image tensors (one per crop type).
            masks_list: list of mask tensors (can contain None).

        Returns:
            List of length T_max; each element is a list of per-crop output
            dicts with keys 'clstoken', 'regtokens', 'patchtokens', 'masks'.
        """
        if self.looped_T_max <= 0:
            raise RuntimeError(
                "forward_features_list_per_step requires looped_T_max > 0. "
                "The standard non-looped backbone produces only one stack of "
                "outputs."
            )

        x_packed = self._prepare_packed(x, masks_list)
        attn_bias, x_cat = get_attn_bias_and_cat(x_packed)

        z_per_step = self.shared_stack(
            x_cat,
            attn_bias=attn_bias,
            grad_checkpoint=self.grad_checkpointing,
            return_per_step=True,
        )

        outputs_per_step = []
        for z_t in z_per_step:
            x_packed_t = attn_bias.split(z_t)
            outputs_t = []
            for x_crop, masks in zip(x_packed_t, masks_list):
                x_norm = self.norm(x_crop)
                outputs_t.append({
                    "clstoken": x_norm[:, 0],
                    "regtokens": x_norm[:, 1:self.numregisters+1],
                    "patchtokens": x_norm[:, self.numregisters+1:],
                    "masks": masks,
                })
            outputs_per_step.append(outputs_t)

        return outputs_per_step

    def forward(
        self,
        x,
        token_masks=None,
        return_dict: bool = False,
        halt_head: Optional[nn.Module] = None,
        epsilon: float = 0.01,
    ):
        """
        Forward with automatic detection of single vs multi-crop input.

        Args:
            x: Either single tensor [B, C, H, W] or list of tensors
            token_masks: Either None, single mask, or list of masks
            return_dict: Single-image path only. False (default) returns the
                pre-self.norm CLS token of shape [B, D]. True returns a dict
                with prenorm and postnorm versions of CLS, patches, and
                register tokens (and `halt_step` in looped mode). Ignored on
                the list-input path.
            halt_head: Looped, single-image path only. When provided (an
                `nn.Module` mapping postnorm CLS [B, D] -> halt prob [B]),
                drives PonderNet-style early exit per sample. When None,
                runs the full T_max recursion (legacy behavior).
            epsilon: PonderNet halt threshold. A sample halts when its
                cumulative not-halted mass falls below this value.

        Returns:
            - If list: List of dicts (unchanged)
            - If single, return_dict=False: Tensor [B, D] (pre-norm CLS)
            - If single, return_dict=True: Dict with prenorm/postnorm tokens
              (plus 'halt_step' [B] in looped mode)
        """
        if isinstance(x, list):
            if token_masks is None:
                token_masks = [None] * len(x)
            elif not isinstance(token_masks, list):
                token_masks = [token_masks] + [None] * (len(x) - 1)

            return self.forward_features_list(x, token_masks)

        else:
            if token_masks is not None:
                x = self.prepare_tokens_with_masks(x, token_masks)
            else:
                x = self.prepare_tokens(x)

            x = self.patch_drop(x)
            x = self.norm_pre(x)

            # After prepare_tokens / patch_drop / norm_pre, `x` is z_0.
            if self.looped_T_max > 0:
                z0 = x
                B = z0.shape[0]
                T_max = self.looped_T_max
                halt_step = torch.full(
                    (B,), T_max, dtype=torch.long, device=z0.device
                )

                if halt_head is None:
                    # No halt head: run all T_max steps (legacy behavior).
                    z = self.shared_stack(
                        z0,
                        attn_bias=None,
                        grad_checkpoint=self.grad_checkpointing,
                        return_per_step=False,
                    )
                else:
                    not_halted_mass = torch.ones(
                        B, device=z0.device, dtype=z0.dtype
                    )
                    halted_mask = torch.zeros(
                        B, dtype=torch.bool, device=z0.device
                    )
                    z_out = None
                    z_prev = z0
                    z_t = z0

                    for t in range(T_max):
                        tau_t = self.shared_stack.tau[t].to(z_prev.dtype)
                        z_t = self.shared_stack.step(
                            z_prev, z0, tau_t, attn_bias=None
                        )

                        # Side tap: self.norm only to feed halt_head with its
                        # training-time input distribution. The residual
                        # stream returned to MIL consumers stays pre-self.norm.
                        cls_for_halt = self.norm(z_t)[:, 0]
                        h_t = halt_head(cls_for_halt)

                        not_halted_mass = not_halted_mass * (1.0 - h_t)
                        new_halts = (not_halted_mass < epsilon) & ~halted_mask
                        if new_halts.any():
                            if z_out is None:
                                z_out = z_t.clone()
                            else:
                                z_out[new_halts] = z_t[new_halts]
                            halt_step[new_halts] = t + 1
                            halted_mask = halted_mask | new_halts

                        if halted_mask.all():
                            break
                        z_prev = z_t

                    # Samples that never crossed threshold get the final z_t.
                    if not halted_mask.all():
                        if z_out is None:
                            z_out = z_t.clone()
                        else:
                            z_out[~halted_mask] = z_t[~halted_mask]

                    z = z_out
                x = z  # common tail consumes `x` regardless of path

            elif self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                for blk in self.blocks:
                    x = blk(x, attn_bias=None)

            # Common tail. `x` is the residual stream (pre-self.norm).
            if not return_dict:
                return x[:, 0]

            x_norm = self.norm(x)
            out = {
                'clstoken_prenorm':     x[:, 0],
                'clstoken_postnorm':    x_norm[:, 0],
                'patchtokens_prenorm':  x[:, self.numregisters + 1:],
                'patchtokens_postnorm': x_norm[:, self.numregisters + 1:],
                'regtokens_prenorm':    x[:, 1:self.numregisters + 1],
                'regtokens_postnorm':   x_norm[:, 1:self.numregisters + 1],
                'masks':                token_masks,
            }
            if self.looped_T_max > 0:
                out['halt_step'] = halt_step
            return out



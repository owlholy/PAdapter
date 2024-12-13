from .attn import *

from operator import mul
from functools import reduce

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, config=None, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.norm1 = norm_layer(dim)
        if self.config.supervised_mode == 1:
            self.attn = Attention_supervised__video(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
                config=config, layer_id=layer_id,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # rewrite FFN here
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)
        if config.ffn_adapt:
            self.adaptmlp = Adapter(self.config, dropout=drop, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option
                                    )
        if config.padapt_on == 1:
            self.padapter = PAdapter(self.config, dropout=0., adapter_layernorm_option=None)

    def forward(self, x):
        if self.config.padapt_on == 1 and self.config.padapt_local == 'p+b':
            x_pb = self.padapter(x, add_residual=False)  # block

        if self.config.padapt_on == 1 and self.config.padapt_local == 'p_a_m':
            x = self.padapter(x)  #p_a_m

        if self.config.padapt_on == 1 and self.config.padapt_local == 'p+a_m':
            x_pa = self.padapter(x, add_residual=False)  # p+a_m
        residual = x
        x = self.norm1(x)
        x = residual + self.drop_path(self.attn(x))
        if self.config.padapt_on == 1 and self.config.padapt_local == 'p+a_m':
            x = x + x_pa  # p+a_m

        if self.config.padapt_on == 1 and self.config.padapt_local == 'a_p_m':
            x = self.padapter(x)  # a_p_m

        if self.config.padapt_on == 1 and self.config.padapt_local == 'a_p+m':
            x_pm = self.padapter(x, add_residual=False)  # a_p+m
        if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
            adapt_x = self.adaptmlp(x, add_residual=False)

        residual = x
        x = self.act(self.fc1(self.norm2(x)))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                x = self.adaptmlp(x)
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        if self.config.padapt_on == 1 and self.config.padapt_local == 'a_p+m':
            x = x + x_pm  # a_p+m

        if self.config.padapt_on == 1 and self.config.padapt_local == 'a_m_p':
            x = self.padapter(x)  # a_m_p

        if self.config.padapt_on == 1 and self.config.padapt_local == 'p+b':
            x = x + x_pb  # block

        return x

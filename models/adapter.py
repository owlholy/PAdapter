# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=768,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class PAdapter(nn.Module):
    def __init__(self,
                 config=None,
                 dropout=0.0,
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model
        self.down_size = config.padapt_bottleneck
        self.L_size = config.padapt_L_size
        self.learn_mode = config.padapt_learn_mode
        self.cat_mode = config.padapt_cat_mode
        self.adapter_scalar = config.padapt_scalar
        self.init_option = config.ffn_adapter_init_option

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if self.adapter_scalar == "learnable_scalar" or self.adapter_scalar == "0.":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(self.adapter_scalar)

        self.non_linear_func = nn.ReLU()
        # self.non_linear_func = nn.GELU()

        if self.cat_mode == "x_p":
            self.down_proj = nn.Linear(self.n_embd, self.down_size)
            self.up_proj = nn.Linear(self.down_size * 2, self.n_embd)
            self.prompt = nn.Parameter(torch.empty(1, self.L_size, self.down_size))
            torch.nn.init.xavier_uniform_(self.prompt.data)

        elif self.cat_mode == "p_x_p":
            self.down_proj = nn.Linear(self.n_embd, self.down_size)
            self.up_proj = nn.Linear(self.down_size * 3, self.n_embd)
            self.prompt1 = nn.Parameter(torch.empty(1, self.L_size, self.down_size))
            torch.nn.init.xavier_uniform_(self.prompt1.data)
            self.prompt2 = nn.Parameter(torch.empty(1, self.L_size, self.down_size))
            torch.nn.init.xavier_uniform_(self.prompt2.data)

        elif self.cat_mode == "p_2x_p":
            self.down_proj = nn.Linear(self.n_embd, self.down_size * 2)
            self.up_proj = nn.Linear(self.down_size * 4, self.n_embd)
            self.prompt1 = nn.Parameter(torch.empty(1, self.L_size, (self.down_size)))
            torch.nn.init.xavier_uniform_(self.prompt1.data)
            self.prompt2 = nn.Parameter(torch.empty(1, self.L_size, (self.down_size)))
            torch.nn.init.xavier_uniform_(self.prompt2.data)
        else:
            raise ValueError(self.cat_mode)

        self.dropout = dropout
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        B, L, C = x.size()

        if self.cat_mode == "x_p":
            if self.learn_mode == "cat":
                prompt = self.prompt.expand(B, L, self.down_size)  # cat
            elif self.learn_mode == "add":
                prompt_clone = self.prompt.expand(B, L, self.down_size)  # add
                prompt = prompt_clone + down  # add
            else:
                raise ValueError(self.learn_mode)
            down = torch.cat((down, prompt), dim=2)

        if self.cat_mode == "p_x_p":
            if self.learn_mode == "cat":
                prompt1 = self.prompt1.expand(B, L, self.down_size)  # cat
                prompt2 = self.prompt2.expand(B, L, self.down_size)  # cat
            elif self.learn_mode == "add":
                prompt1_clone = self.prompt1.expand(B, L, self.down_size)  # add
                prompt1 = prompt1_clone + down  # add
                prompt2_clone = self.prompt2.expand(B, L, self.down_size)  # add
                prompt2 = prompt2_clone + down  # add
            else:
                raise ValueError(self.learn_mode)
            down = torch.cat((prompt1, down, prompt2), dim=2)

        if self.cat_mode == "p_2x_p":
            if self.learn_mode == "cat":
                prompt1 = self.prompt1.expand(B, L, int(self.down_size/2))  # cat
                prompt2 = self.prompt2.expand(B, L, int(self.down_size/2))  # cat
            elif self.learn_mode == "add":
                prompt1_clone = self.prompt1.expand(B, L, self.down_size)  # add
                prompt1 = prompt1_clone + down[:, :, :self.down_size]  # add
                prompt2_clone = self.prompt2.expand(B, L, self.down_size)  # add
                prompt2 = prompt2_clone + down[:, :, self.down_size:]  # add
            else:
                raise ValueError(self.learn_mode)
            down = torch.cat((prompt1, down, prompt2), dim=2)

        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

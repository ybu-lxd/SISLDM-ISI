import torch
import torch.nn as nn
import numpy as np

from ldm.modules.diffusionmodules.openaimodel import  AttentionBlock, Upsample, Downsample, TimestepBlock
from ldm.modules.diffusionmodules.util import normalization, conv_nd, linear, zero_module, checkpoint
import einops
from timm.models.vision_transformer import Mlp, PatchEmbed, Attention


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(val):
    return val is not None


def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h
class Mydownsampleto32(nn.Module):
    def __init__(self,
                 in_channels=4,
                 model_channels=64,
                 out_channels=4,
                 num_res_blocks=2,
                 attention_resolutions=[False,False,True,True],
                 dropout=0,
                 channel_mult=[1,2,2,4],
                 init_kernel_sizes = 7
                 ):
        super().__init__()

        assert len(attention_resolutions) == len(channel_mult),"确保长度相等"
        self.input_blocks= nn.Conv2d(in_channels,out_channels=model_channels,kernel_size=init_kernel_sizes,padding= (init_kernel_sizes - 1) // 2)
        channel = model_channels
        dims = [channel,*map(lambda m: channel*m,channel_mult)]
        print(dims)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(in_out)
        num_resnet_blocks = cast_tuple(num_res_blocks, len(channel_mult))
        print(num_resnet_blocks)
        layer_params = [num_resnet_blocks,attention_resolutions]
        # print(*zip(in_out, *layer_params))
        self.downs = nn.ModuleList([])
        for step ,((dim_in,dim_out),layer_num_resnet_blocks,layer_attn) in enumerate(zip(in_out, *layer_params)):
            # outdown = nn.Identity()
            if step !=len(attention_resolutions)-1 :
                outdown = ResBlock(channels=dim_in,out_channels=dim_out,dropout=0.0,dims=2,down=True)

            else:
                outdown = ResBlock(channels=dim_in,out_channels=dim_out,dropout=0.0,dims=2,down=False)
            self.downs.append(nn.ModuleList(
              [

                nn.ModuleList([ResBlock(channels=dim_in,out_channels=dim_in,dropout=0.0,dims=2) for _ in range(layer_num_resnet_blocks)]),
                AttentionBlock(
                      dim_in,
                      use_checkpoint=False,
                      num_heads=8,
                      num_head_channels=-1,
                      use_new_attention_order=False,
                  ) if layer_attn else nn.Identity(),
                  outdown
              ]



            ))


        self.out = nn.Sequential(ResBlock(channel_mult[-1]*model_channels,out_channels=model_channels,dropout=0.0),
                                 normalization(model_channels),
                                 nn.SiLU(),
                                 zero_module(conv_nd(2,model_channels,out_channels,3,padding=1)))




    def forward(self,x):
     
        x = self.input_blocks(x)

        for resnet_blocks,att_block,down in self.downs:
            for resnet_block in resnet_blocks:
                x = resnet_block(x)
            x = att_block(x)
            x = down(x)
        # print(x.shape)
        x = self.out(x)

        return x





def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class Transformer(nn.Module):
    def __init__(self, hidden_size=384, num_heads=8,depth=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([])
        if hidden_size<=384:
            hidden_features = hidden_size*4
        else:
            hidden_features = hidden_size
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [
                    Attention(hidden_size,num_heads),
                    Mlp(in_features=hidden_size,hidden_features=hidden_features,out_features =hidden_size )
                ]


            ))





    def forward(self,x):

        for attention,mlp in self.layers:
            x = attention(x)+x
            x = mlp(x)+x



        return self.norm(x)





class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x):
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

class Transformers_Conditions_merge(nn.Module):
    def __init__(self,
                 input_size=32,
                 in_chans=7,
                 embed_dim=384,
                 patch_size=8,
                 layers = 2
                 ):
        super().__init__()
        self.input_size = input_size
        self.in_chans  = in_chans
        self.out_channels = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_chans, embed_dim, bias=True)
        num_patches = self.x_embedder.num_patches
        hidden_size = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size),requires_grad=False)
        # self.trm = Transformer()
        self.trm = nn.ModuleList([
        ])

        for seps,i in enumerate(range(layers)):
            self.trm.append(nn.ModuleList([
                Transformer(),
                # Transformer(hidden_size=64)
                nn.Identity()
            ]))






        self.trmt = Transformer(hidden_size=64)
        self.final_layer = FinalLayer(hidden_size=384,patch_size=patch_size,out_channels=in_chans)
        self.initialize_weights()
    def unpatchify(self,x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self,x):
        x = self.x_embedder(x)+self.pos_embed
        for trms,trmt in self.trm:
            x = trms(x)
            # x = einops.rearrange(x,"(b t) c l -> b l (c t)",t = 4)
            # x= trmt(x)
            # x = einops.rearrange(x,"b l (c t ) ->(b t) c l ",t = 4)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x





















# device = "cuda"
# net = myditr().to(device)
# # print(count_params(net,True))
# x = torch.rand(size=(8,7,32,32)).to(device)
# print(net(x).shape)





# def calculate_model_size(model):
#     num_params = sum(p.numel() for p in model.parameters())
#     model_size_bytes = num_params * 4  # 每个参数通常为4个字节（32位浮点数）
#     return model_size_bytes
# net = Mydownsampleto32()
# print(net)
# print(calculate_model_size(net)/(1024*1024))
# print(net(torch.randn(size=(4, 4, 256, 256))).shape)











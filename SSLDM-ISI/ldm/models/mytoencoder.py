import torch
# import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
import yaml
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config

class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 colorize_nlabels=None
                 ):
        super().__init__()
        embed_dim = ddconfig['model']['params']['embed_dim']
        z_channels  = ddconfig['model']['params']['ddconfig']['z_channels']
        self.encoder = Encoder(**ddconfig['model']['params']['ddconfig'])
        self.decoder = Decoder(**ddconfig['model']['params']['ddconfig'])
        self.quant_conv = torch.nn.Conv2d(2 *z_channels , 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    def forward(self, input, encoder=True,sample_posterior=True):

        if encoder:
            posterior = self.encode(input)
            return  posterior.sample()


        else:
            dec = self.decode(input)

            return dec

    def get_last_layer(self):
        return self.decoder.conv_out.weight
































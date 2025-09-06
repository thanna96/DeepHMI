# =========================================================================
#   (c) Copyright 2025
#   All rights reserved
#   Programs written by Chunhui Xu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

import os
import torch
import torch.nn as nn
import numpy as np
import einops
from astropy.io import fits

torch.set_default_dtype(torch.float32)

num_of_gpus = torch.cuda.device_count()

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")


class DilationLayer(nn.Module):
	def __init__(self, ker_in, ker_out, dek):
		super(DilationLayer, self).__init__()
		self.ker_in = ker_in		
		self.ker_out = ker_out//4
		self.dek = dek
		self.dilayer = nn.ModuleList([nn.Conv2d(self.ker_in, self.ker_out, 3, 1, 2**i, 2**i) for i in range(4)])
		self.act = nn.LeakyReLU(0.2)
	def forward(self, x):
		out_0 = self.dilayer[0](x)
		out_1 = self.dilayer[1](x)
		out_2 = self.dilayer[2](x)
		out_3 = self.dilayer[3](x)
		out = torch.cat((out_0, out_1, out_2, out_3),dim=1)
		return self.act(out)

class FourTimes(nn.Module):
	def __init__(self, ker_in, ker_out, dek):
		super(FourTimes, self).__init__()
		self.dek = dek
		self.ker_in = ker_in
		self.ker_out = ker_out//4
		self.one = DilationLayer(self.ker_in, self.ker_out, self.dek)
		self.two = DilationLayer(self.ker_out, self.ker_out, self.dek)
		self.four = DilationLayer(self.ker_out*2, self.ker_out*2, self.dek)
	def forward(self, x):
		out_one = self.one(x)
		out_two = self.two(out_one)
		out_two = torch.cat((out_one, out_two), dim=1)
		out_four = self.four(out_two)
		out_four = torch.cat((out_two, out_four), dim=1)
		return out_four
	
class Conditional(nn.Module):
  def __init__(self, ker_in, ker_out, dek):
    super(Conditional, self).__init__()
    self.ker_in = ker_in
    self.ker_out = ker_out
    self.dek = dek
    self.facter = self.ker_out // self.dek
    self.conv = nn.Conv2d(ker_in, ker_out*2, 3, self.facter, 1)
    self.act = nn.Sigmoid()
  def forward(self, x):
    return self.act(self.conv(x)).chunk(2, dim = 1)


class Beta_em(nn.Module):
  def __init__(self, cha, dek, theta = 10000):
    super(Beta_em, self).__init__()
    self.cha = cha
    self.dek = dek
    self.theta = theta
    self.mlp = nn.Sequential(
            nn.Linear(self.cha, self.cha * 4),
            nn.LeakyReLU(0.2)
        )
    self.lin_1 = nn.Linear(self.cha * 4, self.cha, bias=False)
    self.lin_2 = nn.Linear(self.cha * 4, self.cha, bias=False)

  def forward(self, x):
    device = x.device
    x *= 50
    half_dim = self.cha // 2
    emb = (torch.log(torch.tensor(self.theta)) / (half_dim - 1)).to(device)
    emb = torch.exp(torch.arange(half_dim).to(device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    emb = self.mlp(emb)
    return self.lin_1(emb), self.lin_2(emb)
  
class RMSNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

  def forward(self, x):
    return nn.functional.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class Att_block(nn.Module):
  def __init__(self, cha, dek=32, mul_head=4, sea=5, drop=0.2):
    super().__init__()
    self.cha = cha
    self.dek = dek
    self.mul_head = mul_head
    self.sea = sea
    self.norm = RMSNorm(self.cha)
    self.to_qkv = nn.Conv2d(self.cha, int(self.cha*3), 1, bias = False)
    self.to_out = nn.Conv2d(self.cha, self.cha, 1)
    self.attn_dropout = nn.Dropout(drop)

  def forward(self, x):
    if self.sea > 1:
      x = nn.functional.interpolate(x, scale_factor=1/self.sea, mode="bilinear", align_corners=False)
    b, c, h, w = x.shape

    x = self.norm(x)

    qkv = self.to_qkv(x).chunk(3, dim = 1)
    q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.mul_head), qkv)

    sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k)

    attn = sim.softmax(dim = -1)
    attn = self.attn_dropout(attn)

    out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)
    out = einops.rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

    x = self.to_out(out)

    if self.sea > 1:
      x = nn.functional.interpolate(x, scale_factor=self.sea, mode="bilinear", align_corners=False)
    return x
  
class Emb_layer(nn.Module):
  def __init__(self, cha, dek=32):
    super().__init__()
    self.cha = cha
    self.dek = dek
    self.condition = Conditional(self.dek*4, self.cha, self.dek)
    self.beta_em = Beta_em(self.cha, self.dek)
    self.act = nn.LeakyReLU(0.2)
    self.norm_1 = RMSNorm(self.cha)
    self.norm_2 = RMSNorm(self.cha)
    self.conv_1 = nn.Conv2d(self.cha, self.cha, 1, 1, 0)
    self.conv_2 = nn.Conv2d(self.cha, self.cha, 1, 1, 0)

  def forward(self, flow, inp, beta_inp):
    beta, gamma = self.condition(inp)
    beta_b, gamma_b = self.beta_em(beta_inp)
    beta_b = beta_b.view(beta_b.shape[0], beta_b.shape[-1])
    gamma_b = gamma_b.view(gamma_b.shape[0], gamma_b.shape[-1])

    flow = self.conv_1(flow)
    flow = self.act(flow)
    flow = flow*(beta+1)+gamma
    flow = self.norm_1(flow)
    flow = self.conv_2(flow)
    flow = self.act(flow)
    flow = flow*(beta_b+1)[:, :, None, None] + gamma_b[:, :, None, None]
    flow = self.norm_2(flow)
    return flow
  
class Res_layer(nn.Module):
  def __init__(self, cha, dek=32, att_sca = 5, att_use = False):
    super().__init__()
    self.cha = cha
    self.dek = dek
    self.att_sca = att_sca
    self.att_use = att_use
    self.norm = RMSNorm(self.cha)
    self.act = nn.LeakyReLU(0.2)
    self.conv_1 = nn.Conv2d(self.cha, self.cha, 3, 1, 1)
    self.conv_2 = nn.Conv2d(self.cha, self.cha, 3, 1, 1)
    if self.att_use:
      self.att = Att_block(self.cha, self.dek, sea=self.att_sca)

  def forward(self, x):
    flow = self.conv_1(x)
    flow = self.act(flow)
    flow = self.conv_2(flow)
    flow = self.act(flow)
    flow = self.norm(flow)
    if self.att_use:
      out_att = self.att(flow+x)
      return flow+out_att
    else:
      return flow
    
class Res_block(nn.Module):
  def __init__(self, cha, dek=32, att_sca = 5, att_use = False):
    super().__init__()
    self.cha = cha
    self.dek = dek
    self.att_sca = att_sca
    self.att_use = att_use
    self.res = Res_layer(self.cha, self.dek)
    self.emb = Emb_layer(self.cha, self.dek)
    if self.att_use:
      self.att = Att_block(self.cha, self.dek, sea=self.att_sca)

  def forward(self, x, inp, beta_inp):
    out_res = self.res(x)
    out_emb = self.emb(x, inp, beta_inp)
    flow = out_res+out_emb+x
    if self.att_use:
      out_att = self.att(flow)
      return flow+out_att, inp, beta_inp
    else:
      return flow, inp, beta_inp
    
class Unet_block(nn.Module):
  def __init__(self, def_ker=8, res_number=[2, 2, 2, 2, 2, 2], att_sca = 5, att_out_use = False, att_out = 1):
    super().__init__()
    self.def_ker = def_ker
    self.res_number = res_number
    self.att_sca = att_sca
    self.att_out_use = att_out_use
    self.att_out = att_out
    self.act = nn.LeakyReLU(0.2)

    self.cond_inp = FourTimes(1, self.def_ker*4, self.def_ker)

    self.start = nn.Conv2d(1, self.def_ker, 7, 1, 15, 5)
    
    self.res_blocks_1 = nn.ModuleList([Res_block(self.def_ker, self.def_ker, self.att_sca*2, att_use=True) for i in range(self.res_number[0])])
    self.res_blocks_2 = nn.ModuleList([Res_block(self.def_ker*2, self.def_ker, self.att_sca*2, att_use=True) for i in range(self.res_number[1])])
    self.res_blocks_3 = nn.ModuleList([Res_block(self.def_ker*4, self.def_ker, self.att_sca, att_use=True) for i in range(self.res_number[2])])
    self.res_blocks_4 = nn.ModuleList([Res_block(self.def_ker*2, self.def_ker, self.att_sca*2, att_use=True) for i in range(self.res_number[3])])
    self.res_blocks_5 = nn.ModuleList([Res_block(self.def_ker, self.def_ker, self.att_sca*2, att_use=True) for i in range(self.res_number[4])])
    self.resize_1 = nn.Conv2d(self.def_ker*4, self.def_ker*2, 1, 1, 0)
    self.resize_2 = nn.Conv2d(self.def_ker*2, self.def_ker, 1, 1, 0)
    self.down_1 = nn.Conv2d(self.def_ker, self.def_ker*2, 2, 2, 0)
    self.down_2 = nn.Conv2d(self.def_ker*2, self.def_ker*4, 2, 2, 0)
    self.up_1 = nn.ConvTranspose2d(self.def_ker*4, self.def_ker*2, 2, 2, 0)
    self.up_2 = nn.ConvTranspose2d(self.def_ker*2, self.def_ker, 2, 2, 0)
    self.end = nn.Conv2d(self.def_ker, 1, 1, 1, 0)

  def forward(self, x, inp, beta_inp):
    x = self.start(x)
    x = self.act(x)

    inp = self.cond_inp(inp)

    for i in range(self.res_number[0]):
      x, inp, beta_inp = self.res_blocks_1[i](x, inp, beta_inp)
    r_1, _, __ = x, inp, beta_inp
    d_1 = self.down_1(r_1)
    d_1 = self.act(d_1)
    for i in range(self.res_number[1]):
      d_1, inp, beta_inp = self.res_blocks_2[i](d_1, inp, beta_inp)
    r_2, _, __ = d_1, inp, beta_inp
    d_2 = self.down_2(r_2)
    d_2 = self.act(d_2)
    for i in range(self.res_number[2]):
      d_2, inp, beta_inp = self.res_blocks_3[i](d_2, inp, beta_inp)
    r_3, _, __ = d_2, inp, beta_inp
    u_1 = self.up_1(r_3)
    u_1 = self.act(u_1)
    u_1 = self.resize_1(torch.cat((u_1, r_2), dim=1))
    for i in range(self.res_number[3]):
      u_1, inp, beta_inp = self.res_blocks_4[i](u_1, inp, beta_inp)
    r_4, _, __ = u_1, inp, beta_inp
    u_2 = self.up_2(r_4)
    u_2 = self.act(u_2)
    u_2 = self.resize_2(torch.cat((u_2, r_1), dim=1))
    for i in range(self.res_number[4]):
      u_2, inp, beta_inp = self.res_blocks_5[i](u_2, inp, beta_inp)
    r_5, _, __ = u_2, inp, beta_inp
    return self.end(r_5)

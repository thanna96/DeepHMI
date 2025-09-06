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
from model import Unet_block

torch.set_default_dtype(torch.float32)

num_of_gpus = torch.cuda.device_count()

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

def create_dataset(x_hmi_train, y_gst_train, beta, alpha_prod):
  x_gst_noi_data = []
  y_gst_noi_data = []
  x_hmi_inp_data = []

  for _ in range(len(y_gst_train)):
    x0 = y_gst_train[_]
    x_sample = [x0]

    hmi0 = x_hmi_train[_]

    for i in range(beta.shape[0]):
      eps = np.random.normal(0, 1, x0.shape)
      x_i = np.sqrt(alpha_prod[i]) * x0 + np.sqrt(1 - alpha_prod[i]) * eps
      x_sample.append(x_i)

      x_hmi_inp_data.append(hmi0)

    x_gst_noi_data += x_sample[1:]
    y_gst_noi_data += x_sample[:-1]

  x_gst_noi_data = np.array(x_gst_noi_data, dtype=np.float32)
  y_gst_noi_data = np.array(y_gst_noi_data, dtype=np.float32)
  x_hmi_inp_data = np.array(x_hmi_inp_data, dtype=np.float32)
  return x_gst_noi_data, y_gst_noi_data, x_hmi_inp_data

def create_dataset_noi(x_hmi_train, y_gst_train, beta, alpha_prod):
  x_gst_noi_data = []
  y_gst_noi_data = []
  x_hmi_inp_data = []

  for _ in range(len(y_gst_train)):
    x0 = y_gst_train[_]
    hmi0 = x_hmi_train[_]

    for i in range(beta.shape[0]):
      eps = np.random.normal(0, 1, x0.shape)
      x_i = np.sqrt(alpha_prod[i]) * x0 + np.sqrt(1 - alpha_prod[i]) * eps
      x_gst_noi_data.append(x_i)
      y_gst_noi_data.append(eps)

      x_hmi_inp_data.append(hmi0)

  x_gst_noi_data = np.array(x_gst_noi_data, dtype=np.float32)
  y_gst_noi_data = np.array(y_gst_noi_data, dtype=np.float32)
  x_hmi_inp_data = np.array(x_hmi_inp_data, dtype=np.float32)
  return x_gst_noi_data, y_gst_noi_data, x_hmi_inp_data

def data_train_create(img, img2):
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:
                out = img[:, :, :360, :360]
                out2 = img2[:, :, :360, :360]
            else:
                out = np.concatenate([out, img[:, :, i*180:i*180+360, j*180:j*180+360]], 0)
                out2 = np.concatenate([out2, img2[:, :, i*180:i*180+360, j*180:j*180+360]], 0)
    count = np.random.randint(4)
    if count>0:
        out = np.rot90(out, k=count, axes=(-2, -1))
        out2 = np.rot90(out2, k=count, axes=(-2, -1))
    return out, out2

input_path = "dataset_train/"
output_path = "model_out/"

beta = np.linspace(0.001, 0.02, 1000)
alpha = 1 - beta
alpha_prod = np.cumprod(alpha)
alpha_prod = np.array(alpha_prod, dtype=np.float32)

input_x_train_row = []
input_y_train_row = []
for name in os.listdir(input_path):
  fits_file = fits.open(input_path + name)
  fits_data = fits_file[0].data
  x_data = np.concatenate((np.reshape(fits_data[1], (1, 720, 720)), np.reshape(fits_data[3], (1, 720, 720)),np.reshape(fits_data[5], (1, 720, 720))), 0)
  y_data = np.concatenate((np.reshape(fits_data[0], (1, 720, 720)), np.reshape(fits_data[2], (1, 720, 720)),np.reshape(fits_data[4], (1, 720, 720))), 0)
  input_x_train_row.append((name, x_data))
  input_y_train_row.append((name, y_data))

input_x_train_row = sorted(input_x_train_row,key = lambda x:x[0])
input_y_train_row = sorted(input_y_train_row,key = lambda x:x[0])

x_hmi_train_data = []
for i in range(len(input_x_train_row)):
	x_1 = input_x_train_row[i][1][0, :, :]
	x_3 = np.clip(input_x_train_row[i][1][2, :, :], 0, 180)*np.pi/180
	x_z = x_1 * np.cos(x_3)
	x_t = x_1 * np.sin(x_3)
	x_hmi_train_data.append(np.concatenate((np.reshape(x_z/1500, (1, 720, 720)),
											np.reshape(x_t/750-1, (1, 720, 720))), axis = 0))
x_hmi_train_data = np.array(x_hmi_train_data, dtype=np.float32)

y_gst_train_data = []
for i in range(len(input_y_train_row)):
	y_1 = input_y_train_row[i][1][0, :, :]
	y_3 = np.pi - np.clip(input_y_train_row[i][1][2, :, :], 0, np.pi)
	y_z = y_1 * np.cos(y_3)
	y_t = y_1 * np.sin(y_3)
	y_gst_train_data.append(np.concatenate((np.reshape(y_z/1500, (1, 720, 720)),
											np.reshape(y_t/750-1, (1, 720, 720))), axis = 0))
y_gst_train_data = np.array(y_gst_train_data, dtype=np.float32)

model = Unet_block(def_ker=32, res_number=[4, 4, 4, 4, 4, 2], att_sca = 3)
model = nn.DataParallel(model, device_ids=[0])
model.to(device)

model.train()
epo = 1000
batch_size = 16
loss_fn = nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=0.0001)

data_batch = 5

for _ in range(epo):
  
  shu_data = np.arange(x_hmi_train_data.shape[0])
  np.random.shuffle(shu_data)
  x_hmi_train_data = x_hmi_train_data[shu_data]
  y_gst_train_data = y_gst_train_data[shu_data]

  for db in range(x_hmi_train_data.shape[0]//data_batch):
    running_loss = 0.0

    x_hmi_train, y_gst_train = data_train_create(x_hmi_train_data[db*data_batch:(db+1)*data_batch], y_gst_train_data[db*data_batch:(db+1)*data_batch])

    p = np.random.randint(2)
    if p >0:
      ind = []
      for i in range(x_hmi_train.shape[0]):
        if np.mean(np.abs(x_hmi_train[i, 1]*750 + 750)) < 100:
          ind.append(i)
      x_hmi_train = np.delete(x_hmi_train, ind, 0)
      y_gst_train = np.delete(y_gst_train, ind, 0)

    x_gst_noi_data, y_gst_noi_data, x_hmi_inp_data, beta_train = create_dataset_noi(x_hmi_train[:, 1:2, :, :], y_gst_train[:, 1:2, :, :], beta, alpha_prod)

    shu_list = np.arange(x_gst_noi_data.shape[0])
    np.random.shuffle(shu_list)

    y_gst_noi_data = y_gst_noi_data[shu_list]
    x_gst_noi_data = x_gst_noi_data[shu_list]
    x_hmi_inp_data = x_hmi_inp_data[shu_list]
    beta_train = beta_train[shu_list]

    train_len = int(y_gst_noi_data.shape[0]/batch_size)
    for i in range(train_len):
      opti.zero_grad()
      x_1 = x_gst_noi_data[i*batch_size:(i+1)*batch_size]
      x_2 = x_hmi_inp_data[i*batch_size:(i+1)*batch_size]
      x_3 = beta_train[i*batch_size:(i+1)*batch_size]
      x_1 = torch.tensor(x_1, dtype=torch.float32).to(device)
      x_2 = torch.tensor(x_2, dtype=torch.float32).to(device)
      x_3 = torch.tensor(x_3, dtype=torch.float32).to(device)
      target = y_gst_noi_data[i*batch_size:(i+1)*batch_size]
      target = torch.tensor(target).to(device)

      y = model(x_1, x_2, x_3)
      l = loss_fn(y, target)
      l.backward()
      opti.step()

      running_loss += l.item()
    if train_len > 0:
      print("epoch: ", _+1, " loss: ", running_loss/train_len, flush=True)
      torch.save(model.state_dict(), output_path + "model_bt_new")
      

model = Unet_block(def_ker=32, res_number=[4, 4, 4, 4, 4, 2], att_sca = 3)
model = nn.DataParallel(model, device_ids=[0])
model.to(device)

model.train()
epo = 1000
batch_size = 16
loss_fn = nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=0.0001)

data_batch = 5

for _ in range(epo):
  
  shu_data = np.arange(x_hmi_train_data.shape[0])
  np.random.shuffle(shu_data)
  x_hmi_train_data = x_hmi_train_data[shu_data]
  y_gst_train_data = y_gst_train_data[shu_data]

  for db in range(x_hmi_train_data.shape[0]//data_batch):
    running_loss = 0.0

    x_hmi_train, y_gst_train = data_train_create(x_hmi_train_data[db*data_batch:(db+1)*data_batch], y_gst_train_data[db*data_batch:(db+1)*data_batch])

    p = np.random.randint(2)
    if p >0:
      ind = []
      for i in range(x_hmi_train.shape[0]):
        if np.mean(np.abs(x_hmi_train[i, 1]*750 + 750)) < 100:
          ind.append(i)
      x_hmi_train = np.delete(x_hmi_train, ind, 0)
      y_gst_train = np.delete(y_gst_train, ind, 0)

    x_gst_noi_data, y_gst_noi_data, x_hmi_inp_data, beta_train = create_dataset_noi(x_hmi_train[:, :1, :, :], y_gst_train[:, :1, :, :], beta, alpha_prod)

    shu_list = np.arange(x_gst_noi_data.shape[0])
    np.random.shuffle(shu_list)

    y_gst_noi_data = y_gst_noi_data[shu_list]
    x_gst_noi_data = x_gst_noi_data[shu_list]
    x_hmi_inp_data = x_hmi_inp_data[shu_list]
    beta_train = beta_train[shu_list]

    train_len = int(y_gst_noi_data.shape[0]/batch_size)
    for i in range(train_len):
      opti.zero_grad()
      x_1 = x_gst_noi_data[i*batch_size:(i+1)*batch_size]
      x_2 = x_hmi_inp_data[i*batch_size:(i+1)*batch_size]
      x_3 = beta_train[i*batch_size:(i+1)*batch_size]
      x_1 = torch.tensor(x_1, dtype=torch.float32).to(device)
      x_2 = torch.tensor(x_2, dtype=torch.float32).to(device)
      x_3 = torch.tensor(x_3, dtype=torch.float32).to(device)
      target = y_gst_noi_data[i*batch_size:(i+1)*batch_size]
      target = torch.tensor(target).to(device)

      y = model(x_1, x_2, x_3)
      l = loss_fn(y, target)
      l.backward()
      opti.step()

      running_loss += l.item()
    if train_len > 0:
      print("epoch: ", _+1, " loss: ", running_loss/train_len, flush=True)
      torch.save(model.state_dict(), output_path + "model_bz_new")

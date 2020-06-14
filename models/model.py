from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from models.networks.msra_resnet import get_pose_net as get_resnet
from models.networks.msra_resnet_prune import get_pose_net as get_resnet_prune
from models.networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from models.networks.large_hourglass import get_large_hourglass_net
from models.networks.dlav0 import get_pose_net as get_dlav0
from models.networks.dlav0_module import DLA34_v0 as get_dla34
from models.networks.dlav0_prune import DLA34_v0 as get_dla34_prune
_model_factory = {
  'dla': get_dla_dcn,
  'hourglass': get_large_hourglass_net,
  'dlav0': get_dlav0,
  'dla34': get_dla34,
  'dla35': get_dla34_prune,
  'res':   get_resnet,
  'resPrune': get_resnet_prune,
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  print("num layer is", num_layers)
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def create_model_101_prune(arch, heads, head_conv, percent):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  print("num layer is", num_layers)
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, percent=percent)
  return model


def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  # print("current checkpoint is", checkpoint)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    # print("k is", k)
    if k.startswith('module') and not k.startswith('module_list'):
      print("k is", k)
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model


def load_model_prune(model, model_path):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

  state_dict_ = checkpoint
  state_dict = {}

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):

      state_dict[k[7:]] = state_dict_[k]
    else:
      # print("k is", k)
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, ' \
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)
  print("finished init weights in use prune weights")
  return model


def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)


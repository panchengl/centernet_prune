import torch
import torch.nn as nn
import math
import sys
import numpy as np

sys.setrecursionlimit(100000)
def gather_l1_weights_a(module_list, prune_idx):
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]
    l1_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        sum = torch.sum(module_list[idx][0].weight.data.abs(), )
        l1_weights[index:(index+size)] = torch.flatten(module_list[idx][0].weight.data.abs().clone())
        index += size

    return l1_weights

def gather_l1_weights(model):
    size_list = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            size_list.append(layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels * layer.in_channels)
    l1_weights = torch.zeros(sum(size_list))
    index = 0
    for a, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d):
            size = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels * layer.in_channels
            l1_weights[index:(index+size)] = torch.flatten(layer.weight.data.abs().clone())
            # print('layer.weight.data.abs() is', layer.weight.data.abs().shape)
            # print("output channels is", layer.out_channels)
            index += size
    return l1_weights

def gather_l1_weights_dict(model):
    size_list = []
    for name in model.state_dict():
        if ('conv' in name and "weight"  in name) or ('downsample' in name and "weight" in name) :
            if len(model.state_dict()[name].shape) > 1:
                # print("name is", name)
                # print(model.state_dict()[name].shape)
                channels, N, H, W = model.state_dict()[name].shape
                size_list.append(channels*N*H*W)
    l1_weights = torch.zeros(sum(size_list))
    index = 0
    for name in model.state_dict():
        if ('conv' in name and "weight" in name) or ('downsample' in name and "weight" in name):
            if len(model.state_dict()[name].shape) > 1:
                channels, N, H, W = model.state_dict()[name].shape
                size = channels*N*H*W
                l1_weights[index:(index+size)] = torch.flatten(model.state_dict()[name].abs().clone())
                index += size
    return l1_weights


def get_mask_remain(layer, layer_keep, pruned, thre_l1):
    # pruned = 0
    # if isinstance(layer, nn.Conv2d):
    print("normal layer is", layer)
    channels, N, H, W = layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1]
    min_channel_num = int(layer.out_channels * layer_keep) if int(layer.out_channels * layer_keep) > 0 else 1
    weight_copy_sum = torch.sum(torch.reshape(layer.weight.data.abs(), (channels, -1)), dim=1) / (N * H * W)
    mask = weight_copy_sum.cuda().gt(thre_l1).float()
    if int(torch.sum(mask)) < min_channel_num:
        _, sorted_index_weights = torch.sort(weight_copy_sum, descending=True)
        mask[sorted_index_weights[:min_channel_num]] = 1.
    remain = int(mask.sum())
    pruned = pruned + mask.shape[0] - remain
    return pruned, mask, remain

def obtain_filters_mask_l1(model, thre_l1, layer_keep):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d):
            pruned, mask, remain = get_mask_remain(layer, layer_keep, pruned, thre_l1)
            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
        elif isinstance(layer, nn.Sequential):
            for se_layer in layer.modules():
                if isinstance(se_layer, nn.Conv2d):
                    pruned, mask, remain = get_mask_remain(se_layer, layer_keep, pruned, thre_l1)
                    print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
                elif isinstance(se_layer, nn.Sequential):
                    print("current se_layer is", se_layer)
                    for se_se_layer in layer.modules():
                        if isinstance(se_se_layer, nn.Conv2d):
                            pruned, mask, remain = get_mask_remain(se_se_layer, layer_keep, pruned, thre_l1)
                            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
                        else:
                            print("current se_se_layer is", se_se_layer)
                            channels = se_se_layer.weight.data.shape[0]
                            mask = torch.ones(channels)
                            remain = int(mask.sum())
                            print( 'not prune layer info:  layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
        # else:
    #         print("current layer is  ", layer)
    #         channels = layer.weight.data.shape[0]
    #         mask = torch.ones(channels)
    #         remain = int(mask.sum())
    #         print(f'not prune layer info:  layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
    #     total += mask.shape[0]
    #     num_filters.append(remain)
    #     filters_mask.append(mask.clone())
    # prune_ratio = pruned / total
    # print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
    return num_filters, filters_mask



def obtain_filters_mask_l1_dict(model, thre_l1, layer_keep):
    pruned = 0
    total = 0
    prune_idx = []
    prune_bn_bias_idx = []
    num_filters = []
    filters_mask = []

    for idx, name in enumerate(model.state_dict()):
        # if ('conv' in name and "weight"  in name) or ('downsample' in name ):
        # if ("weight"  in name) or ('downsample' in name ):
        if len(model.state_dict()[name].shape)== 4:
                prune_idx.append(idx)
                channels, N, H, W = model.state_dict()[name].shape[0], model.state_dict()[name].shape[1], model.state_dict()[name].shape[2], model.state_dict()[name].shape[3]
                min_channel_num = int(channels * layer_keep) if int(channels * layer_keep) > 0 else 1
                weight_copy_sum = torch.sum(torch.reshape(model.state_dict()[name].abs(), (channels, -1)), dim=1) / (N * H * W)
                mask = weight_copy_sum.cuda().gt(thre_l1).float()
                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy_sum, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
                total += mask.shape[0]
                num_filters.append(remain)
                filters_mask.append(mask.clone())
        else:
            try:
                # print("prune bias name is", name)
                channels = model.state_dict()[name].shape[0]
                prune_bn_bias_idx.append(idx)
            except:
                continue
            # print(f'not prune layer info:  layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
    return num_filters, filters_mask, prune_idx, prune_bn_bias_idx


def obtain_filters_mask_l1_dict_percent(model, percent_rate):
    pruned = 0
    total = 0
    prune_idx = []
    prune_bn_bias_idx = []
    num_filters = []
    filters_mask = []
    not_prune_idx = []
    for idx, name in enumerate(model.state_dict()):
        # if ('conv' in name and "weight"  in name) or ('downsample' in name ):
        # if ("weight"  in name) or ('downsample' in name ):
        if len(model.state_dict()[name].shape) == 4 and "hm" not in name and "reg" not in name and "wh" not in name and "deconv" not in name:
        # if len(model.state_dict()[name].shape) == 4 and "hm.2" not in name and "reg.2" not in name and "wh.2" not in name:
            prune_idx.append(idx)
            weights =  model.state_dict()[name]
            channels, N, H, W = model.state_dict()[name].shape[0], model.state_dict()[name].shape[1], model.state_dict()[name].shape[2], model.state_dict()[name].shape[3]
            save_channels = int(np.floor(channels * percent_rate))
            channels_weights = torch.flatten(torch.sum(torch.reshape(weights, (channels, -1)), dim=-1))
            sorted_channels_sum, sorted_l1_index = torch.sort(channels_weights)
            thresh_l1 = sorted_channels_sum[int(np.floor(len(sorted_channels_sum) * (1-percent_rate)))]
            mask = channels_weights.gt(thresh_l1).float()
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain
            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.clone())
        # elif "hm" in name or "reg" in name or "wh" in name or "layer4.2.conv3" in name or "layer4.2.bn3" in name:
        elif "hm" in name or "reg" in name or "wh" in name or "deconv" in name:
            not_prune_idx.append(idx)
        else:
            try:
                # print("prune bias name is", name)
                channels = model.state_dict()[name].shape[0]
                prune_bn_bias_idx.append(idx)
            except:
                continue
            # print(f'not prune layer info:  layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t 'f'remaining channel: {remain:>4d}')
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
    return num_filters, filters_mask, prune_idx, prune_bn_bias_idx



def get_prune_id(model):
    CBL_idx = []
    Conv_idx = []
    prune_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    conv_nb = 0

    for conv_id, name in enumerate(model.state_dict()):
        if 'weight' in name and "bn" not in name and len(model.state_dict()[name].shape)>1 and 'fc' not in name:
            CBL_idx.append(conv_id)
            print("name is ", name )
            print("weights shape is ", model.state_dict()[name].shape)
            if "base" in name:
                prune_idx.append(conv_id)
    print("conv nb is", conv_nb)
    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all

def get_name_weights_dict(model):
    name_weights_dict = {}
    id_layer_name_dict = {}
    for i, name in enumerate(model.state_dict()):
        name_weights_dict[name] = model.state_dict()[name]
        id_layer_name_dict[i] = name
    return name_weights_dict, id_layer_name_dict

def get_zero_tensor_num(tensor):
    tensor = torch.flatten(tensor)
    tensor = tensor.cpu().numpy()
    # print("tensor is ", tensor)
    zero_num = 0
    for value in tensor:
        # print("valuse is", value)
        if value<(0.000001):
            zero_num += 1
    return zero_num

def del_current_weights(weights, percent):
    if len(weights.shape) > 1:
        channels, n, w, h = weights.shape
        save_channels = int(np.floor(channels * percent))
        channels_weights = torch.flatten(torch.sum(torch.reshape(weights, (channels, -1)), dim=-1))
        sorted_channels_sum, sorted_l1_index = torch.sort(channels_weights)
        thresh_l1 = sorted_channels_sum[int(np.floor(len(sorted_channels_sum) * percent))]
        mask = channels_weights.gt(thresh_l1).float()
        return mask
        # print("current layer mask is", mask)
    elif len(weights.shape) == 1:
        print("current shape is", weights.shape)
        return 0


def del_current_weights_mask(weights, mask):
    if len(weights.shape) > 1:
        channels, n, w, h = weights.shape
        save_channels = int(np.floor(channels * percent))
        channels_weights = torch.flatten(torch.sum(torch.reshape(weights, (channels, -1)), dim=-1))
        sorted_channels_sum, sorted_l1_index = torch.sort(channels_weights)
        thresh_l1 = sorted_channels_sum[int(np.floor(len(sorted_channels_sum) * percent))]
        mask = channels_weights.gt(thresh_l1).float()
        return mask
        # print("current layer mask is", mask)
    elif len(weights.shape) == 1:
        print("current shape is", weights.shape)
        return 0
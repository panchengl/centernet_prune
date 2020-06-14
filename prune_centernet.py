from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
from models.model import create_model, load_model, save_model, create_model_101_prune
from models.opts import opts
from models.dataloaders.dataset_factory import get_dataset
import numpy as np
from models.trains.ct_trainer import CtTrainer
from models.networks.dlav0_module import DLA34_v0

train_model = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/default/model_best_map_0.48.pth'
global_percent = 0.6
keep_layer = 0.01
percent = 0.8
def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('Creating model...')
    model = create_model('res_101', opt.heads, opt.head_conv)

    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    model, optimizer, start_epoch = load_model(model, train_model, optimizer=optimizer, resume=opt.resume, lr=opt.lr, lr_step=opt.lr_step)
    from utils.prune_util import obtain_filters_mask_l1_dict, gather_l1_weights_dict, obtain_filters_mask_l1_dict_percent
    # l1_weights = gather_l1_weights_dict(model)
    # sorted_l1, sorted_l1_index = torch.sort(l1_weights)
    # thresh_index_l1 = int(len(l1_weights)*global_percent)
    # thresh_l1 = sorted_l1[thresh_index_l1].cuda()
    # print("global percent l1_weights is", thresh_l1)
    # print("len l1 is", len(l1_weights))
    # num_filters, filters_mask, prune_idx, prune_bn_bias_idx= obtain_filters_mask_l1_dict(model, thresh_l1, keep_layer)

    # print(model)
    num_filters, filters_mask, prune_idx, prune_bn_bias_idx= obtain_filters_mask_l1_dict_percent(model, percent)
    print("num_filters is", num_filters)
    print("prune idx is", prune_idx)
    # print("filters_mask is", filters_mask)
    print("prune bn and bias idx is", prune_bn_bias_idx)

    prune_weight_dict = {}
    idx_name = {}
    idx_bias_name = {}
    id_mask = 0
    from utils.prune_util import del_current_weights

    ############################# del channels and save weights and save bias and save bn params ######################
    for idx, name in enumerate(model.state_dict()):
        prune_weight_dict[name] = model.state_dict()[name]
        if idx in prune_idx:
            idx_name[idx] = name
            del_weigh_mask = filters_mask[id_mask]
            id_mask += 1
            del_weigh_mask_numpy = del_weigh_mask.cpu().numpy()
            weights_numpy = model.state_dict()[name].cpu().numpy()
            channel_indices = set(np.arange(len(del_weigh_mask_numpy)))
            non_zero_id = set(list(np.nonzero(del_weigh_mask_numpy)[0]))
            current_zero_id = list(channel_indices.difference(non_zero_id))
            new_weights = np.delete(weights_numpy, current_zero_id, axis=0)
            prune_weight_dict[idx_name[idx]] = torch.from_numpy(new_weights)
            # print("finished del channels name is", name)
            # print("orignal shape is             ", model.state_dict()[name].shape)
            # print("last    shape is             ", prune_weight_dict[name].shape)
        elif idx in prune_bn_bias_idx:
            idx_bias_name[idx] = name

    print("prune idx is ", prune_idx)
    ################################ del input weights and del bias anddel  bn params #################################
    id_mask = 0
    prune_id = 0
    deconv_zero_id = []
    for i, (layer_name, value) in enumerate(prune_weight_dict.items()):
        if i in prune_idx:
            current_idx = i
            try:
                next_idx = prune_idx[prune_idx.index(current_idx) + 1]
            except:
                print("current idx is", current_idx)
                print("len prune idx is", len(prune_idx))
                continue
            if "downsample" in idx_name[next_idx]:
                id_mask -= 3

            del_weigh_mask = filters_mask[id_mask]
            del_weigh_mask_numpy = del_weigh_mask.cpu().numpy()
            channel_indices = set(np.arange(len(del_weigh_mask_numpy)))
            non_zero_id = set(list(np.nonzero(del_weigh_mask_numpy)[0]))
            current_zero_id = list(channel_indices.difference(non_zero_id))
            # print("next layer name is", idx_name[next_idx])
            # print("ori layer name shape is ", prune_weight_dict[idx_name[next_idx]].shape)
            prune_weight_dict[idx_name[next_idx]] = torch.from_numpy(np.delete( prune_weight_dict[idx_name[next_idx]].numpy(), current_zero_id, axis=1) )
            # print("last layer name shape is ", prune_weight_dict[idx_name[next_idx]].shape)
            if "downsample" in idx_name[next_idx]:
                id_mask += 3
            id_mask += 1

            if "downsample" in idx_name[next_idx]:
                print(" idx_name[next_idx] is",  idx_name[next_idx])
                print("current_zero_id is", current_zero_id)
                # break
            prune_id += 1
        elif i in prune_bn_bias_idx:
            current_bias_idx = i
            if id_mask == 0:
                continue
            current_bias_mask_idx = id_mask -1
            del_weigh_mask = filters_mask[current_bias_mask_idx]
            ############# because del channels last layer don not use id_mask+1, so in herem last layer must use current_bias_mask_idx= current_bias_mask_idx+1
            if "layer4.2.bn3" in idx_bias_name[current_bias_idx]:
                # print('idx_bias_name[current_bias_idx] is', idx_bias_name[current_bias_idx])
                del_weigh_mask = filters_mask[current_bias_mask_idx + 1]
                print("ori del mask is",del_weigh_mask )
                del_weigh_mask = filters_mask[-1]
                print("last del mask is", del_weigh_mask)
            del_weigh_mask_numpy = del_weigh_mask.cpu().numpy()
            # print("current name is ", layer_name)
            # print("current prune_idx_bias name is ", idx_bias_name[current_bias_idx])
            # print("orignal bias layer shape is", prune_weight_dict[idx_bias_name[current_bias_idx]].shape)
            channel_indices = set(np.arange(len(del_weigh_mask_numpy)))
            non_zero_id = set(list(np.nonzero(del_weigh_mask_numpy)[0]))
            current_zero_id = list(channel_indices.difference(non_zero_id))
            prune_weight_dict[idx_bias_name[current_bias_idx]] = torch.from_numpy(np.delete(prune_weight_dict[idx_bias_name[current_bias_idx]].numpy(), current_zero_id, axis=0))
            if 'layer4.2.bn3.running_var' == idx_bias_name[current_bias_idx]:
                print("before deconv_layers.0.weight shape is", prune_weight_dict["deconv_layers.0.weight"].shape)
                print("current zero id is", current_zero_id)
                prune_weight_dict["deconv_layers.0.weight"] = torch.from_numpy(np.delete(prune_weight_dict["deconv_layers.0.weight"].numpy(), current_zero_id, axis=0))
                print("last deconv_layers.0.weight shape is", prune_weight_dict["deconv_layers.0.weight"].shape)
                print("finished del deconv input")
    torch.save(prune_weight_dict, 'exp/ctdet/default/l1_prune_model.pt')
    prune_model = create_model_101_prune('resPrune_101', opt.heads, opt.head_conv, percent=percent)
    prune_model.load_state_dict(torch.load('exp/ctdet/default/l1_prune_model.pt'), strict=False)


    #
    # ####################### calc thr_weights ####################################################
    # from utils.prune_util import gather_l1_weights, gather_l1_weights_dict
    # # l1_weights = gather_l1_weights(model)
    # # sorted_l1, sorted_l1_index = torch.sort(l1_weights)
    # # thresh_index_l1 = int(len(l1_weights)*global_percent)
    # # thresh_l1 = sorted_l1[thresh_index_l1].cuda()
    # # print("global percent l1_weights is", thresh_l1)
    # # print("len l1 is", len(l1_weights))
    #
    #
    #
    # l2_weights = gather_l1_weights_dict(model)
    # sorted_l2, sorted_l2_index = torch.sort(l2_weights)
    # thresh_index_l2 = int(len(l2_weights)*global_percent)
    # thresh_l2 = sorted_l2[thresh_index_l2].cuda()
    # print("global percent l2_weights is", thresh_l2)
    # print("len l2 is", len(l2_weights))
    #
    # ###################### get prune mask  #######################################################
    # from utils.prune_util import obtain_filters_mask_l1, obtain_filters_mask_l1_dict, get_zero_tensor_num
    # num_filters, filters_mask = obtain_filters_mask_l1_dict(model, thresh_l2, keep_layer, prune_idx)
    # # print("num_filters is", num_filters)
    # CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    # CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}
    # # print("CBLidx2mask is", CBLidx2mask)
    # print("CBLidx2filters is", CBLidx2filters)
    # from utils.prune_util import get_name_weights_dict
    # name_weight_dict, id_name_dict = get_name_weights_dict(model)
    # def prune_and_eval(model, CBL_idx, CBLidx2mask):
    #     import copy
    #     model_copy = copy.deepcopy(model)
    #     for idx, name in enumerate(model_copy.state_dict()):
    #         if idx in prune_idx:
    #             bn_module = model_copy.state_dict()[name]
    #             mask = CBLidx2mask[idx].cuda()
    #             for i in range(mask.shape[0]):
    #                 if mask[i].cpu().numpy() == 0:
    #                     bn_module[i,:,:,:].zero_()
    #     val_loader = torch.utils.data.DataLoader(Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
    #                                              pin_memory=True)
    #     trainer = CtTrainer(opt, model_copy, optimizer)
    #     trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    #     with torch.no_grad():
    #         log_dict_val, preds = trainer.val(0, val_loader)
    #         val_loader.dataset.run_eval(preds, opt.save_dir)
    #     # print(f'mask the gamma as zero, mAP of the model is {mAP:.4f}')
    #
    # prune_and_eval(model, CBL_idx, CBLidx2mask)
    #
    # for i in CBLidx2mask:
    #     CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()

    # pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

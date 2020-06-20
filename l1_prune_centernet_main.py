from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
from models.model import create_model, load_model, save_model, create_model_101_prune, load_model_prune
from models.opts import opts
from models.dataloaders.dataset_factory import get_dataset
import numpy as np
from models.trains.ct_trainer import CtTrainer

# train_model = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/default/model_best_val_loss_0.64.pth'
train_model = 'exp/ctdet/coco_res_prune/model_best_map_0.48.pth'
prune_model_name = 'prune_model.pth'
percent_rate = 0.8
prune_cnt = 2
ori_prune_cnt = max(prune_cnt-1,0)
print("ori prune_cnt is", ori_prune_cnt)
def mkdir_pth_dir(prune_model_dir):
    if not os.path.isdir(prune_model_dir):
        os.mkdir(prune_model_dir)
        print("success create prune model dir")
    else:
        print("prune model dir is exist")

def main(opt):
    mkdir_pth_dir(opt.save_dir)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('Creating model...')
    model = create_model('resPrune_101', opt.heads, opt.head_conv, percent_rate=percent_rate, prune_cnt=ori_prune_cnt)
    # model = create_model('res_101', opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    model, optimizer, start_epoch = load_model(model, train_model, optimizer=optimizer, resume=opt.resume, lr=opt.lr, lr_step=opt.lr_step)
    from utils.prune_util import obtain_filters_mask_l1_dict, gather_l1_weights_dict, obtain_filters_mask_l1_dict_percent
    num_filters, filters_mask, prune_idx, prune_bn_bias_idx = obtain_filters_mask_l1_dict_percent(model, percent_rate)
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
            prune_weight_dict[idx_name[next_idx]] = torch.from_numpy(
                np.delete(prune_weight_dict[idx_name[next_idx]].numpy(), current_zero_id, axis=1))
            # print("last layer name shape is ", prune_weight_dict[idx_name[next_idx]].shape)
            if "downsample" in idx_name[next_idx]:
                id_mask += 3
            id_mask += 1

            if "downsample" in idx_name[next_idx]:
                print(" idx_name[next_idx] is", idx_name[next_idx])
                print("current_zero_id is", current_zero_id)
                # break
            prune_id += 1
        elif i in prune_bn_bias_idx:
            current_bias_idx = i
            if id_mask == 0:
                continue
            current_bias_mask_idx = id_mask - 1
            del_weigh_mask = filters_mask[current_bias_mask_idx]
            ############# because del channels last layer don not use id_mask+1, so in herem last layer must use current_bias_mask_idx= current_bias_mask_idx+1
            if "layer4.2.bn3" in idx_bias_name[current_bias_idx]:
                # print('idx_bias_name[current_bias_idx] is', idx_bias_name[current_bias_idx])
                del_weigh_mask = filters_mask[current_bias_mask_idx + 1]
                print("ori del mask is", del_weigh_mask)
                del_weigh_mask = filters_mask[-1]
                print("last del mask is", del_weigh_mask)
            del_weigh_mask_numpy = del_weigh_mask.cpu().numpy()
            # print("current name is ", layer_name)
            # print("current prune_idx_bias name is ", idx_bias_name[current_bias_idx])
            # print("orignal bias layer shape is", prune_weight_dict[idx_bias_name[current_bias_idx]].shape)
            channel_indices = set(np.arange(len(del_weigh_mask_numpy)))
            non_zero_id = set(list(np.nonzero(del_weigh_mask_numpy)[0]))
            current_zero_id = list(channel_indices.difference(non_zero_id))
            prune_weight_dict[idx_bias_name[current_bias_idx]] = torch.from_numpy(
                np.delete(prune_weight_dict[idx_bias_name[current_bias_idx]].numpy(), current_zero_id, axis=0))
            if 'layer4.2.bn3.running_var' == idx_bias_name[current_bias_idx]: #in here, must del input in first deconv , mask use layer4.bn3.runing_var mask---no_zero_id
                print("before deconv_layers.0.weight shape is", prune_weight_dict["deconv_layers.0.weight"].shape)
                print("current zero id is", current_zero_id)
                prune_weight_dict["deconv_layers.0.weight"] = torch.from_numpy(
                    np.delete(prune_weight_dict["deconv_layers.0.weight"].numpy(), current_zero_id, axis=0))
                print("last deconv_layers.0.weight shape is", prune_weight_dict["deconv_layers.0.weight"].shape)
                print("finished del deconv input")

    torch.save(prune_weight_dict, os.path.join(opt.save_dir, prune_model_name))
    prune_model = create_model_101_prune('resPrune_101', opt.heads, opt.head_conv, percent_rate=percent_rate, prune_cnt=prune_cnt)
    optimizer = torch.optim.Adam(prune_model.parameters(), opt.lr)
    prune_model.load_state_dict(torch.load(os.path.join(opt.save_dir, prune_model_name)), strict=True)
    start_epoch = 0
    train_loader = torch.utils.data.DataLoader(Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    trainer = CtTrainer(opt, prune_model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    best_val_loss = 1e10
    best_ap = 1e-10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, prune_model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
                val_loader.dataset.run_eval(preds, opt.save_dir)
                # result_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_res_prune/results.json'
                result_json_pth = os.path.join(opt.save_dir, "results.json")
                anno_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/data/dianli/annotations/test.json'
                ap_list, map = trainer.run_epoch_voc(result_json_pth, anno_json_pth, score_th=0.01, class_num=opt.num_classes)
                print(ap_list, map)
            if log_dict_val[opt.metric] <= best_val_loss:
                best_val_loss = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best_val_loss_' + str(round(best_val_loss, 2)) + '.pth'),
                           epoch, prune_model)
            if map > best_ap:
                best_ap = map
                save_model(os.path.join(opt.save_dir, 'model_best_map_' + str(round(best_ap, 3)) + '.pth'), epoch,
                           prune_model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, prune_model, optimizer)
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, prune_model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

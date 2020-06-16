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

prune_model_dir = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_res_prune/'
prune_model_name = 'model_best_map_0.46.pth'
percent = 0.8
def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('Creating model...')
    prune_model = create_model_101_prune('resPrune_101', opt.heads, opt.head_conv, percent=percent)
    optimizer = torch.optim.Adam(prune_model.parameters(), opt.lr)
    prune_model, optimizer, start_epoch = load_model(prune_model, os.path.join(prune_model_dir, prune_model_name), optimizer=optimizer, resume=opt.resume, lr=opt.lr,
                                               lr_step=opt.lr_step)
    # prune_model.load_state_dict(torch.load(os.path.join(prune_model_dir, prune_model_name)), strict=True)
    # start_epoch = 0
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
                save_model(os.path.join(opt.save_dir, 'model_best_map_' + str(round(best_ap, 2)) + '.pth'), epoch,
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

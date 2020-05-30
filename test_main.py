from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
from models.model import create_model, load_model, save_model
from models.opts import opts
from models.dataloaders.dataset_factory import get_dataset

from models.trains.ct_trainer import CtTrainer
from models.logger import Logger

# from models.eval_utils.calc_map import calc_voc_ap, calc_voc_ap_model
from models.detectors.detector_factory import detector_factory
def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    logger = Logger(opt)

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    train_loader = torch.utils.data.DataLoader(Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    trainer = CtTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    # detector = detector_factory[TASK](opt)
    best_val_loss = 1e10
    best_ap = 1e-10
    # with torch.no_grad():
    #     log_dict_val, preds = trainer.val(0, val_loader)
    #     val_loader.dataset.run_eval(preds, opt.save_dir)
    #     result_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_dla0/results.json'
    #     anno_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/data/dianli/annotations/test.json'
    #     ap_list, map = trainer.run_epoch_voc(result_json_pth, anno_json_pth, score_th=0.01, class_num=opt.num_classes)
    #     print(ap_list, map)
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)
        print("log_dict_train is: ", log_dict_train)
        # logger.write('epoch: {} |'.format(epoch))
        # for k, v in log_dict_train.items():
        #     logger.scalar_summary('train_{}'.format(k), v, epoch)
        #     logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
                val_loader.dataset.run_eval(preds, opt.save_dir)
                result_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_hg/results.json'
                anno_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/data/dianli/annotations/test.json'
                ap_list, map = trainer.run_epoch_voc(result_json_pth, anno_json_pth, score_th=0.01, class_num=opt.num_classes)
                print(ap_list, map)
            # for k, v in log_dict_val.items():
            #     logger.scalar_summary('val_{}'.format(k), v, epoch)
            #     logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] <= best_val_loss:
                best_val_loss = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best_val_loss_' +  str(round(best_val_loss, 2)) + '.pth'), epoch, model)
            if map > best_ap:
                best_ap = map
                save_model(os.path.join(opt.save_dir, 'model_best_map_' + str(round(best_ap, 2)) + '.pth'), epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        # logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    # logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

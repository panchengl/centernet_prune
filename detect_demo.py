import sys

from models.detectors.detector_factory import detector_factory
from models.opts import opts
from utils.debugger import Debugger
import cv2
import os

# num_classes = 3
num_classes = 5
pause = True
# vis_thresh = 0.01
vis_thresh = 0.3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# MODEL_PATH ='/home/pcl/pytorch_work/my_github/centernet_simple/weights/model_best_dla34.pth'
MODEL_PATH ='/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_res_prune/model_best_map_0.48.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
# opt = opts().init('{} --load_model {} --flip_test'.format(TASK, MODEL_PATH).split(' '))
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
print(opt)
detector = detector_factory[opt.task](opt)
img_dir = '/home/pcl/pytorch_work/CenterNet-master/dianli_images/'

def show_results(debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, num_classes + 1):
        for bbox in results[j]:
            if bbox[4] > vis_thresh:
                debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=pause)

def inference(img_dir):
    results_imgs = []
    for img in os.listdir(img_dir):
        current_img = img_dir + img
        print('current image is', current_img)
        image = cv2.imread(img_dir + img)
        ret_1 = detector.run(current_img)
        ret = ret_1['results']
        print(ret)
        tot_time = ret_1['tot']
        load_time = ret_1['load']
        pre_time = ret_1['pre']
        net_time =  ret_1['net']
        dec_time = ret_1['dec']
        post_time = ret_1['post']
        merge_time = ret_1['merge']

        print('total time is', tot_time)
        print('load_time time is', load_time)
        print('pre_time time is', pre_time)
        print('net_time time is', net_time)
        print('dec_time time is', dec_time)
        print('post_time time is', post_time)
        print('merge_time time is', merge_time)
        results_imgs.append(ret)
        debugger = Debugger(dataset='dianli', ipynb=False,
                            theme='white')
        show_results(debugger, image, ret)
    return results_imgs

def inference_img(img_dir):
    current_img = img_dir
    # print('current image is', current_img)
    image = cv2.imread(current_img)
    ret = detector.run(current_img)['results']
    return ret

if __name__ == "__main__":
    inference(img_dir=img_dir)

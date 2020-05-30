from __future__ import division, print_function
from utils.common_util import read_class_names, get_classes_standard_dict

import numpy as np
dataset = "dianli"
score_th = 0.3
via_flag = False

backbone = "dla_34"
# common params
root_data_dir = "/home/pcl/tf_work/map/data"
names_file = '/home/pcl/tf_work/map/data/zhongkeyaun6label.names'
# names_file = '/home/pcl/tf_work/map/data/coco.names'

names_class = read_class_names(names_file)
class_dict = get_classes_standard_dict(names_class)
class_num = len(names_class)
print("class is", class_num)

#centernet params
not_cuda_benchmark = False
test = False
flip_test = False
cat_spec_wh = False
seed = 312
K = 100
nms = False
down_ratio = 4
center_thresh = 0.01
pause = False
gpus = [4,5]

arch = 'dla_34'
# arch = 'hourglass'
heads = {'hm': class_num, 'wh': 2 , 'reg': 2}
head_conv = 256 if 'dla' in arch else 64
load_model ='/home/pcl/tf_work/map/weights/model_best_dla34.pth'

ori_mean = np.array([0.40789654, 0.44719302, 0.47026115],
                dtype=np.float32).reshape(1, 1, 3)
ori_std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)
mean = np.array(ori_mean, dtype=np.float32).reshape(1, 1, 3)
std = np.array(ori_std, dtype=np.float32).reshape(1, 1, 3)

test_scales = [1.0]
fix_res = False
pad = 127 if 'hourglass' in arch else 31
num_stacks = 2 if arch == 'hourglass' else 1
dataset = 'dianli'
debugger_theme = 'white'
debug = 0

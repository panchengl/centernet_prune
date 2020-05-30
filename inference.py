# import sys
# CENTERNET_PATH = "/home/pcl/pytorch_work/CenterNet-master/src/lib"
# sys.path.insert(0, CENTERNET_PATH)

from centernet.detector_factory import detector_factory
from centernet.debugger import Debugger
import cv2
import os
import cfg

def show_results(debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, cfg.class_num + 1):
        for bbox in results[j]:
            if bbox[4] > cfg.score_th:
                debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=cfg.pause, show=True)

def inference(img_dir, model):
    results_imgs = []
    for img in os.listdir(img_dir):
        current_img = img_dir + img
        # print('current image is', current_img)
        image = cv2.imread(img_dir + img)
        ret_1 = model.run(current_img)
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

def inference_img(detector, img_dir):
    current_img = img_dir
    # print('current image is', current_img)
    image = cv2.imread(current_img)
    result = detector.run(current_img)['results']
    return result

def centernet_inference_single_img(detector, img_dir, via_flag=False):
    current_img = os.path.join(cfg.img_dir,img_dir)
    # print('current image is', current_img)
    result = detector.run(current_img)['results']
    if via_flag:
        image = cv2.imread(current_img)
        debugger = Debugger(dataset='dianli', ipynb=False,
                            theme='white')
        show_results(debugger, image, result)
    return result

def get_preds_gpu(id_list, name_list, number):
    '''
    Given the y_pred of an input image, get the predicted bbox and label info.
    return:
        pred_content: 2d list.
    '''
    image_id = id_list[number]
    inference_img_name = name_list[number]
    pred_content = []
    results = inference_img(inference_img_name)
    for j in range(1, cfg.class_num + 1):
        for bbox in results[j]:
            if bbox[4] > cfg.score_th:
                x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
                score = bbox[4]
                label = j -1
                pred_content.append([image_id, x_min, y_min, x_max, y_max, score, label])
    print("obj is", pred_content)
                # debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    return pred_content


if __name__ == "__main__":
    # MODEL_PATH ='/home/pcl/tf_work/map/weights/model_best_dla34.pth'
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    detector = detector_factory[TASK]()
    img_dir = '/home/pcl/pytorch_work/CenterNet-master/dianli_images/'
    inference(img_dir=img_dir, model=detector)

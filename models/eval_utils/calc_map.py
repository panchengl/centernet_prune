from models.eval_utils.ap_utils import parse_gt_rec,  voc_eval, AverageMeter
import json
def calc_voc_map(result_json_pth, anno_json_pth, score_th=0.3, class_num=5):
    try:
        file = open(result_json_pth, "rb")
    except:
        print(" No such file or directory %s"%result_json_pth)
        ap_list = [0,0,0,0,0]
        map = 0
        return ap_list, map
    fileJson = json.load(file)
    pred_content = []
    for file in fileJson:
        # print(file)
        if file['score'] > score_th:
            gt_xmin, gt_ymin, gt_w, gt_h = float(file['bbox'][0]), float(file['bbox'][1]), float(file['bbox'][2]), float(file['bbox'][3])
            gt_box = [ int(file['image_id']), gt_xmin, gt_ymin, gt_xmin+gt_w, gt_ymin+gt_h, float(file['score']), int(file['category_id']) ]
            pred_content.append(gt_box)
    test_file = open(anno_json_pth, 'rb')
    val_json = json.load(test_file)
    imgs_info_list = val_json['images']
    box_info_list = val_json['annotations']
    img_box_id_info = {}
    for info in imgs_info_list:
        img_box_id_info[int(info['id'])] = []
    for info_dict in box_info_list:
        info = list(info_dict['bbox'])
        xmin, ymin, w, h = float(info[0]), float(info[1]), float(info[2]), float(info[3])
        xmax = xmin + w
        ymax = ymin + h
        bbox = [xmin, ymin, xmax, ymax, int(info_dict['category_id'])]
        img_box_id_info[int(info_dict['image_id'])].append(bbox)
    gt_dict = {}
    for key, values in img_box_id_info.items():
        if len(values) != 0:
            gt_dict[key] = values
        else:
            gt_dict[key] = [[0, 0, 0, 0, 100]]
    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    info = ""
    ap_list = []
    for ii in range(class_num):
        npos, nd, rec, prec, ap = voc_eval(gt_dict, pred_content, ii, iou_thres=0.5, use_07_metric=False)
        info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)
        ap_list.append(ap)
    mAP = ap_total.average
    info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average, mAP)
    print(info)
    return ap_list, mAP

if __name__ == "__main__":
    result_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_dla0/results.json'
    anno_json_pth = '/home/pcl/pytorch_work/my_github/centernet_simple/data/dianli/annotations/test.json'
    ap_list, map = calc_voc_map(result_json_pth, anno_json_pth, score_th=0.01, class_num=5)


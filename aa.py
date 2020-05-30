# import json
#
#
# path = '/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_dla0/results.json'
# test_json = '/home/pcl/pytorch_work/my_github/centernet_simple/data/dianli/annotations/test.json'
# # json.loads("/home/pcl/pytorch_work/my_github/centernet_simple/exp/ctdet/coco_dla0/results.json")
# file = open(path, "rb")
# fileJson = json.load(file)
# score_th = 0.3
# pred_content = []
# for file in fileJson:
#     # print(file)
#     if file['score'] > score_th:
#         pred_content.append([ file['image_id'], float(file['bbox'][0]),  float(file['bbox'][1]),  float(file['bbox'][2]),  float(file['bbox'][3]), float(file['score']), int(file['category_id']) ])
#         # print(file)
# print("pred_content is", pred_content)
# test_file = open(test_json, 'rb')
# val_json = json.load(test_file)
# gt_content = []
# imgs_info_list = val_json['images']
# box_info_list = val_json['annotations']
# img_ids = []
# img_box_id_info = {}
# for info in imgs_info_list:
#     # img_ids.append(info['id'])
#     img_box_id_info[info['id']] = []
# for info_dict in box_info_list:
#     info = list(info_dict['bbox'])
#     info.append(int(info_dict['category_id']))
#     img_box_id_info[info_dict['image_id']].append(info)
# # print(img_ids)
# print("gt is ", img_box_id_info)


a = '34 8 44 54 4 40 32 19 7 56 22 25 47 3 11 6 42 0 48 41 16 33 49 35 28 58 2 13 31 17 5 10 39 59 60 46 18 20 29 55 50 51 52 63 23 45 30 14 62 37 12'
print(len(a.split(' ')))
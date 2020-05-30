import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = -1
image_id = 20180000000
annotation_id = 0
orignal_class = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe", "ShanHuo", "YanWu", "SuLiaoBu"]
class_sgjx = ["TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
class_yanhuo = ["YanWu", "ShanHuo"]
# class_3label = ["DiaoChe", "TaDiao", "ShiGongJiXie"]
class_zhongkeyuan = ["DiaoChe", "TaDiao", "ShiGongJiXie", "YanHuo", "SuLiaoBu"]


def initItemId():
    global category_item_id
    for name in class_zhongkeyuan:
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item_id += 1
        category_item['id'] = category_item_id
        category_item['name'] = name
        coco['categories'].append(category_item)
        category_set[name] = category_item_id
    print('category_set is ', coco['categories'])

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def parseXmlFiles(xml_path):
    error_num = 0
    for i, f in enumerate(os.listdir(xml_path)):
        if i >100:
            break
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    print("current image_set is", image_set)
                    error_num += 1
                    break
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        print("true name is",object_name)
                        if object_name in class_sgjx:
                            # current_category_id = addCatItem(object_name)
                            current_category_id = category_set["ShiGongJiXie"]
                            print("but is SGJX current c=id is", current_category_id)
                        elif object_name in class_yanhuo:
                            current_category_id = category_set["YanHuo"]
                            print("but is YanHuo current c=id is", current_category_id)
                        else:
                            break
                    else:
                        current_category_id = category_set[object_name]
                        print("current c=id is", current_category_id)

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    print("object_name is", object_name)
                    if object_name in orignal_class:
                        if object_name is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        bbox = []
                        # x
                        bbox.append(bndbox['xmin'])
                        # y
                        bbox.append(bndbox['ymin'])
                        # w
                        bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        # h
                        bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        print("object name is", object_name)
                        if object_name in class_sgjx:
                            print("orignal class is %s but change ShiGongJiXie"%object_name)
                            object_name = "ShiGongJiXie"
                        if object_name in class_yanhuo:
                            print("orignal class is %s but change YanHuo"%object_name)
                            object_name = "YanHuo"
                        print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                                                                       bbox))
                        addAnnoItem(object_name, current_image_id, current_category_id, bbox)
    print("error num is", error_num)

if __name__ == '__main__':
    xml_path = "/home/pcl/data/VOC2007/AnnotationsTest" # 这是xml文件所在的地址
    # xml_path = "/home/pcl/data/dianli_zhongkeyuan" # 这是xml文件所在的地址
    json_file = '/home/pcl/pytorch_work/my_github/centernet_simple/data/dianli_test/annotations/test.json'  # 这是你要生成的json文件

    initItemId()
    parseXmlFiles(xml_path)  # 只需要改动这两个参数就行了
    json.dump(coco, open(json_file, 'w'))



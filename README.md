# centernet_simple

first: this code is forked from https://github.com/xingyizhou/CenterNet

changes:

    1. del any other tasks , only support 2d detection in order to convienent understanding and use

    2. In the original code, the best loss is used as the requirement for saving the model, but usually the minimum loss does not mean the highest ap, so ap50 is added as a criterion for saving model

    3. The calculation method of coco ap50 is different from my calculation method. I use the calculation method of voc challenge provided by facebook, but the difference is about 1/1000~5/1000

     https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py

    4. add P-R curve for every class, just like 0_PR.jpg

support:

    backbone: dla_34v0, dla_34_dcn, hg104

attention:

    1. hg104 performance is worse than dla_34 or dla_v0

    2. dla_34_dcn backbone fps in lower than paper, about 50ms and some postprocess time in 2080Ti



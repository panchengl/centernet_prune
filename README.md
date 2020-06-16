20200614 updates:
    add centernet prune, attention:

    1. this prune version only support res101_no_dcn backbone, res101_dcn u can try, dla houglass backbone is too complex, i have no time reconstruction network backbone by code when pruning this backbone.

    2. the prune algorithm principle is according to l1_weights(DeepCompression-by HanSong), the next version i will update sliming prune(using bn layer gamma params as importance criterion)

    3. attention: this version only support to each layer is pruned with a fixed percentage, if u are intrested global percent, u can pushes issues for me, global percent prune is more effectiveness and robustness

    4. how to use:

        first: you can train a best model(best map) in using my code

        second: python l1_prune_centernet_main.py ctdet --exp_id coco_res_prune --gpus 0 --test (attention: if u want use map as criterion, u must add --test)

        last: the code will prune one model, the default channels pruning rate is 0.2(204M -> 145M). and the code will auto finetune model , u just wait to get a best model

        attentions: if u early stop tarining or want finetune a prune model, u need do this command: python finetune_main.py ctdet --exp_id coco_res_prune --gpus 0 --test

    5. the map values compared:

        in my datasets, the scene is more complicated than coco datasets, map(204M) is 0.49 before prune, map(139M) also is 0.47 after prune(pruning rate is 0.2), so, u can use this code in your datasets

     6. next stages:

        1. change annother prune algorithm, just like sliming, kmeans(just like my another repository-yolov3_prune)

        2. if time support, i will add channel merge details in code, it`s more effective than algorithm changed, but it depends on the my mood and time.....


# centernet_simple

first: this code is forked from https://github.com/xingyizhou/CenterNet

changes:

    1. del any other tasks , only support 2d detection in order to convienent understanding and use

    2. In the original code, the best loss is used as the requirement for saving the model, but usually the minimum loss does not mean the highest ap, so ap50 is added as a criterion for saving model

    3. The calculation method of coco ap50 is different from my calculation method. I use the calculation method of voc challenge provided by facebook, but the difference is about 1/1000~5/1000

     https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py

    4. add P-R curve for every class, just like 0_PR.jpg

support:

    backbone: dla_34v0, dla_34_dcn, hg104, res101

attention:

    1. hg104 performance is worse than dla_34 or dla_v0

    2. dla_34_dcn backbone fps in lower than paper, about 50ms and some postprocess time in 2080Ti

    3. res101 is worse than hg104 and dla_34

how to train:

    python test_main.py  ctdet --exp_id coco_dla --batch_size 32  --lr 1.25e-4 --gpus 0,1
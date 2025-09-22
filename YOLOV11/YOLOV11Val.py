#!/usr/bin/env python
# _*_ coding: utf-8 _*_
from ultralytics import YOLO
import time
import numpy as np
import os
import pandas as pd
import torch
from torchutil import engine, coco_utils, utils

from Utility import PublicFunction
import config

device = torch.device('cpu')

if __name__ == '__main__':
    best_Or_last = "last"  # best,last
    size = 512

    rootModelDir = os.path.join(config.ModelDir, "YOLOV11")
    image_path = config.YOLODir
    json_path = config.COCODir

    for i in range(0, 1):
        model_name = os.path.join(rootModelDir, r'{size}_{i}\weights\{best_Or_last}.pt'.format(
            size=size, i=i,
            best_Or_last=best_Or_last))

        model = YOLO(model_name)
        model.eval()
        starttime = time.time()

        test_img_folder = os.path.join(image_path, "images", "val", "{size}".format(size=size))
        test_ann_file = os.path.join(json_path, 'val{size}.json'.format(size=size))
        train = False

        testDataset = coco_utils.CocoDetection(test_img_folder, test_ann_file, None)

        test_data_loader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=10,
            shuffle=False,
            num_workers=4,
            collate_fn=utils.collate_fn
        )

        # overwrite yolo metrics,because of the different caculate AP method
        coco_evaluator, test_metric_logger = engine.YOLOEvaluate(model, test_data_loader, device=device)

        mAP05_95 = coco_evaluator.coco_eval['bbox'].stats[0]  # IoU=0.50:0.95
        mAP50 = coco_evaluator.coco_eval['bbox'].stats[1]  # IoU=0.50
        mAP75 = coco_evaluator.coco_eval['bbox'].stats[2]  # IoU=0.75
        # AP_3 = coco_evaluator.coco_eval['bbox'].stats[3]
        # AP_4 = coco_evaluator.coco_eval['bbox'].stats[4]
        # AP_5 = coco_evaluator.coco_eval['bbox'].stats[5]
        #
        # AR_0 = coco_evaluator.coco_eval['bbox'].stats[6]
        # AR_1 = coco_evaluator.coco_eval['bbox'].stats[7]
        # AR_2 = coco_evaluator.coco_eval['bbox'].stats[8]
        # AR_3 = coco_evaluator.coco_eval['bbox'].stats[9]
        # AR_4 = coco_evaluator.coco_eval['bbox'].stats[10]
        # AR_5 = coco_evaluator.coco_eval['bbox'].stats[11]
        recall = coco_evaluator.coco_eval['bbox'].stats[12]  # Recall IOU=0.5
        precision = coco_evaluator.coco_eval['bbox'].stats[13]  # Precision IOU=0.5

        cocoeval_result = coco_evaluator.coco_eval['bbox'].eval
        pr_array_0_5 = np.array(cocoeval_result['precision'])[0, :, 0, 0, 2].reshape((-1, 1))
        recall_array = np.arange(0.0, 1.01, 0.01).reshape((-1, 1))
        mAP05_95_Array = np.full(np.shape(pr_array_0_5), mAP05_95)
        mAP50_Array = np.full(np.shape(pr_array_0_5), mAP50)
        mAP75_Array = np.full(np.shape(pr_array_0_5), mAP75)
        recall_array_0_5 = np.full(np.shape(pr_array_0_5), recall)
        precision_array_0_5 = np.full(np.shape(pr_array_0_5), precision)

        endtime = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(starttime)))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(endtime)))
        print(endtime - starttime)
        modeltime = endtime - starttime
        timesum_Array = np.full(np.shape(pr_array_0_5), modeltime)
        modelDir = os.path.join(rootModelDir, r"{size}_{i}\val".format(size=size, i=i))
        PublicFunction.mkDir(modelDir)
        PRNameList = ["RecallX", "PR50", "mAP50-95", "mAP50", "mAP75", "time_sum", "Recall50", "precision50"]
        csvAccFileName = os.path.join(modelDir, "resultPR_AP_{bestorlast}.csv".format(bestorlast=best_Or_last))
        dfAcc = pd.DataFrame(np.concatenate([recall_array, pr_array_0_5,
                                             mAP05_95_Array, mAP50_Array, mAP75_Array, timesum_Array, recall_array_0_5,
                                             precision_array_0_5], axis=1),
                             columns=PRNameList)
        dfAcc.to_csv(csvAccFileName)

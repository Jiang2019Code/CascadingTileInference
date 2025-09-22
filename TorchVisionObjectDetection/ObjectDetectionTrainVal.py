#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import numpy as np
import pandas as pd
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchutil import engine, coco_utils, utils
from torchutil import transforms as T
import os
from functools import partial
import time
from Utility import PublicFunction
import config

starttime = time.time()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# FasterRCNN model
def get_fasterrcnn_resnet50_fpn_model(size, num_classes, weights=True):
    if weights is True:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    model.transform.min_size = (size,)
    model.transform.max_size = size
    # num_classes =kwargs["num_classes"]

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # size = kwargs["size"]

    return model


# MobileNet model
def get_fasterrcnn_mobilenet_v3_large_320_fpn_model(size, num_classes, weights=True):
    # num_classes = kwargs["num_classes"]
    # size = kwargs["size"]
    if weights is True:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    else:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()

    model.transform.min_size = (size - 10,)
    model.transform.max_size = size + 10
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# SSD model
def get_ssd300_vgg16(size=1024, num_classes=2, weights=True):
    if weights is True:
        model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT, size=size
        )
    else:
        model = torchvision.models.detection.ssd300_vgg16(
        )
    # Image size for transforms.
    model.transform.min_size = (size - 10,)
    model.transform.max_size = size + 10

    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    print(in_channels)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    return model


# RetinaNet model
def get_retinanet_resnet50_fpn(size=1024, num_classes=2, weights=True):
    if weights is True:
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
    else:
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        )

    # Retrieve the list of input channels.
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))[0]
    num_anchors = model.head.classification_head.num_anchors
    # List containing number of anchors based on aspect ratios.
    # num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    # Image size for transforms.
    # 去掉Transform,精度就不为0了
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model


def get_transform(train=False):
    transformsList = []
    transformsList.append(coco_utils.ConvertCocoPolysToMask())
    transformsList.append(T.PILToTensor())
    if train:
        transformsList.append(T.RandomRotation())
    transformsList.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transformsList)


def getDataset(train_img_folder, train_ann_file, test_img_folder, test_ann_file, train_transforms, test_transform):
    trainDataset = coco_utils.CocoDetection(train_img_folder, train_ann_file, train_transforms)

    testDataset = coco_utils.CocoDetection(test_img_folder, test_ann_file, test_transform)
    train_data_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=10,
        shuffle=True,
        num_workers=5,
        collate_fn=utils.collate_fn
    )

    test_data_loader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )
    return train_data_loader, test_data_loader



def FScore(PA, UA):
    F = 2 * PA * UA / (PA + UA)
    return F


starttime = time.time()


def train_model(num_epochs, train_data_loader, test_data_loader, modelName="FasterRCNN", **kwargs):
    modelDir = kwargs["modelDir"]
    PublicFunction.mkDir(modelDir)

    if "epochList" in kwargs.keys():
        print(kwargs)
        epochList = kwargs["epochList"]

    # # let's train_old it just for 2 epochs
    modelFileName = os.path.join(modelDir, '{modelName}_last.pth'.format(modelName=modelName))
    model = None
    if PublicFunction.check_existence(modelFileName):
        print(modelFileName)
        model = torch.load(modelFileName)
    else:
        if modelName == "FasterRCNN":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            model = get_fasterrcnn_resnet50_fpn_model(size=size, num_classes=num_classes, weights=True)
        elif modelName == "SSD":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            model = get_ssd300_vgg16(size=size, num_classes=num_classes)
        elif modelName == "Retinanet":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            model = get_retinanet_resnet50_fpn(size=size, num_classes=num_classes)
        elif modelName == "MobileNet":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            model = get_fasterrcnn_mobilenet_v3_large_320_fpn_model(size=size, num_classes=num_classes)
        model.to(device)

    model.train()

    timeDict = {}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005
    )
    trainLossFileName = os.path.join(modelDir, "TrainLoss.csv")
    valLossFileName = os.path.join(modelDir, "ValLoss.csv")

    best_acc = 0
    valList = []
    PRList = []
    epoch_test_acc = -1
    starttime = time.time()

    for epoch in range(num_epochs):
        train_metric_logger = engine.train_one_epoch(model, optimizer, train_data_loader, test_data_loader, device,
                                                     epoch, print_freq=10)
        loss_str = []
        col_name_list = []
        for name, meter in train_metric_logger.meters.items():
            # loss_str.append(f"{name}: {str(meter)}")
            col_name_list.append(name)
            loss_str.append(str(meter))

        valList.append(loss_str)
        # lr_scheduler.step()
        # evaluate on the test dataset

        if epoch == 0:
            epoch_coco_evaluator, epoch_test_metric_logger = engine.evaluate(model, test_data_loader, device=device)
            epoch_test_acc = epoch_coco_evaluator.coco_eval['bbox'].stats[0]
            AP_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[0]
            AP_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[1]
            AP_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[2]
            AP_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[3]
            AP_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[4]
            AP_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[5]

            AR_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[6]
            AR_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[7]
            AR_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[8]
            AR_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[9]
            AR_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[10]
            AR_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[11]
            AR_6 = epoch_coco_evaluator.coco_eval['bbox'].stats[12]
            precision = epoch_coco_evaluator.coco_eval['bbox'].stats[13]

            PRList.append([AP_0, AP_1, AP_2, AP_3, AP_4, AP_5, AR_0, AR_1, AR_2, AR_3, AR_4, AR_5, AR_6, precision])

        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, os.path.join(modelDir, '{modelName}_best.pth'.format(modelName=modelName)))
        if epoch == num_epochs - 1 or epoch % 5 == 0:
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, os.path.join(modelDir, '{modelName}_last.pth'.format(modelName=modelName)))

            epoch_coco_evaluator, epoch_test_metric_logger = engine.evaluate(model, test_data_loader, device=device)
            epoch_test_acc = epoch_coco_evaluator.coco_eval['bbox'].stats[0]

            AP_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[0]
            AP_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[1]
            AP_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[2]
            AP_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[3]
            AP_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[4]
            AP_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[5]

            AR_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[6]
            AR_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[7]
            AR_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[8]
            AR_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[9]
            AR_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[10]
            AR_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[11]
            AR_6 = epoch_coco_evaluator.coco_eval['bbox'].stats[12]
            precision = epoch_coco_evaluator.coco_eval['bbox'].stats[13]

            PRList.append([AP_0, AP_1, AP_2, AP_3, AP_4, AP_5, AR_0, AR_1, AR_2, AR_3, AR_4, AR_5, AR_6, precision])

        endtime = time.time()
        timeDict["YOLO"] = [endtime - starttime]
        csvpath = os.path.join(modelDir, str(512) + "time.csv")
        dataformat = pd.DataFrame(timeDict)
        dataformat.to_csv(csvpath)

        lossDf = pd.DataFrame(np.asarray(valList), columns=col_name_list)
        PRDf = pd.DataFrame(np.asarray(PRList), columns=["AP_0", "mAP50", "AP_2", "AP_3", "AP_4", "AP_5",
                                                         "AR_0", "AR_1", "AR_2", "AR_3", "AR_4", "AR_5", "recall",
                                                         "Precision_10"])
        lossDf.to_csv(trainLossFileName)
        PRDf.to_csv(valLossFileName)
        if "epochList" in kwargs.keys():
            epochList.append(epoch)


def val_test(test_data_loader, **kwargs):
    model_weight = kwargs["model_weight"]
    bestorlast = kwargs["bestorlast"]

    modelDir = PublicFunction.getFileExtName(model_weight)[0]

    #   # torch.save(model.state_dict(), model_path)但实际上它保存的不是模型文件，而是参数文件文件。在模型文件中，存储完整的模型，而在状态文件中，仅存储参数。因此，collections.OrderedDict只是模型的值。
    model = torch.load(model_weight)
    model.to(device)
    starttime = time.time()

    coco_evaluator, test_metric_logger = engine.evaluate(model, test_data_loader, device=device)

    cocoeval_result = coco_evaluator.coco_eval['bbox'].eval

    pr_array_0_5 = np.array(cocoeval_result['precision'])[0, :, 0, 0, 2].reshape((-1, 1))
    pr_array_0_75 = np.array(cocoeval_result['precision'])[5, :, 0, 0, 2].reshape((-1, 1))
    recall_array_0_5 = np.array(cocoeval_result['recall'])[0, 0, 0, 2].reshape((-1, 1))

    recall_array = np.arange(0.0, 1.01, 0.01).reshape((-1, 1))
    modeltime = None
    for name, meter in test_metric_logger.meters.items():
        if name == "model_time_sum":
            modeltime = meter

    # 保存精度
    AP_0 = coco_evaluator.coco_eval['bbox'].stats[0]
    AP_1 = coco_evaluator.coco_eval['bbox'].stats[1]
    AP_2 = coco_evaluator.coco_eval['bbox'].stats[2]
    AP_0_Array = np.full(np.shape(pr_array_0_75), AP_0)
    AP_1_Array = np.full(np.shape(pr_array_0_75), AP_1)
    AP_2_Array = np.full(np.shape(pr_array_0_75), AP_2)

    endtime = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(starttime)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(endtime)))
    print(endtime - starttime)
    timesum_Array = np.full(np.shape(pr_array_0_75), modeltime)
    recall_array_0_5 = np.full(np.shape(pr_array_0_75), recall_array_0_5)

    PRNameList = ["Recall", "PR50", "PR75", "mAP50-95", "mAP50", "mAP75", "time_sum", "Recall50", "AR_6"]
    csvAccFileName = os.path.join(modelDir, "resultPR_AP_{bestorlast}.csv".format(bestorlast=bestorlast))
    dfAcc = pd.DataFrame(np.concatenate([recall_array, pr_array_0_5, pr_array_0_75,
                                         AP_0_Array, AP_1_Array, AP_2_Array, timesum_Array, recall_array_0_5], axis=1),
                         columns=PRNameList)
    dfAcc.to_csv(csvAccFileName)


def val_test_cocoeval(test_data_loader, **kwargs):
    starttime = time.time()

    model_weight = kwargs["model_weight"]
    bestorlast = kwargs["bestorlast"]

    modelDir = PublicFunction.getFileExtName(model_weight)[0]
    valDir = os.path.join(modelDir, "val")
    PublicFunction.mkDir(valDir)

    model = torch.load(model_weight)
    model.to(device)

    coco_evaluator, test_metric_logger = engine.evaluate(model, test_data_loader, device=device)

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
    # pr_array_0_75 = np.array(cocoeval_result['precision'])[5, :, 0, 0, 2].reshape((-1, 1))#0.75IOU
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

    PRNameList = ["RecallX", "PR50", "mAP50-95", "mAP50", "mAP75", "time_sum", "Recall50", "precision50"]
    csvAccFileName = os.path.join(valDir, "resultPR_AP_{bestorlast}.csv".format(bestorlast=bestorlast))
    dfAcc = pd.DataFrame(np.concatenate([recall_array, pr_array_0_5,
                                         mAP05_95_Array, mAP50_Array, mAP75_Array, timesum_Array, recall_array_0_5,
                                         precision_array_0_5], axis=1),
                         columns=PRNameList)
    dfAcc.to_csv(csvAccFileName)

def predict_test(test_data_loader, modelName="FasterRCNN", score_threshold=0.5, **kwargs):
    modelDir = kwargs["modelDir"]
    outputDir = os.path.join(modelDir, modelName + "Predict")
    PublicFunction.mkDir(outputDir)
    model_weight = kwargs["model_weight"]

    model = torch.load(model_weight)
    model.to(device)

    predict_test_logger = engine.predictTest(model, test_data_loader, device, score_threshold=score_threshold,
                                             outputDir=outputDir)


if __name__ == '__main__':

    # Train
    image_path = config.YOLODir
    json_path = config.COCODir
    modelNameList = ["Retinanet"] + ["SSD", "MobileNet", "Retinanet"] + ["FasterRCNN"]
    sizeList = [512]
    num_classes = 2
    num_epochs = 200
    for modelName in modelNameList:
        for size in sizeList:
            for i in range(0, 5):
                train_img_folder = os.path.join(image_path, "images", r"train\{size}".format(size=size))
                train_ann_file = os.path.join(json_path, 'train{size}.json'.format(size=size))
                test_img_folder = os.path.join(image_path, "images", r"val\{size}".format(size=size))
                test_ann_file = os.path.join(json_path, 'val{size}.json'.format(size=size))
                train = True
                train_transforms, test_transform = get_transform(train), get_transform()
                train_data_loader, test_data_loader = getDataset(train_img_folder, train_ann_file, test_img_folder,
                                                                 test_ann_file, train_transforms, test_transform)
                modelDir = os.path.join(config.ModelDir, modelName, str(size) + "_" + str(i))
                train_model(num_epochs, train_data_loader, test_data_loader, modelName=modelName, size=512,
                            num_classes=2, modelDir=modelDir)

    # Validation
    sizeList = [512]
    best_Or_last = "last"  # best,last
    modelNameList = ["FasterRCNN"] + ["SSD", "MobileNet", "Retinanet"]
    json_path = config.COCODir

    for modelName in modelNameList:
        for size in sizeList:
            for i in range(0, 5):
                image_path = config.YOLODir

                train_img_folder = os.path.join(image_path, "images", "train", "{size}".format(size=size))
                train_ann_file = os.path.join(json_path, 'train{size}.json'.format(size=size))
                test_img_folder = os.path.join(image_path, "images", "val", "{size}".format(size=size))
                test_ann_file = os.path.join(json_path, 'val{size}.json'.format(size=size))
                train = False
                train_transforms, test_transform = get_transform(train), get_transform()
                train_data_loader, test_data_loader = getDataset(train_img_folder, train_ann_file, test_img_folder,
                                                                 test_ann_file, train_transforms, test_transform)

                modelDir = os.path.join(config.ModelDir,modelName, str(size) + "_" + str(i))
                model_weight = os.path.join(modelDir, r"{modelName}_{bestorlast}.pth".format(modelName=modelName, bestorlast=best_Or_last))

                # val_test( test_data_loader,modelName=modelName,modelDir=modelDir,model_weight=model_weight,bestorlast=best_Or_last)
                val_test_cocoeval(test_data_loader, modelName=modelName, modelDir=modelDir, model_weight=model_weight, bestorlast=best_Or_last)

#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import torch
import numpy as np
import config
import time
from PIL import Image
from torchvision.transforms import functional as F
import pandas as pd
import math
from pycocotools import coco
from Utility import PublicFunction
from Utility import parallComputer
from ultralytics.utils.plotting import Annotator, colors
import os
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
import cv2

def getCocoAnno(cocodata, imgName):
    catIds = cocodata.getCatIds(catNms=['DiKengYuan'])  # 根据类型获取类型id
    imgIds = cocodata.getImgIds(catIds=catIds)
    resultRectList = []
    for index in range(len(imgIds)):
        img = cocodata.loadImgs(imgIds[index])[0]
        if imgName == PublicFunction.getFileExtName(img['file_name'])[1] + '.jpg':
            annIds = cocodata.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = cocodata.loadAnns(annIds)
            for ann in anns:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                resultRectList.append([bbox_x, bbox_y] + [bbox_x + bbox_w, bbox_y + bbox_h])
    return resultRectList

def plot(img, pred_boxes, labels=None, scores=None, line_width=None, font_size=None, font="Arial.ttf",
         color_mode="class", filename="", trueboxlist=None):
    # """
    # Plots detection results on an input RGB image.
    annotator = Annotator(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), line_width, font_size, font)
    if labels is not None:
        labels = [f"Silo-cave: {score:.2f}" for label, score in zip(labels, scores)]
    else:
        if trueboxlist is not None:
            labels = []
            for p in trueboxlist:
                labels.append("Ground Truth")
    if trueboxlist is not None:
        for i, d in enumerate(trueboxlist):
            annotator.box_label(
                d,
                labels[i],
                color=(0, 0, 255)
            )
    if pred_boxes is not None:
        for i, d in enumerate(pred_boxes):
            label = labels[i]
            annotator.box_label(
                d,
                label,
                color=colors(
                    0
                    if color_mode == "class"
                    else id
                    if id is not None
                    else i
                    if color_mode == "instance"
                    else None,
                    True,
                )
            )
    annotator.save(filename)

def predictValimage(valCoco, score_threshold=0.5, model=None, modelDir=None):
    device = torch.device('cpu')
    outputImgDir = os.path.join(modelDir, "val", "Predict")
    outputTxtDir = os.path.join(modelDir, "val", "txt")
    PublicFunction.mkDir(outputImgDir)
    PublicFunction.mkDir(outputTxtDir)
    # if "model_weight" in kwargs:
    model.eval()
    model.to(device)

    cocodata = coco.COCO(valCoco)
    catIds = cocodata.getCatIds(catNms=['DiKengYuan'])  # 根据类型获取类型id
    imgIds = cocodata.getImgIds(catIds=catIds)
    for index in range(len(imgIds)):
        imgCoco = cocodata.loadImgs(imgIds[index])[0]
        imageName = imgCoco['file_name']

        annIds = cocodata.getAnnIds(imgIds=imgCoco['id'], catIds=catIds, iscrowd=None)
        anns = cocodata.loadAnns(annIds)
        resultRectList = []
        for ann in anns:
            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            resultRectList.append([bbox_x, bbox_y] + [bbox_x + bbox_w, bbox_y + bbox_h])
        imgRaw = Image.open(imageName)
        img = F.pil_to_tensor(imgRaw.convert("RGB"))
        img = F.convert_image_dtype(img, torch.float)
        img = img.to(device)
        output = model([img])[0]
        boxes = output['boxes'].data.cpu()  # .numpy()
        # pred_boxes = pred["boxes"].long()
        scores = output['scores'].data.cpu()  # .numpy()
        labels = output['labels'].data.cpu()  # .numpy()

        mask = scores >= score_threshold
        boxes = boxes[mask]  # .astype(np.int32)
        scores = scores[mask]
        labels = labels[mask]
        labels = [f"Silo-cave: {score:.2f}" for label, score in zip(labels, scores)]
        output_image_filename = os.path.join(outputImgDir, PublicFunction.getFileExtName(imageName)[1] + ".jpg")

        resultRectList = None
        plot(imgRaw, boxes, labels, scores, line_width=None, font_size=None, font="Arial.ttf", color_mode="class",
             filename=output_image_filename, trueboxlist=resultRectList)

        txtName = os.path.join(outputTxtDir, PublicFunction.getFileExtName(imageName)[1] + ".txt")
        with open(txtName, "w") as f:
            # for box, score, class_name in zip(boxes, scores, class_names):
            for box, score in zip(boxes, scores):
                f.write("{}\n".format("_".join([str(i) for i in box.tolist()] + [str(score.cpu().numpy())])))

def predict_image_parrall(imageNameList, imageDir, model_weight, outputDir, score_threshold=0.8):
    device = torch.device('cpu')
    model = torch.load(model_weight)
    model.eval()
    model.to(device)
    n = len(imageNameList)
    b = 300

    existFiles = []
    existCsvFileName = os.path.join(outputDir, "existFiles.csv")
    if PublicFunction.check_existence(existCsvFileName):
        existFiles = pd.read_csv(existCsvFileName)["existFiles"].values.tolist()
    else:
        existFiles = PublicFunction.listFiles(outputDir, ".jpg")
        df = pd.DataFrame(existFiles, columns=["existFiles"])
        df.to_csv(existCsvFileName)
    for k in range(int((n + b) / b)):
        bNameList = []
        tifList = imageNameList[k * b:((k + 1) * b)]
        imageArrayList = []
        imgRawList = []

        for imageName in tifList:
            if imageName.replace(".tif", ".jpg") in existFiles:
                continue
            im_data, im_width, im_height, im_bands, im_geotrans, im_proj = PublicFunction.readTiff(
                os.path.join(imageDir, imageName))

            imgRaw = Image.fromarray(im_data.transpose(1, 2, 0))
            img = F.to_tensor(imgRaw)
            img = F.convert_image_dtype(img, torch.float)
            img = img.to(device)
            imgRawList.append(imgRaw)
            imageArrayList.append(img)
            bNameList.append(imageName)

        if len(imageArrayList) == 0:
            continue
        outputs = model(imageArrayList)
        index = 0
        for output in outputs:
            boxes = output['boxes'].data.cpu()  # .numpy()
            scores = output['scores'].data.cpu()  # .numpy()
            labels = output['labels'].data.cpu()  # .numpy()
            print(scores)
            mask = scores >= score_threshold
            boxes = boxes[mask]  # .astype(np.int32)
            if len(boxes) != 0:
                scores = scores[mask]
                labels = labels[mask]

                txtName = os.path.join(outputDir, PublicFunction.getFileExtName(bNameList[index])[1] + ".txt")
                with open(txtName, "w") as f:
                    for box, score in zip(boxes, scores):
                        f.write("{}\n".format("_".join([str(i) for i in box.tolist()] + [str(score.cpu().numpy())])))
                # if ok is True:
                output_image_filename = os.path.join(outputDir, bNameList[index].replace(".tif", ".jpg"))
                print(output_image_filename)
                plot(imgRawList[index], boxes, labels, scores, line_width=None, font_size=None, font="Arial.ttf",
                     color_mode="class",
                     filename=output_image_filename)
            index = index + 1


if __name__ == '__main__':

    size = 512
    best_Or_last = "last"  # best,last

    # modelNameList = ["SSD", "MobileNet", "Retinanet", "FasterRCNN"]
    # #predict validation
    # valCocoJSON = os.path.join(config.COCODir, r"val512.json")
    # for i in range(0, 5):
    #     timeDict = {}
    #     for modelName in modelNameList:
    #         starttime = time.time()
    #         model_weight = os.path.join(config.ModelDir,modelName, r"{size}_{i}\{modelName}_{bestorlast}.pth".format(i=i, modelName=modelName, size=size, bestorlast=best_Or_last))
    #         model = torch.load(model_weight)
    #         predictValimage(valCocoJSON, score_threshold=0.5, model=model,
    #                         modelDir=PublicFunction.getFileExtName(model_weight)[0])
    #         endtime = time.time()
    #         timeDict[modelName] = [endtime - starttime]
    #     csvpath = os.path.join(config.ModelDir, str(size) + "_predict_time.csv")
    #     dataformat = pd.DataFrame(timeDict)
    #     dataformat.to_csv(csvpath)

    # predict study area
    modelName = "MobileNet"
    imageDir = config.CascadingTifDir
    print(imageDir)
    for i in range(1):
        timeDict = {}
        starttime = time.time()

        predictDir = os.path.join(config.PredictResultDir, r'{modelName}\512_{i}'.format(modelName=modelName, i=i))
        PublicFunction.mkDir(predictDir)

        model_weight = os.path.join(config.ModelDir,r"{modelName}\{size}_{i}\{modelName}_{bestorlast}.pth".format(modelName=modelName, size=512, i=i, bestorlast=best_Or_last))
        print(imageDir)

        fileNameList = PublicFunction.listFiles(imageDir)
        parallelCount = 2 # parall Count
        n = len(fileNameList)
        b = int(math.ceil(n / parallelCount))
        jpgNameList = []
        for k in range(parallelCount):
            tifList = fileNameList[k * b:((k + 1) * b - 1)]
            jpgNameList.append([tifList, imageDir, model_weight, predictDir])

        resultsList = parallComputer.parrallCompute(predict_image_parrall, jpgNameList)
    #     # predict_image_parrall(fileNameList,imageDir,model_weight,predictDir)

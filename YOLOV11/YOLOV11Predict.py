#!/usr/bin/env python
# _*_ coding: utf-8 _*_
from ultralytics import YOLO
import time
import pandas as pd
from PIL import Image
import cv2
import os
import skimage as ski

print(ski.__version__)
from pycocotools import coco
from skimage.io import imread
from matplotlib import pyplot as plt

from Utility import PublicFunction
import config


def plotBox(jsonFile, imgDir, outputDir):
    cocodata = coco.COCO(jsonFile)
    catidlist = cocodata.getCatIds()
    catcls = cocodata.loadCats()
    img_id_list = cocodata.getImgIds()
    catIds = cocodata.getCatIds(catNms=['DiKengYuan'])
    imgIds = cocodata.getImgIds(catIds=catIds)
    for index in range(len(imgIds)):
        img = cocodata.loadImgs(imgIds[index])[0]
        print(img)
        outputFileName = os.path.join(outputDir, PublicFunction.getFileExtName(img['file_name'])[1] + ".jpg")
        imgName = os.path.join(imgDir, PublicFunction.getFileExtName(img['file_name'])[1] + '.jpg')
        if not PublicFunction.check_existence(imgName):
            continue
        i = imread(imgName)
        # i = cv2.imread(os.path.join(r'E:\datasets\coco\val2017', img['file_name']))
        plt.imshow(i)
        plt.axis('off')
        annIds = cocodata.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        print(annIds)
        anns = cocodata.loadAnns(annIds)
        cocodata.showAnns(anns)
        plt.savefig(outputFileName)
        plt.clf()
        # plt.show()


if __name__ == '__main__':

    # predict study area
    best_Or_last = "last"  # best,last
    rootModelDir = os.path.join(config.ModelDir, "YOLOV11", r"{size}_{i}".format(size=512, i=0))
    imageDir = config.CascadingTifDir

    size = 512
    for i in range(1):
        timeDict = {}
        starttime = time.time()

        predictDir = os.path.join(config.PredictResultDir, 'YOLOV11', r'512_{i}'.format(i=i))
        PublicFunction.mkDir(predictDir)

        model_name = os.path.join(config.ModelDir,
                                  r'YOLOV11\{size}_{i}\weights\{best_Or_last}.pt'.format(size=size, i=i,
                                                                                         best_Or_last=best_Or_last))
        model = YOLO(model_name)
        fileNameList = PublicFunction.listFiles(imageDir, ".tif")
        n = len(fileNameList)
        b = 1200
        for k in range(0, int((n + b) / b)):
            bNameList = []
            tifList = fileNameList[k * b:((k + 1) * b)]
            imageArrayList = []
            for imageName in tifList:
                im_data, im_width, im_height, im_bands, im_geotrans, im_proj = PublicFunction.readTiff(
                    os.path.join(imageDir, imageName))
                imgRaw = Image.fromarray(im_data.transpose(1, 2, 0))
                imageArrayList.append(imgRaw)
                bNameList.append(imageName)
            results = model.predict(imageArrayList, save=False, conf=0.5)

            index = 0
            for result in results:
                boxes = result.boxes.xyxy  # 边界框坐标
                scores = result.boxes.conf  # 置信度分数
                if len(boxes) is not 0:
                    txtName = os.path.join(predictDir, PublicFunction.getFileExtName(bNameList[index])[1] + ".txt")
                    ok = False
                    with open(txtName, "w") as f:
                        for box, score in zip(boxes, scores):
                            f.write(
                                "{}\n".format("_".join([str(i) for i in box.tolist()] + [str(score.cpu().numpy())])))
                    # if ok is True:
                    file_path = os.path.join(predictDir, bNameList[index])
                    cv2.imwrite(file_path, result.plot())
                index = index + 1

        endtime = time.time()
        timeDict["YOLO"] = [endtime - starttime]
        csvpath = os.path.join(predictDir, str(size) + "time.csv")
        dataformat = pd.DataFrame(timeDict)
        dataformat.to_csv(csvpath)

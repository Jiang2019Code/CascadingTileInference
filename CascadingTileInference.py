#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import pandas as pd
import math
from Utility import PublicFunction
import numpy as np
import os
import copy
from Utility import PublicFunction
import config


def tile_to_wgs(x, y, z):
    n = 2 ** z * 1.0
    lon = x / n * 360.0 - 180.0  # 经度
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = lat_rad * 180.0 / math.pi
    return lon, lat_deg

def getPosition(c1,r1,z,extend):
    lon, lat = tile_to_wgs(int(c1), int(r1), z=z)
    lon=lon+(extend[0]+extend[2])/2*0.00000134
    lan=lat-(extend[1]+extend[3])/2*0.0000011
    return lon,lan

def getExtendPosition(c1,r1,z,extend):
    # resolution = 0.0000022949
    lon, lat = tile_to_wgs(int(c1), int(r1), z=z)
    lon1=lon+extend[0]*0.00000134
    lan1=lat-extend[1]*0.0000011

    lon2 = lon + extend[2] * 0.00000134
    lan2 = lat - extend[3] * 0.0000011
    return [lon1,lan1,lon2,lan2]



def py_cpu_filter_nms(dets, thresh):
    dets=np.asarray(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)#所有面积
    scores = dets[:, 4]
    print('areas  ', areas)
    print('scores ', scores)

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []
    masks=np.ones(np.shape(dets)[0])
    indexList = np.asarray([i for i in range(np.shape(dets)[0])])

    for index in range(dets.shape[0]):
        if masks[index]==1:

            x11 = np.maximum(x1[index], x1)  # calculate the points of overlap
            y11 = np.maximum(y1[index], y1)
            x22 = np.minimum(x2[index], x2)
            y22 = np.minimum(y2[index], y2)

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h
            print('overlaps is', overlaps)

            ious = overlaps / (areas[index] + areas - overlaps)

            overlaps_other=np.zeros(np.shape(overlaps)[0]-1)
            iou_other=np.zeros(np.shape(ious)[0]-1)
            area_other=np.zeros(np.shape(areas)[0]-1)

            index_other=np.zeros(np.shape(areas)[0]-1,dtype=int)

            overlaps_other[:index]=overlaps[:index]
            overlaps_other[index:]=overlaps[index+1:]

            iou_other[:index] = ious[:index]
            iou_other[index:] = ious[index + 1:]

            area_other[:index] = areas[:index]
            area_other[index:] = areas[index + 1:]

            index_other[:index] = indexList[:index]
            index_other[index:] = indexList[index + 1:]
            iou_other_index=np.where(iou_other > 0)[0]

            if np.shape(iou_other_index)[0] > 0:
                maxID=np.argmax(area_other[iou_other_index])
                if area_other[iou_other_index][maxID]<=areas[index]:
                    for i in index_other[iou_other_index]:
                        print(i)
                        masks[i] = 0
                else:
                    # masks[index]=0
                    for i in index_other[iou_other_index]:
                        masks[i] = 0
                    masks[index_other[iou_other_index][maxID]]=1


    return keep,masks


def filterCasscadePredicts(predictDir,outputDir):
    boxListDict={}

    for fileName in PublicFunction.listFiles(predictDir, ".txt"):
        boxListDict[fileName]=[]
        fileFullName = os.path.join(predictDir, fileName)
        with open(fileFullName, 'r') as f:
            for i in f.readlines():
                iList = [float(j) for j in i.split("_")]
                boxListDict[fileName].append(iList+[1])

    for fileName in PublicFunction.listFiles(predictDir, ".txt"):
        fnameList = PublicFunction.getFileExtName(fileName)[1].split("_")

        r1 = int(fnameList[0])
        c1 = int(fnameList[1])
        f1 = str(r1) + "_" + str(c1) + ".txt"


        f2 = str(r1) + "_" + str(c1-1) + ".txt"#（-256，0）
        f3 = str(r1) + "_" + str(c1+1) + ".txt" # （256，0）
        f4 = str(r1-1) + "_" + str(c1) + ".txt"  # （0，-256）
        f5 = str(r1+1) + "_" + str(c1 ) + ".txt"# （0，256）

        f6 = str(r1 - 1) + "_" + str(c1-1) + ".txt"#（-256，-256）
        f7 = str(r1 + 1) + "_" + str(c1+1) + ".txt"
        f8 = str(r1-1) + "_" + str(c1 + 1) + ".txt"
        f9 = str(r1+1) + "_" + str(c1 - 1) + ".txt"

        coodPadList = [
            [0, 0],  # （-256，0）
            [-256, 0],  # （-256，0）
            [256, 0] ,  # （256，0）
            [0, -256] ,  # （0，-256）
            [0, 256],# （0，256）

            [-256,-256],
            [256, 256],
            [256, -256],
            [-256, 256],
             ]

        fCood=dict(zip([f1,f2,f3,f4,f5,f6,f7,f8,f9],coodPadList))
        dataDict={}
        f1Len=0
        for ii,tName in enumerate([f1,f2,f3,f4,f5,
                                   f6,f7,f8,f9]):
            if not PublicFunction.check_existence(os.path.join(predictDir,tName)):
                continue
            dataDict[tName] = copy.deepcopy(boxListDict[tName])

        lineList = []
        fNameList=[]
        iiList=[]
        for k in dataDict.keys():
            if k==f1:
                f1Len = len(dataDict[k])
                ii = 0
                for i in range(len(dataDict[k])):

                    if dataDict[k][i][-1]==1:
                        lineList.append(dataDict[k][i][:-1])

                        fNameList.append(k)
                        iiList.append(ii)
                    ii=ii+1

            else:
                ii = 0
                for i in range(len(dataDict[k])):
                    if dataDict[k][i][-1]==1:
                        iList = dataDict[k][i]
                        iList[0]=iList[0]+fCood[k][0]
                        iList[1]=iList[1]+fCood[k][1]
                        iList[2] = iList[2] + fCood[k][0]
                        iList[3] = iList[3] + fCood[k][1]
                        lineList.append(iList[:-1])
                        iiList.append(ii)
                        fNameList.append(k)

                    ii = ii + 1

        if len(lineList) <= 1:
            continue

        else:

            keepList, masks = py_cpu_filter_nms(lineList, thresh=0.1)
            print(masks)

            for i in range(len(iiList)):
                boxListDict[fNameList[i]][iiList[i]][-1] = masks[i]

    for fileName in boxListDict.keys():
        resultFileName = os.path.join(outputDir, fileName)
        resultList=[]
        for i in range(len(boxListDict[fileName])):
            if boxListDict[fileName][i][-1]==1:
                resultList.append(boxListDict[fileName][i][:-1])
        if len(resultList)>0:
            with open(resultFileName, "w") as f:
                for result in resultList:
                    rList = []
                    for r in result:

                        rList.append(str(r))

                    f.write("{}\n".format("_".join(rList)))

if __name__ == '__main__':

    modelName = "MobileNet"
    predictDir = os.path.join(config.PredictResultDir, r'{modelName}\512_{i}'.format(modelName=modelName, i=0))
    outputDir= os.path.join(config.CascadingResultDir,"Result")


    PublicFunction.mkDir(outputDir)
    filterCasscadePredicts(predictDir, outputDir)


    shapefileName = os.path.join(config.CascadingResultDir, r"cascade_{modelName}.shp".format(modelName=modelName))
    shapefileNameExtend = os.path.join(config.CascadingResultDir, r'cascade_{modelName}Extend.shp'.format(modelName=modelName))

    lonlanList=[]
    id=0
    gridList = []

    for fileName in PublicFunction.listFiles(outputDir,".txt"):

        print(fileName)
        fnameList=PublicFunction.getFileExtName(fileName)[1].split("_")
        txtName=os.path.join(outputDir,PublicFunction.getFileExtName(fileName)[1]+".txt")

        r1 = fnameList[0]
        c1 = fnameList[1]
        z=20
        lineList=[]
        with open(txtName, 'r') as f:
            for i in f.readlines():
                iList=[float(j) for j in i.split("_")]

                extend=iList[:-1]
                lon,lat=getPosition(c1, r1, z, extend)

                lonlanList.append([lon,lat,PublicFunction.getFileExtName(fileName)[1]])

                grid = [id] + getExtendPosition(c1, r1, z, extend)
                gridList.append(grid)
                id = id + 1



    PublicFunction.createShape(shapefileName,["LON","LAN","File"],lonlanList)
    PublicFunction.createShape(shapefileNameExtend, ["ID"], gridList, "rect")



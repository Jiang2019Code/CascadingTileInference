#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import pandas as pd
import numpy as np
from Utility import PublicFunction
import config
import os


def getPixels(geoCoordsList, geoInfoFileName):
    im_data, im_width, im_height, im_bands, im_geotrans, im_proj = PublicFunction.readTiff(geoInfoFileName)
    pixelsList = []
    for geoCoords in geoCoordsList:
        lon = geoCoords[0]
        lan = geoCoords[1]
        X, Y = PublicFunction.lonlat2geo(im_proj, float(lon), float(lan))  # 经纬度转地理坐标
        x, y = PublicFunction.geo2imagexy(im_geotrans, X, Y)
        pixels = im_data[:, x:x + 1, y:y + 1].astype(float).flatten()[0]
        pixelsList.append([lon, lan, pixels])
    return pd.DataFrame(np.asarray(pixelsList), columns=["LON", "LAN", "Value"])


def exportAspect(shapeFileName, demFileName, outputFileName):
    fieldValuesMapList = PublicFunction.readShape(shapeFileName)
    GeoCoodList = []
    for fieldValuesMap in fieldValuesMapList:
        LON = fieldValuesMap["LON"]
        LAN = fieldValuesMap["LAN"]
        GeoCoodList.append([LON, LAN])

    if len(GeoCoodList) > 0:
        pixelsDf = getPixels(GeoCoodList, demFileName)
        pixelsDf.replace(9999, np.NaN, inplace=True)
        pixelsDf.dropna(axis=0, how='any', inplace=True)
        pixelsDf.to_csv(outputFileName)


if __name__ == '__main__':
    # MobileNet result
    modelName = "MobileNet"
    # DEM
    shapeFileName = os.path.join(config.CascadingResultDir, "cascade_"+modelName + ".shp")
    demFileName = config.DEMFileName
    outputFileName = os.path.join(config.CascadingResultDir, modelName + "_DEM.csv")
    exportAspect(shapeFileName, demFileName, outputFileName)

    # Aspect
    shapeFileName = os.path.join(config.CascadingResultDir, "cascade_"+modelName + ".shp")
    aspectFileName = config.AspectFileName

    outputFileName = os.path.join(config.CascadingResultDir, modelName + "_Aspect.csv")
    exportAspect(shapeFileName, aspectFileName, outputFileName)

    # Slope
    shapeFileName = os.path.join(config.CascadingResultDir, "cascade_"+modelName + ".shp")
    slopeFileName = config.SlopeFileName
    outputFileName = os.path.join(config.CascadingResultDir, modelName + "_Slope.csv")
    exportAspect(shapeFileName, slopeFileName, outputFileName)

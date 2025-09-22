import os.path

level = 20

rootDir = os.path.dirname(os.path.abspath(__file__))
print(rootDir)
# data collection
# # #download directory from google earth
#
# downloadTileDir=r"GoogleEarthData\Silo_cave{level}".format(level=level)
#
# # # mosaic tif directory
# #for annotation and train
# mosaicTifDir=r"GoogleEarthData\Silo_cave{level}Mosaic".format(level=level)
# mosaicShpDir=r"GoogleEarthData\Silo_cave{level}Mosaic".format(level=level)

# For cascading tile inference
CascadingTifDir = os.path.join(rootDir, r"DatasetData\Examples\DiKeng20_34Example")

# DatasetData directory
datasetDir = os.path.join(rootDir, r"DatasetData")
YOLODir = os.path.join(rootDir, r"DatasetData\YOLO")
COCODir = os.path.join(rootDir, r"DatasetData\COCO")

# Model directory
ModelDir = os.path.join(rootDir, r"Model")
PredictResultDir = os.path.join(rootDir, r"PredictResult")
CascadingResultDir = os.path.join(rootDir, r"CascadingResult")

# DEM file
DEMFileName = os.path.join(rootDir, r"DatasetData\Examples\DEM.tif")
AspectFileName = os.path.join(rootDir, r"DatasetData\Examples\Aspect.tif")
SlopeFileName = os.path.join(rootDir, r"DatasetData\Examples\Slope.tif")

#JPG directory
JPGDir=os.path.join(rootDir,"DatasetData\JPG")

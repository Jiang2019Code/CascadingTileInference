# CascadingTileInferenceAlgorithm
## Requirements
Python 3.10  
Pytorch 2.4.1+cu118
GDAL 3.7 Download URL (https://github.com/cgohlke/geospatial-wheels/releases)  
The easiest way to install this code is to create a Python virtual environment and to install dependencies using: pip install -r requirements.txt

## Dataset
Download URL (https://github.com/cgohlke/geospatial-wheels/releases)  
Copy the dataset to the CascadingTileInference project directory or modify the config.py file in the code to change the specified directory. 
1. The silo-cave training dataset is used for model training and includes two data formats, coco and yolo, for various model training.  
**COCO** directory:  
**train512.json** - training dataset.  
**val512.json** - validation dataset.  
**YOLO** directory:  
**images** - image data directory.  
**labels** - labeled data directory.  
2. Examples directory for model reasoning and testing.    
**DiKeng20_34Example** directory- Test Data.  
**DEM.tif** - Elevation Data.  
**Aspect.tif** - Aspect Data.  
**Slope.tif** - Slope Data.  
**DiKeng20_34.tif** - Stitched TIFF Image.  
**DiKeng20_34_Extend.shp** - Labeled Shapefile Data.  
3. **PredictResult** directory - stores model prediction results.  
4. **CascadingResult** directory - Cascading Tile Inference Algorithm execution results.  
5. **Model** directory - training model results and accuracy evaluation results.  

## Source code
**config.py** module: used for data and model training directory configuration.  
**torchutil** module: provides basic model data loading, model training, and accuracy evaluation functionality.  
**TorchVisionObjectDetection** module: provides model training, testing, and inference functionality based on the TorchVision module.  
**YOLOV11** module: provides YOLO model training, testing, and inference functionality.  
**Utility** module: provides basic file reading and writing, as well as TIFF and shapefile reading and writing functionality.  
**CascadingTileInference.py** module: provides CascadingTileInferenceAlgorithm inference functionality.  
**SilocaveDEMDistribution.py** module: provides elevation, slope, and aspect statistics.  











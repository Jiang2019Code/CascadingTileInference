[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19248725.svg)](https://doi.org/10.5281/zenodo.19248725)

## Python scripts for the manuscript
## Jiang, D.; Zhang, L.; Chen, P.; Luo, R.; Wang, Z.; Lu, P. Intelligent identification of silo-cave traditional residences in the Chinese Loess Plateau based on deep learning. Journal of Archaeological Science 2026, 192, 106625, doi:https://doi.org/10.1016/j.jas.2026.106625
## The training data must be downloaded from the Zenodo website(https://doi.org/10.5281/zenodo.19248725).
## The repository is organised into the following main directories:

1. **Data directory**
- **GoogleEarthData** directory:  The silo-cave training dataset is used for model training and includes two data formats, coco and yolo, for various model training.
- **Examples** directory for model reasoning and testing.    
    * DiKeng20_34Example directory- Test Data.  
    * DEM_XXXX.tif - Elevation Data.  
    * Aspect_XXXX.tif - Aspect Data.  
    * Slope_XXXX.tif - Slope Data.  
    * DiKeng20_34.tif - TIFF Image.  
    * DiKeng20_34_Extend.shp - Labeled Shapefile Data.  
- **PredictResult**directory - stores model prediction results.
- **CascadingResult** directory - Cascading Tile Inference Algorithm execution results.  
- **JPG** directory - Distribution Statistics of DEM, Aspect and Slope.
- **Results** directory -Silocaves Extraction Results.
2. **Models** directory - training model results and accuracy evaluation results.
3. **Source** directory - Source code directory.
   - **config.py** module: used for data and model training directory configuration.  
   - **torchutil** module: provides basic model data loading, model training, and accuracy evaluation functionality.  
   - **TorchVisionObjectDetection** module: provides model training, testing, and inference functionality based on the TorchVision module.  
   - **YOLOV11ObjectDetection** module: provides YOLO11 model training, testing, and inference functionality.  
   - **Utility** module: provides basic file reading and writing, as well as TIFF and shapefile reading and writing functionality.  
   - **CascadingTileInference.py** module: provides CascadingTileInferenceAlgorithm inference functionality(Figure 7).  
   - **SilocaveDEMDistribution.py** module: Generate accuracy result table(Figure 11). 
   - **statsAccurate.py**module: provides elevation, slope, and aspect statistics (Table2 and Table3).
   - **whl** directory: provides Python 3.11 - GDAL 3.7.2 whl installation files. 
   - **Dockerfile** Build docker image.


## Development/Runtime Environment Configuration
**Environment Preparation**  
Before starting the configuration, we need to ensure that the system meets the following basic requirements:
- Operating System: Windows, Linux
- Python Version: 3.6 or higher
- GPU: NVIDIA graphics card (CUDA-enabled)
- NVIDIA Driver: Ensure the latest version of the NVIDIA driver is installed

**Check GPU and Driver**  
Open the terminal or command prompt and enter the following command to view GPU information.  
**Run on the terminal**:  
```nvidia-smi```  
Project test GPU information: NVIDIA GeForce RTX 3080, Driver Version: 537.13, CUDA Version: 12.2.  
If detailed GPU information is displayed, the driver is installed correctly.  
If no GPU information is displayed, please go to the NVIDIA official website to download and install the latest driver.
### 1.**Windows Development Environment Configuration**
**Final Version Selection: python3.11.4、CUDA11.8、Pytorch2.4.1 GDAL3.7.2**  
Note:Installing the GPU version of PyTorch will install cuDNN 9 by default.  

**(1) Install Python**  
Python Version Reference: https://pytorch.org/get-started/locally/  
Install python 3.11.4：https://www.python.org/downloads/  

**(2) Install CUDA**  
CUDA is a parallel computing platform and programming model developed by NVIDIA.  
CUDA Version Reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html  
Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows   
After installation, you can check the CUDA version with the following command.  
**Run on the terminal**:  
```nvcc --version```  

**(3) Install pytorch**    
PyTorch Version Selection Reference: https://pytorch.org/get-started/previous-versions/  
Note: PyTorch on Windows only supports Python 3.10-3.14; Python 2.x is not supported.  
**Run on the terminal**:  
```pip3.11 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118```  
Here, cu118 denotes CUDA 11.8. Please adjust accordingly if you are using a different CUDA version.  

After installation, you can verify that PyTorch is installed successfully and supports GPU using the following code:  
```import torch```  
```print(torch.version)```  
```print(torch.cuda.is_available())```  
If the output is True, it indicates that PyTorch has successfully detected the GPU.  

Frequently Questions and Solutions:  
**Note:Most issues are due to version incompatibility!**

- GPU unavailable  
If ```torch.cuda.is_available()``` returns False, it may be caused by the following reasons:
Driver issue: Ensure that the NVIDIA driver is installed correctly.  
CUDA installation issue: Check whether CUDA is installed properly and whether the environment variables are configured correctly.  
PyTorch version issue: Ensure that the installed PyTorch version supports the current CUDA version.    
- CUDA version incompatibility  
Check the PyTorch official documentation to confirm the supported CUDA versions for the current PyTorch release.  
If the CUDA version is too high or too low, uninstall the current CUDA and install a version compatible with PyTorch.
If you do not want to reinstall CUDA, you can try installing a PyTorch package that supports a different CUDA version.  

**(4) Install GDAL**  
GDAL (Geospatial Data Abstraction Library) is an open-source,   
cross-platform geospatial data processing library that uniformly supports reading, writing and conversion of raster and vector data,   
and is widely used in remote sensing and GIS fields.  
GDAL Version:3.7.2  
Download URL:https://github.com/cgohlke/geospatial-wheels/releases  
Alternatively, install the precompiled wheel package from the Source/whl directory.   
**Run on the terminal:**  
```pip3.11 install GDAL-3.7.2-cp311-cp311-win_amd64```  
Note: Compatibility between GDAL version and Python version.  
**(4) Install YOLOV11**  
YOLOV11 Version:8.3.133  
**Run on the terminal:**  
```pip3.11 install ultralytics==8.3.133 --no-deps ```  
```pip3.11 install --no-cache-dir pyyaml tqdm pillow psutil py-cpuinfo==9.0.0 requests==2.31.0 seaborn==0.13.2 ultralytics-thop==2.0.14```  
Note: Omitting the ```--no-deps``` parameter will install default dependencies.   
Ultralytics will install the latest PyTorch version, causing version compatibility issues.  
**(5) Install other Python dependency packages**  
**Run on the terminal:**  
```pip3.11 install  --no-cache-dir pandas==2.0.3 pycocotools==2.0.7 matplotlib==3.7.2 opencv-python==4.8.0.76 scikit-learn==1.3.0  numpy==1.26.3```  

**(6) Execute Python code**  
**Run on the terminal:**  
```python Source/YOLOV11ObjectDetection/YOLOV11Train.py```  
```python Source/YOLOV11ObjectDetection/YOLOV11Test.py```  
```python Source/YOLOV11ObjectDetection/YOLOV11Predict.py```  

```python Source/TorchVisionObjectDetection/ObjectDetectionTrain.py```  
```python Source/TorchVisionObjectDetection/ObjectDetectionTest.py```  
```python Source/TorchVisionObjectDetection/ObjectDetectionPredict.py```  
```python Source/TorchVisionObjectDetection/Yolo2CocoDatasetProcess.py```

```python Source/CascadingTileInference.py```  
```python Source/SilocaveDEMDistribution.py```  
```python Source/statsAccurate.py```  

### 2.**Dockerfile Configuration for Ubuntu Environment on Windows**
**(1) WSL2 Configuration** 
- Check System Version  
Windows 10 21H2 or later / Windows 11 (Home / Pro / Enterprise editions are all supported)  
CPU virtualization must be enabled  
- Enable WSL 2
Open PowerShell as administrator:  
**Run on the terminal:**  
```wsl --install```  
```wsl --set-default-version 2``` 
- Enable Full Hyper-V (including Manager, Virtual Switch, etc.)  
**Run on the terminal:**  
```dism.exe /online /enable-feature /featurename:Microsoft-Hyper-V /all /norestart```  
**Run on the terminal:**  
```dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart```  
```dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart```  
Restart-Computer  

**(2) Install Docker Desktop on Windows**  
Download: Docker Desktop for Windows (select the corresponding architecture: x64/ARM64)  
Docker Desktop Installer.exe  
Installation Options:  
Check **Use WSL 2 instead of Hyper-V** (highly recommended)  
Check **Install required Windows components for WSL 2**   
Complete the installation and restart your computer.  

**(3) Build Image from Dockerfile**
**Run on the terminal:**  
Execute the command in the project directory:  
```docker build -t silocaves-ubuntu```  

**(4) Create a Container from the Built Image**  
Mount the Current Project Directory to the /workspace Directory  
**Run on the terminal:**  
```docker run -it --gpus all -v ${PWD}:/workspace --shm-size=16g silocaves-ubuntu```  

**(5) Execute Python code**  
**Run on the terminal:**  
```python Source/YOLOV11ObjectDetection/YOLOV11Train.py```  
```python Source/YOLOV11ObjectDetection/YOLOV11Test.py```  
```python Source/YOLOV11ObjectDetection/YOLOV11Predict.py```  

```python Source/TorchVisionObjectDetection/ObjectDetectionTrain.py```  
```python Source/TorchVisionObjectDetection/ObjectDetectionTest.py```  
```python Source/TorchVisionObjectDetection/ObjectDetectionPredict.py```  
```python Source/TorchVisionObjectDetection/Yolo2CocoDatasetProcess.py```

```python Source/CascadingTileInference.py```  
```python Source/SilocaveDEMDistribution.py```  
```python Source/statsAccurate.py```

## Licence
CC-BY 3.0

## Contact Us
jiangdeyang1004@163.com






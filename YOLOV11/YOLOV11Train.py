#!/usr/bin/env python
# _*_ coding: utf-8 _*_
from ultralytics import YOLO
import time
import os
import pandas as pd
import config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':

    pre_model_name = 'yolo11n.pt'
    # Load a mode1
    model = YOLO(pre_model_name,task="detect")  # load a pretrained model (recommended for training)
    size=512
    dataconfig = r"SilocaveDataConfig{size}.yaml".format(size=512)
    timeDict = {}
    for i in range(0,5):
        starttime = time.time()
        results = model.train(data=dataconfig, epochs=1,imgsz=512,batch=20,cfg="default.yaml",
                              save_dir=os.path.join(config.ModelDir,"YOLOV11",r"{size}_{i}".format(size=size,i=i)))#修改库的配置文件

        endtime = time.time()
        timeDict["YOLO"] = [endtime - starttime]
        csvpath = os.path.join(config.ModelDir,r"YOLOV11",r"{size}_{i}".format(size=size,i=i), str(size) + "time.csv")

        dataformat = pd.DataFrame(timeDict)
        dataformat.to_csv(csvpath)







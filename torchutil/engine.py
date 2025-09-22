#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os.path
import time
import torch
import torchvision.models.detection.mask_rcnn
from . import utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import functional as F
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def train_one_epoch(model, optimizer, train_data_loader, test_data_loader,device, epoch, print_freq, scaler=None):
    # model.cuda()
    # model.eval()
    # model.train_old()
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None

    model_time = time.time()

    for images, targets in metric_logger.log_every(train_data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        # print(images)
        # print(targets)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # print(losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # print(losses_reduced)
        loss_value = losses_reduced.item()

        # if not math.isfinite(loss_value):
        #     print(f"Loss is {loss_value}, stopping training")
        #     print(loss_dict_reduced)
        #     sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()

            optimizer.step()

        # if lr_scheduler is not None:
        #     lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    trainNameList=[]
    trainLossList=[]
    for name, meter in metric_logger.meters.items():
    #     # loss_str.append(f"{name}: {str(meter)}")
         trainNameList.append(name)
         trainLossList.append(meter.value)

    valLossDict=dict(zip(trainNameList,trainLossList))

    valLossDict["time"]=model_time
    metric_logger.update(**valLossDict)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)#真值数据
    model_time_sum=0

    # with torch.no_grad():
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_model_time = time.time()
        outputs = model(images)
        #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - start_model_time

        model_time_sum=model_time_sum+model_time
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}#预测数据
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        metric_logger.update(model_time_sum=model_time_sum)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    #
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator,metric_logger

@torch.inference_mode()
def YOLOEvaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)#真值数据
    model_time_sum=0

    for images, targets in metric_logger.log_every(data_loader, 10, header):

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_model_time = time.time()


        results = model.predict(images, save=False, conf=0.25)  # 同时设置project和save_dir路径



        outputs=[]
        for result in results:

            t={
                "boxes": result.boxes.xyxy,
                "scores":  result.boxes.conf,
                "labels": result.boxes.cls+1,
            }
            outputs.append(t)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]


        model_time = time.time() - start_model_time

        model_time_sum=model_time_sum+model_time
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}#预测数据
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        metric_logger.update(model_time_sum=model_time_sum)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    #
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator,metric_logger

@torch.inference_mode()
def predictTest(model, data_loader, device,score_threshold=0.5,outputDir=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "PredictTest:"

    # coco = get_coco_api_from_dataset(data_loader.dataset)
    # iou_types = _get_iou_types(model)
    model_time_sum=0
    for images, targets in metric_logger.log_every(data_loader, 1, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]


        model_time = time.time() - model_time
        model_time_sum=model_time_sum+model_time
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        res_im = {target["image_id"]: image for target, image in zip(targets, images)}
        targets= {target["image_id"]: target for target in targets}


        for image_id, output in res.items():
            image=res_im[image_id]
            # print(image_id)
            boxes = output['boxes'].data.cpu()#.numpy()
            # pred_boxes = pred["boxes"].long()
            scores = output['scores'].data.cpu()#.numpy()
            labels = output['labels'].data.cpu()#.numpy()

            mask = scores >= score_threshold
            boxes = boxes[mask]#.astype(np.int32)
            scores = scores[mask]
            labels=labels[mask]
            labels = [f"Silo-cave: {score:.2f}" for label, score in zip(labels, scores)]

            target_boxes = targets[image_id]['boxes'].data.cpu()#.numpy()
            # target_labels = targets[image_id]['labels'].data.cpu()#.numpy()

            output_image = draw_bounding_boxes(image, boxes, labels, colors="red",width=2,font_size=25)
            output_image = draw_bounding_boxes(output_image, target_boxes, colors="green",width=2,font_size=20)

            output_image = F.to_pil_image(output_image)
            # image_outputs.append((image_id, boxes, scores))
            output_image_filename=os.path.join(outputDir,str(image_id)+".jpg")
            output_image.save(output_image_filename)



        metric_logger.update(model_time=model_time)
        metric_logger.update(model_time_sum=model_time_sum)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    torch.set_num_threads(n_threads)
    return metric_logger

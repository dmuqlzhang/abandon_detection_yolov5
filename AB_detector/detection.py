import torch
import numpy as np
from pathlib import Path

from models.utils import set_logging,check_img_size,non_max_suppression,scale_coords, select_device, attempt_load
from abandon_config import configs


class Detector:
    def __init__(self):
        # self.weights = 'yolov5s.pt'
        self.weights = configs.weights
        self.imgsz = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        # self.classes = [0, 24, 25, 26, 28, 39, 41, 63, 64, 66, 67, 73]
        self.classes = configs.abandon_list

        # self.coco_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        # 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        # 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        # 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        # 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        # 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        # 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        # 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        # 'hair drier', 'toothbrush']
        # self.all_coco_dict = dict(zip([i for i in range(len(self.coco_list))], self.coco_list))
        self.augment = False
        self.agnostic_nms = False

        # Initialize
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        self.manager=None
        self.tracker=None

    def set_tracker(self, _tracker):
        self.tracker=_tracker

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[0] = (x[0] + x[2]) / 2  # x center
        y[1] = (x[1] + x[3]) / 2  # y center
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
        return y
    def detect(self, _webcam, _path, _img, _im0s,_names, frame_id):
        output=None
        abandon_dict = {}
        for i in self.classes:
            abandon_dict[_names[int(i)]] = int(i)
        _img = torch.from_numpy(_img).to(self.device)
        _img = _img.half() if self.half else _img.float()  # uint8 to fp16/32
        _img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if _img.ndimension() == 3:
            _img = _img.unsqueeze(0)

        # Inference
        pred = self.model(_img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms)

        bbox_xywh = []
        confs = []
        labels = []
        have_goal = False
        for i, det in enumerate(pred):  # detections per image
            if _webcam:  # batch_size >= 1
                p, s, im0 = Path(_path[i]), '%g: ' % i, _im0s[i].copy()
            else:
                p, s, im0 = Path(_path), '', _im0s


            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(_img.shape[2:], det[:, :4], im0.shape).round()
                result_dict = {'frame_id': frame_id, 'image': im0}
                for i in self.classes:
                    result_dict[_names[int(i)]] = []
                for *xyxy, conf, cls in det:
                    # img_h, img_w, _ = im0.shape
                    # x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    # obj = [x_c, y_c, bbox_w, bbox_h]
                    # obj = self.xyxy2xywh(obj)
                    x1, y1, x2, y2 = float(xyxy[0].cpu()), float(xyxy[1].cpu()), float(xyxy[2].cpu()), float(xyxy[
                                                                                                                 3].cpu())
                    # bbox_xywh.append(obj)
                    # confs.append(conf)
                    # labels.append(_names[int(cls)])
                    result = {'axis': [x1, y1, x2, y2],
                              'attribute': {_names[int(cls)]},
                              'conf': float(conf.cpu()),
                              'detect_id': -1,
                              'track_id': -1,
                              'is_deleted': False}
                    result_dict[_names[int(cls)]].append(result)
                    have_goal = True
        best_shot = []
        #print(have_goal, _names)
        #修复无目标时局部变量bug
        if have_goal is False:
            result_dict = {'frame_id': frame_id, 'image': im0}
            for i in self.classes:
                result_dict[_names[int(i)]] = []
        result, best_shot = self.tracker.update(result_dict, best_shot, abandon_dict)
        # if have_goal is False:
        #     print(result)
        # #print(result)
        #该改，返回值
        output = result
        return output


def calculate_distance(_object, _person):
    obj_cen = np.array([_object.location[0].cpu() + _object.location[2].cpu()/2, _object.location[1].cpu() + _object.location[3].cpu()/2])
    per_cen = np.array([_person.location[0].cpu() + _person.location[2].cpu()/2, _person.location[1].cpu() + _person.location[3].cpu()/2])
    distance = np.sqrt(np.sum(np.square(obj_cen - per_cen)))
    return distance


def bbox_rel(image_width, image_height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

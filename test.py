import argparse
import os
import platform
import shutil
import numpy as np
import time
from pathlib import Path
from sort import Sort

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.utils import (
    check_img_size, non_max_suppression, scale_coords,
    xyxy2xywh, plot_one_box, set_logging, attempt_load, select_device, time_synchronized, LoadImages, letterbox)

# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box_video(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    os.makedirs(out, exist_ok=True)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model

    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier

    # Set Dataloader
    vid_path, videoWriter = None, None

    #save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(10000)]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # video parameters
    interval = 20  # 30帧   1s
    fall_threshold = 12
    start_frame = 0
    recorder = {}
    tracker = Sort()

    #inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        if dataset.mode == 'images':
            print('please test in video!')
            break
        elif dataset.mode == 'video':
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if vid_path != save_path:
                vid_path = save_path

                fourcc = 'mp4v'  # output video codec
                if isinstance(videoWriter, cv2.VideoWriter):
                    videoWriter.release()  # release previous video writer
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            for i, det in enumerate(pred):
                ##add sort
                if det is not None and len(det):
                    label_list = ['fall', 'normal']
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    result_dict = {'frame_id': dataset.frame, 'image': im0s, 'person': [], 'fall': []}
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 4 + '%g' + '\n') % (cls, *xywh))  # label format
                        label = label_list[int(cls)]
                        x1, y1, x2, y2 = float(xyxy[0].cpu()), float(xyxy[1].cpu()), float(xyxy[2].cpu()), float(xyxy[
                            3].cpu())
                        result = {'axis': [x1, y1, x2, y2],
                                  'attribute': {label},
                                  'conf': float(conf.cpu()),
                                  'detect_id': -1,
                                  'track_id': -1,
                                  'is_deleted': False}
                        result_dict['person'].append(result)
            best_shot = []
            result, best_shot = tracker.update(result_dict, best_shot)
            #print(result)
            if result is None:
                continue
            else:
                detected_vehicles = result['person']
                # tmp_speed = []
                fall_unfall_list = []  # 后边维护固定长度
                for vehicle in detected_vehicles:
                    track_id = vehicle['track_id']
                    if track_id == -1:
                        continue
                    coordinates = vehicle['axis']
                    if track_id in recorder:
                        for attribute_, frame_id_ in recorder[track_id]:  # 判断interval帧内有多少fall
                            if 'person' in list(vehicle['attribute']):
                                if len(fall_unfall_list) == interval:
                                    fall_unfall_list.pop(0)
                                    fall_unfall_list.append('person')
                                else:
                                    fall_unfall_list.append('person')
                            elif 'fall' in list(vehicle['attribute']):
                                if len(fall_unfall_list) == interval:
                                    fall_unfall_list.pop(0)
                                    fall_unfall_list.append('fall')
                                else:
                                    fall_unfall_list.append('fall')
                        if len(list(vehicle['attribute'])) == 1 and list(vehicle['attribute'])[0] in ['normal', 'fall']:
                            plot_one_box_video(coordinates, im0s, color=colors[int(track_id)],
                                         label=list(vehicle['attribute'])[0] + ' trackID ' + str(vehicle['track_id']),
                                         line_thickness=None)
                        recorder[track_id].append((vehicle['attribute'], dataset.frame))
                        # if len(recorder[track_id]) > 10:
                        #     recorder[track_id].pop(0)
                    else:
                        recorder[track_id] = [(vehicle['attribute'], dataset.frame)]
            if view_img:
                cv2.imshow(p, im0s)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            if save_img:
                img_vis = np.zeros((h + 60, w, 3), np.uint8)
                img_vis[60:, :, :] = im0s
                if len(fall_unfall_list) < interval or fall_unfall_list.count('fall') < fall_threshold:

                    cv2.putText(img_vis, 'normal', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                elif fall_unfall_list.count('fall') >= fall_threshold:

                    cv2.putText(img_vis, 'fall', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                img_vis = cv2.resize(img_vis, (w, h))
                videoWriter.write(img_vis)
        if save_txt or save_img:
            print('Results saved to %s' % Path(out))
            if platform.system() == 'Darwin' and not opt.update:  # MacOS
                os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/austin/docker_project/model_example/PersonFallDetectionModel/models/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='images/video1.avi', help='source')  # file/folder
    parser.add_argument('--output', type=str, default='images/output', help='output folder')  # output folder; warning, will del the dir before get result
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)
    save_img = True
    # save_txt = True
    with torch.no_grad():
        detect(save_img)

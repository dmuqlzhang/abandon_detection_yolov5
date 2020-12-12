import cv2
import torch
import torch.backends.cudnn as cudnn
from models.utils import LoadImages

from AB_detector.utils import get_roi_contours, compute_iou_shp, plot_one_box, compute_iou_match, count_overtime_trackid
from AB_detector.detection import Detector
from AB_detector.sort import Sort
from numpy import random

import numpy as np



import sys
#sys.path.insert(0, '../..')
from abandon_config import configs

class ABDetector:
    def __init__(self):
        self.detector = Detector()
        #self.manager = Manager()

        self.tracker = Sort()
        #self.tracker.set_manager(self.manager)

        self.detector.set_tracker(self.tracker)

    def process(self, opt):
        output_path = opt.output
        source = opt.source
        webcam = None

        imgsz = self.detector.imgsz
        model = self.detector.model
        device = self.detector.device
        half = self.detector.half


        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        vid_path, vid_writer = None, None
        #
        # 统计遗留物在roi内停留的时间，超出时间一直报警
        frame_counter = {}
        time_counter = {}#注意需要根据跳帧数去计算时间
        track_id_coordinates = {}
        #
        # 用于记录 缓慢移动 or 停止的次数，用于判断
        abandon_status = {}  #move or stop

        for path, img, im0s, vid_cap in dataset:
            ##判断是否有可视化画图
            if configs.has_GUI:
                roi_contours = get_roi_contours(img=im0s)
            else:
                roi_contours = np.load(configs.roi_contours_npy_path)
            if configs.save_as_video:
                 # 画出roi区域框 blue
                cv2.drawContours(im0s, [roi_contours], -1, (255, 0, 0), 2)
            ###获取画图区域最下外接矩形
            x_min, y_min = np.min(roi_contours, axis=0)
            x_max, y_max = np.max(roi_contours, axis=0)
            roi_coordinate = (x_min, y_min, x_max, y_max)
            ##
            #print(webcam, path, img, im0s, names, dataset.frame)
            if dataset.frame % configs.skip_frames == 0:
                pass
            else:
                ##获得跟踪结果
                result_sort = self.detector.detect(webcam, path, img, im0s, names, dataset.frame)

                if result_sort is None:#跟踪为空，则跳过
                    continue
                else:
                    #合并遗留物列表的所有跟踪结果
                    detected_abandon = []
                    for i in configs.abandon_list:
                        detected_abandon.extend(result_sort[names[int(i)]])###获得所有遗留物列表的跟踪结果
                    print(detected_abandon)
                    #对当前帧的所有跟踪结果进行处理，包括1判断是否进入禁止区域（比较与禁止区域的iou阈值） 2.是否处于静止状态（比较与上一帧的iou阈值）
                    for subs_abandon in detected_abandon:
                        coordinates_subs_abandon = subs_abandon['axis']#坐标
                        track_id_subs_abandon = subs_abandon['track_id']#track id
                        x1, y1, x2, y2 = coordinates_subs_abandon[0:4]  # float
                        ##计算与划定区域的iou
                        iou_forbid_aera = compute_iou_shp(roi_contours, (x1, y1, x2, y2), roi_coordinate)  # 这里是否需要int
                        ##判断是否进入禁止遗留物区域
                        if iou_forbid_aera > configs.iou_forbid_area_threshold:#进入禁止遗留物区域
                            if track_id_subs_abandon not in time_counter:
                                # 相减少1
                                time_counter[track_id_subs_abandon] = 1
                            else:
                                time_counter[track_id_subs_abandon] += 1

                            ##比较与历史上一抽帧的iou
                            if track_id_subs_abandon not in track_id_coordinates:  # 首次出现的 不计算 iou匹配
                                track_id_coordinates[track_id_subs_abandon] = (int(x1), int(y1), int(x2), int(y2))
                                # 初始帧需要设置 它的移动类别数量
                                abandon_status[track_id_subs_abandon] = {'move_num': 0, 'stop_num': 0}
                            else:
                                # 计算当前帧 和 前一帧的 iou_match 匹配
                                pre_coordinate = track_id_coordinates[track_id_subs_abandon]
                                track_id_coordinates[track_id_subs_abandon] = (int(x1), int(y1), int(x2), int(y2))
                                iou_match = compute_iou_match(pre_coordinate, (x1, y1, x2, y2))  # 这里是否也需要int？？
                                # 已经进入此区域内，那么就要跟踪，过程可以判断移动-停止
                                # if track_id not in move_class:  # 不应该在这里出现
                                #     move_class[track_id] = {'move_num': 0, 'stop_num': 0}
                                # 具体判断移动-停止在这个过程的次数？？？  还是以最后2次为主？？？
                                print(iou_match)
                                if iou_match >= configs.iou_history_frame_threshold:
                                    abandon_status[track_id_subs_abandon]['stop_num'] += 1
                                else:
                                    abandon_status[track_id_subs_abandon]['move_num'] += 1
                                # print(track_id, move_class[track_id])

                            # 记录track_id初始的frame_id即历史frame_id，用于判断跟丢
                            if track_id_subs_abandon not in frame_counter:
                                frame_counter[track_id_subs_abandon] = dataset.frame
                        #

                        # 判断遗留时间 从时间 转成 帧数量
                        return_abandon_coordinates, move_stop_classes = count_overtime_trackid(dataset.frame, time_counter,
                                                                                       track_id_coordinates,
                                                                                       frame_counter, abandon_status,
                                                                                       configs.abandon_stop_time,
                                                                                       configs.lost_threshold, configs.skip_frames)
                        ###
                        print(return_abandon_coordinates, move_stop_classes)
                        coordinates = subs_abandon['axis']
                        track_id = subs_abandon['track_id']
                        if len(list(subs_abandon['attribute'])) == 1 and list(subs_abandon['attribute'])[0] in [i for ind, i in enumerate(names) if ind in configs.abandon_list]:
                            # plot_one_box(coordinates, im0s, color=colors[int(track_id)], label=list(subs_abandon['attribute'])[0],
                            #          line_thickness=None)
                            if len(return_abandon_coordinates)>0:#判断是否存在遗留物
                                print(return_abandon_coordinates)
                                for ind, abandon_coordinates_i in enumerate(return_abandon_coordinates):
                                    plot_one_box(abandon_coordinates_i, im0s, color=colors[int(track_id)],
                                                 label=f"type: {list(subs_abandon['attribute'])[0]} status:{move_stop_classes[ind].split('-')[0]}",
                                                 line_thickness=None)

                    '''
                    完成
                    1.计算与指定区域出现遗留物的iou（判断是否在指定区域出现疑似遗留物）
                    2.计算与历史帧的iou（判断是否移动，确定疑似遗留物是否为遗留物【静止状态】）
                    '''

                    #cv2.imshow("result", output)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                    if (not webcam) and (result_sort!=''):
                        if vid_path != output_path:
                            vid_path = output_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0s)

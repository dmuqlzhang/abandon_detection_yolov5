class DefaultConfigs(object):
    has_GUI = False
    # 当has_GUI=False时，就需要将roi区域读取进来
    roi_contours_npy_path = '/home/austin/docker_project/model_example/DeskAbandonDetectionModel/images/mouse_abandon.npy'
    save_as_video = True # 画出roi区域框 blue
    abandon_list = [24, 64]#背包 鼠标
    weights = '/home/austin/docker_project/yolov5_0924/yolov5x.pt'
    iou_forbid_area_threshold = 0.3 #与划定区域的iou阈值
    iou_history_frame_threshold = 0.85#与物体历史帧的iou阈值，判断是否移动
    abandon_stop_time = 5 #时间阈值 单位s,对应帧数 time*fps, 抽帧时 时间阈值应该更大，超过时间阈值视为遗留物
    skip_frames = 5 #跳帧检测
    lost_threshold = 5 ## 允许跟丢10帧，超出10帧，则删除track_id，这里的帧是指 检测的帧，当抽帧大时，这里应该设置较小，反之较大


configs = DefaultConfigs()


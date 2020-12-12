'''
包含计算iou、画图等辅助函数
'''
from shapely.geometry import Polygon
import numpy as np
import cv2
import random
# 采用gis中的shp计算重叠面积
def compute_iou_shp(roi_contours, bbox, roi_coordinate):
    '''
    :param roi_contours: 感兴趣区轮廓点 array shape=[N,2] [x,y] int 只有1个
    :param bbox:   检测框 float list
    :param roi_coordinate: roi最小正外接矩形坐标
    :return:
    '''
    # 当坐标框不相交时，完全不需要计算
    x_min, y_min, x_max, y_max = roi_coordinate
    if (bbox[0] >= x_max) or (bbox[1] >= y_max) or (bbox[2] <= x_min) or (bbox[3] <= y_min):
        return 0
    # 此函数bbox为float，包含小数
    # bbox 对角坐标转成顺时针4角坐标
    x1, y1, x2, y2 = bbox
    bbox_ = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])

    poly1 = Polygon(roi_contours)
    poly2 = Polygon(bbox_)

    iou = poly1.intersection(poly2).area / poly2.area
    return iou
# 当有可视化界面时可以调用首帧，划定roi区域框多个点
def get_roi_contours(img):
    roi_contours = []
    # 鼠标事件
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_contours.append([x, y])
            xy = "%d,%d" % (x, y)

            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0) # 按空格键 or 其它键

    cv2.drawContours(img, [np.array(roi_contours, dtype=np.int64)], -1, (0, 255, 0), 1)
    cv2.imshow('image', img)  # 相当于覆盖掉了之前的窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(roi_contours, dtype=np.int64)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
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

# 计算前后帧的同一track_id的iou匹配，当然也可以用GIS的，但是就不用了
def compute_iou_match(pre_coordinate, current_coordinate):
    '''
    :param pre_coordinate:     上一帧的对象坐标
    :param current_coordinate: 当前帧的对象坐标
    :return:   计算 同一track_id的前后帧框的iou匹配程度
    '''
    # (x1,y1,x2,y2) = pre_coordinate
    # (x1_,y1_,x2_,y2_) = current_coordinate
    pre_coordinate = np.array(pre_coordinate)
    current_coordinate = np.array(current_coordinate)
    lt = np.maximum(pre_coordinate[:2], current_coordinate[:2])
    rb = np.minimum(pre_coordinate[2:], current_coordinate[2:])

    # if min_r < max_l or min_d < max_u:
    #     return 0

    area_i = np.prod(rb - lt) * (lt < rb).all()
    area_a = np.prod(pre_coordinate[2:] - pre_coordinate[:2])
    area_b = np.prod(current_coordinate[2:] - current_coordinate[:2])
    return area_i / (area_a + area_b - area_i)

# 统计超过时间的track_id对应的坐标框
def count_overtime_trackid(frame_id, time_counter, track_id_coordinates, frame_counter, move_class, time_thresthold, lost_threshold, skip_frames):
    '''
    :param frame_id:             当前帧id
    :param time_counter:         track_id在roi内时间 dict
    :param track_id_coordinates: track_id最后时刻的坐标，当跟丢时不对应当前帧 dict
    :param frame_counter:        记录track_id初始时刻的frame_id，属于历史记录frame_id，用于判断跟丢 dict
    :param move_class:           用于判断是移动 还是 静止
    :param time_thresthold:      时间阈值  时间s * fps
    :param lost_threshold:       跟丢阈值
    :param skip_frames:          抽帧数量
    :return:                     对于跟丢的，删除track_id; 实时返回超出时间的track_id坐标
    '''
    return_coordinates = []
    move_stop_classes = []  # 与return_coordinates顺序一致

    for (track_id, time_) in list(time_counter.items()):
        if (time_-1)*skip_frames+1 > time_thresthold:
            return_coordinates.append(track_id_coordinates[track_id])
            if move_class[track_id]['move_num'] >= move_class[track_id]['stop_num']:
                move_stop_classes.append('moving-'+str(move_class[track_id]['stop_num'])+'-'+str(move_class[track_id]['move_num'])+'-'+str(track_id))
            else:
                move_stop_classes.append('stopping-'+str(move_class[track_id]['stop_num'])+'-'+str(move_class[track_id]['move_num'])+'-'+str(track_id))
            # print(track_id, move_class[track_id]['stop_num'])

        # 判断跟丢，超出lost_threshold帧就认为跟丢，删除track_id
        if (frame_id-frame_counter[track_id]-(time_-1)*skip_frames)/skip_frames >= lost_threshold:
            # pop是原址操作，对原始变量操作，因此需要将dict转成list for循环变量不可变
            time_counter.pop(track_id)
            frame_counter.pop(track_id)
            track_id_coordinates.pop(track_id)
            move_class.pop(track_id)

    return return_coordinates, move_stop_classes
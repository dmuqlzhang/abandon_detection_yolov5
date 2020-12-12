'''
本地画图函数，作为检测遗留物的遗留物区域
'''

import cv2
import numpy as np
import glob
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
if __name__ == '__main__':
    video_path = r'D:\develop_local\abandoned-objects\video1.avi'
    vid_cap = cv2.VideoCapture(video_path)
    ret, frame = vid_cap.read()
    if frame is not None:
        npy = get_roi_contours(frame)
        np.save("video1_roi.npy", npy)
        #b = np.load("filename.npy")


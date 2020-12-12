"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import cv2
import time
import argparse
from filterpy.kalman import KalmanFilter
from AB_detector.blur_detector import BlurDetector


@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] +
                         w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] +
                         w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, class_name, ori_img, frame_id):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                             0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [
                             0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_name = class_name
        x1, y1, x2, y2 = bbox[0:4]
        x1 = [0 if x1<0 else x1][0]
        y1 = [0 if y1<0 else y1][0]
        self.best_shot = ori_img[int(y1):int(y2), int(x1):int(x2)]
        self.confidence = bbox[4]
        self.frame_id = frame_id
        bd = BlurDetector()
        self.blur_score = bd.get_blurness(self.best_shot)
        snap_gray = cv2.cvtColor(self.best_shot, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(snap_gray, bins=16)
        self.highlight_ratio = hist[15] * 1.0 / ((y2 - y1) * (x2 - x1))
        self.label = bbox[5]
        self.padding = 0

    def update_shot(self, bbox, ori_img, frame_id):
        x1, y1, x2, y2 = bbox[0:4]
        confidence = bbox[4]
        x1 = [0 if x1<0 else x1][0]
        y1 = [0 if y1<0 else y1][0]
        shot = ori_img[int(y1):int(y2), int(x1):int(x2)]
        snap_gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(snap_gray, bins=16)
        highlight_ratio = hist[15] * 1.0 / ((y2 - y1) * (x2 - x1))
        if highlight_ratio < 0.15:
            bd = BlurDetector()
            blur_score = bd.get_blurness(shot)
            if confidence - blur_score > self.confidence - self.blur_score:
                self.confidence = confidence
                self.blur_score = blur_score
                self.best_shot = shot
                self.frame_id = frame_id

    def update(self, bbox, ori_img, frame_id):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.update_shot(bbox, ori_img, frame_id)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty(
            (0, 2), dtype=int), np.arange(
            len(detections)), np.empty(
            (0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if det[5] == trk[4]:
                iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    #matched_indices = np.squeeze(np.dstack(matched_indices))
    matched_indices = np.stack(matched_indices,axis=1)
    #print(matched_indices.shape)
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(
        unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=70, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, results, best_shot, abandon_dict):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # object_dict = {'person': 0, 'backpack': 24}
        object_dict = abandon_dict
        frame_id = results['frame_id']
        ori_img = results['image']
        bbox_xyxyc = []
        class_names = []
        detect_id = 0
        for key in results.keys():
            if key == 'image' or key == 'frame_id':
                continue
            for target in results[key]:
                class_names.append(key)
                cf = target['conf']

                cl = object_dict[key]
                x1, y1, x2, y2 = target['axis']
                xyxyc = np.array([x1, y1, x2, y2, cf, cl])
                bbox_xyxyc.append(xyxyc)
                target['detect_id'] = detect_id
                detect_id += 1
        bbox_xyxyc = np.array(bbox_xyxyc)
        # 检测框按概率从大到小排序
        # indices = np.argsort(-bbox_xyxyc[:, 4])
        # class_names = np.array(class_names)[indices]
        # bbox_xyxyc = bbox_xyxyc[indices]
        if len(bbox_xyxyc) == 0:
            return results, best_shot

        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cls = self.trackers[t].label
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cls, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            bbox_xyxyc, trks)

        if len(matched) > 0:
            for key in results.keys():
                if key == 'image' or key == 'frame_id':
                    continue
                for target in results[key]:
                    detect_id = target['detect_id']
                    track_id = matched[np.where(matched[:, 0] == detect_id)[0], 1]
                    if len(track_id):
                        target['track_id'] = self.trackers[int(track_id)].id

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(bbox_xyxyc[d, :][0], ori_img, frame_id)
            else:
                pos = trk.predict()[0]
                if pos[0] > results['image'].shape[1] or pos[1] > results['image'].shape[0] or pos[2] < 0 \
                        or pos[3] < 0 or trk.padding > 3:
                    continue
                pos[0] = max(pos[0], 0)
                pos[1] = max(pos[1], 0)
                pos[2] = min(pos[2], results['image'].shape[1])
                pos[3] = min(pos[3], results['image'].shape[0])
                result = {'axis': [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])],
                          'attribute': {},
                          'conf': trk.confidence,
                          'detect_id': -1,
                          'track_id': trk.id,
                          'is_deleted': False}
                trk.padding += 1
                results[trk.class_name].append(result)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(bbox_xyxyc[i, :], class_names[i], ori_img, frame_id)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                target = {'frame_id': trk.frame_id,
                          'track_id': trk.id,
                          'best_shot': trk.best_shot,
                          'confidence': trk.confidence,
                          'class_name': trk.class_name,
                          'attribute': {}}
                best_shot.append(target)
                self.trackers.pop(i)
        if len(ret) > 0:
            return results, best_shot
        return None, best_shot


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument(
        '--display',
        dest='display',
        help='Display online tracker output (slow) [False]',
        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    sequences = [
        'PETS09-S2L1',
        'TUD-Campus',
        'TUD-Stadtmitte',
        'ETH-Bahnhof',
        'ETH-Sunnyday',
        'ETH-Pedcross2',
        'KITTI-13',
        'KITTI-17',
        'ADL-Rundle-6',
        'ADL-Rundle-8',
        'Venice-2']
    args = parse_args()
    display = args.display
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if display:
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n'
                  '(https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n'
                  '$ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()

    if not os.path.exists('output'):
        os.makedirs('output')

    for seq in sequences:
        mot_tracker = Sort()  # create instance of the SORT tracker
        seq_dets = np.loadtxt(
            'data/%s/det.txt' %
            seq, delimiter=',')  # load detections
        with open('output/%s.txt' % seq, 'w') as out_file:
            print("Processing %s." % seq)
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if display:
                    ax1 = fig.add_subplot(111, aspect='equal')
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (
                        phase, seq, frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(
                        '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                        (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
                        ax1.set_adjustable('box-forced')

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))
    if display:
        print("Note: to get real runtime results run without the option: --display")

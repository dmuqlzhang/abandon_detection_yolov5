import argparse
from AB_detector.detector import ABDetector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='images/mouse_abandon.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='outputs_mouse.avi', help='source')
    args = parser.parse_args()

    detector = ABDetector()
    detector.process(args)
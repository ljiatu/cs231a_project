from moviepy.editor import VideoFileClip

from car_detection import process_car_detection
from lane_detection import LaneDetector
from model import get_model


def run_detection(frame, lane_detector, clf, X_scaler):
    lane_detected = lane_detector.process_frame(frame)
    # car_detected = process_car_detection(frame, clf, X_scaler)
    return lane_detected


def main():
    lane_detector = LaneDetector()
    clf, X_scaler = None, None

    clip = VideoFileClip('challenge.mp4', audio=False)
    result = clip.fl_image(lambda frame: run_detection(frame, lane_detector, clf, X_scaler))
    result.write_videofile('output.mp4')


if __name__ == '__main__':
    main()

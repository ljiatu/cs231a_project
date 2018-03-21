from moviepy.editor import VideoFileClip

from car_detection import CarDetector
from lane_detection import LaneDetector
from model import get_svm_model, get_cnn_model


def main():
    svm_clf, X_scaler = get_svm_model()
    cnn_clf = get_cnn_model()

    lane_detector = LaneDetector()
    car_detector = CarDetector(svm_clf, cnn_clf, X_scaler)

    clip = VideoFileClip('input.mp4', audio=False)
    result = clip.fl_image(lambda frame: run_detection(frame, lane_detector, car_detector))
    result.write_videofile('output.mp4')


def run_detection(frame, lane_detector, car_detector):
    frame = lane_detector.process_frame(frame)
    return car_detector.process_frame(frame)


if __name__ == '__main__':
    main()

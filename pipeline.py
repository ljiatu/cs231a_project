from moviepy.editor import VideoFileClip

from car_detection import process_car_detection
from lane_detection import process_frame
from model import get_model


def run_detection(frame, clf, X_scaler):
    # lane_detected = process_frame(frame)
    car_detected = process_car_detection(frame, clf, X_scaler)
    return car_detected


def main():
    clf, X_scaler = get_model()

    clip = VideoFileClip('extra.mp4', audio=False)
    result = clip.fl_image(lambda frame: run_detection(frame, clf, X_scaler))
    result.write_videofile('output.mp4')


if __name__ == '__main__':
    main()

from moviepy.editor import VideoFileClip
import numpy as np
import ast
import cv2
import os


def main():
    
    # reading the predicted boundary boxes
    bboxes_file = open('bboxes.txt', 'r')
    nms_bboxes = []
    for line in bboxes_file:
        if line[0] == 'F':
            continue
        box = ast.literal_eval(line)
        nms_bboxes.append(box)
    
    # reading the ground truth boundary boxes
    if not os.path.exists('ground_truth.txt'):
        print "Need ground truth lables"
        return
    else:
        gt_file = open('ground_truth.txt','r')
        gt_bboxes = []
        for line in gt_file:
            if not line[0] == '[':
                continue
            box = ast.literal_eval(line)
            gt_bboxes.append(box)
    
    iou_accuracy = []    
    bin_accuracy = []
    for predict_box, gt_box in zip(nms_bboxes, gt_bboxes):
        iou = intersection_over_union(predict_box, gt_box)
        if len(iou) > 0:
            iou_accuracy.append(sum(iou)/len(iou))
            bin_accuracy.append(np.count_nonzero(iou)/float(len(iou)))
            
    print "IOU accuracy: " + str(sum(iou_accuracy)/len(iou_accuracy))
    print "Binary accuracy: " + str(sum(bin_accuracy)/len(bin_accuracy))

    clip = VideoFileClip('input.mp4', audio=False)
    result = clip.fl_image(lambda frame: draw_boxes(frame, nms_bboxes))
    result.write_videofile('output.mp4')
    
    clip = VideoFileClip('input.mp4', audio=False)
    result = clip.fl_image(lambda frame: draw_boxes(frame, gt_bboxes))
    result.write_videofile('ground_truth.mp4')


def draw_boxes(frame, bboxes):
    """
    Draws windows or bounding boxes on the image

    :param frame: a video frame
    :param bboxes: bounding boxes
    :returns: the frame with bounding boxes drawn on it
    """
    for bbox in bboxes[0]:
        cv2.rectangle(frame, bbox[0], bbox[1], (255,0,0), thickness = 4)
    bboxes.pop(0)
    return frame

def intersection_over_union(predict_box, gt_box):
    
    iou = []
    
    for boxA in predict_box:
        max_iou = 0
        for boxB in gt_box:
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0][0], boxB[0][0])
            yA = max(boxA[0][1], boxB[0][1])
            xB = min(boxA[1][0], boxB[1][0])
            yB = min(boxA[1][1], boxB[1][1])
            
            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)
            
            # compute the area of both the prediction and ground-truth rectangles
            boxA_Area = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
            boxB_Area = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
            
            # compute the intersection over union
            temp_iou = interArea / float(boxA_Area + boxB_Area - interArea)
            if temp_iou > max_iou:
                max_iou = temp_iou
        iou.append(max_iou)
    
    if len(gt_box) > len(predict_box):
        for i in xrange(len(gt_box) - len(predict_box)):
            iou.append(0)
    elif len(gt_box) < len(predict_box):
        iou = sorted(iou)[-len(gt_box):]
            
    # return the intersection over union value
    return iou

if __name__ == '__main__':
    main()

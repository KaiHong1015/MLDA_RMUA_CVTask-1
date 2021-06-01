import gc
import torch
import numpy as np 
import cv2
import torchvision

from ensemble_boxes import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label2color = {
    1: (30,144,255),
    2: (0,255,255),
    3: (255,69,0),
    4: (240,128,128),
}

label2str = {
    1: 'Ally Robot',
    2: 'Ally Armor',
    3: 'Enemy Robot',
    4: 'Enemy Armor',
}

def get_model(weight_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model = model.to(device)
    model.eval()
    
    del weight
    gc.collect()
    return model

def get_distance(box, label, dm, isleft):
    '''
    Calculate the distance of the bounding box predicted from death map
    
    Input:
    box: list or numpy -> a single predicted box
    label: list or numpy -> a single predicted label
    isleft: boolean -> Is it the left side camera?
    
    Return:
    d: float -> The calculated distance
    '''
    if isleft:
        c = 50   
    if not isleft:
        if label == 1 or label == 3:
            c = round(175/366 * (box[3]-box[1]))
        else:
            c = round(175/45 * (box[3]-box[1]))
        
    xmid = int(round(box[0] + (box[2] - box[0])/2 + c))
    ymid = int(round(box[1] + (box[3] - box[1])/2))
    
    d = dm[ymid-10:ymid, xmid:xmid+17][dm[ymid-10:ymid, xmid:xmid+17]!=0]
    return 0.0 if not len(d) else d.mean()

def draw_info(np_image, dm, boxes, labels, isleft, fps=None):
    '''
    Draw predicted bounding box and distance on the image
    
    Input:
    np_image: numpy -> an image in the numpy form
    boxes: numpy or list -> Predicted bounding boxes of a single image
    labels: numpy or list -> Predicted labels of a single image
    isleft: boolean -> Is it the left side camera?
    
    Return:
    np_image: numpy -> An image with info on it
    '''
    np_image = np_image.copy()
    
    for box, label in zip(boxes, labels):
        color = label2color[label]
        name = label2str[label]
        distance = get_distance(box, label, dm, isleft)
        text = name + f'({distance/1000:.3f}m)'
        
        cv2.rectangle(np_image, (box[0], box[1]), (box[2], box[3]), color, 3)
        cv2.rectangle(np_image, (box[0], box[1] - 30), (box[0] + round(len(text)/19 * 230), box[1]), color, -1)
        cv2.putText(np_image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.25, (0,0,0), 2)
    if fps:
        cv2.putText(np_image, f'FPS: {fps:.2f}', (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (225,0,0), 2)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return np_image

def get_valid_prediction(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
    '''
    Get predictions which satisfied the conditions. We will choose the boxes with confidence>0.75, and set the
    robot's box as a reference, then calculate the ymin threshold for armor's boxes, as all armor should be at the bottom half of 
    the robot's box
    
    Input:
    boxes: numpy -> Predicted boxes on an single image
    scores: numpy -> Predicted scores on an single image
    labels: numpy -> Predicted labels on an single image
    
    Return:
    valid_boxes: list
    valid_scores: list
    valid_labels: list
    '''
    
    robot_boxes = boxes[np.logical_or(labels==1, labels==3)]
    robot_labels = labels[np.logical_or(labels==1, labels==3)]
    robot_scores = scores[np.logical_or(labels==1, labels==3)]
    
    armor_boxes = boxes[np.logical_or(labels==2, labels==4)]
    armor_labels = labels[np.logical_or(labels==2, labels==4)]
    armor_scores = scores[np.logical_or(labels==2, labels==4)]
    
    if len(robot_boxes) > 0:
        ymin_ref = robot_boxes[:, 1].mean()
        ymax_ref = robot_boxes[:, 3].mean()

        armor_ymin_threshold = ymin_ref + (ymax_ref - ymin_ref) / 2

        armor_labels = armor_labels[armor_boxes[:, 1] >= armor_ymin_threshold]
        armor_scores = armor_scores[armor_boxes[:, 1] >= armor_ymin_threshold]
        armor_boxes = armor_boxes[armor_boxes[:, 1] >= armor_ymin_threshold]

    valid_boxes = []
    valid_labels = []
    valid_scores = []

    for box, score, label in zip(robot_boxes, robot_scores, robot_labels):
        valid_boxes.append(box)
        valid_scores.append(score)
        valid_labels.append(label)

    for box, score, label in zip(armor_boxes, armor_scores, armor_labels):
        valid_boxes.append(box)
        valid_scores.append(score)
        valid_labels.append(label)
    
    return valid_boxes, valid_scores, valid_labels

@torch.no_grad()
def make_prediction(model, image):
    '''
    Make prediction on a single image
    
    Input:
    image: tensor with shape (channels, height, width) #without batch_size!!!!
    
    Return:
    prediction: dictionary which contain (boxes, scores, labels)
    np_image: a numpy image of the input image
    '''
    np_image = np.ascontiguousarray(image.permute(1,2,0).mul(255).byte().cpu())
    prediction = model(image.unsqueeze(0).to(device))
    return prediction, np_image

def run_wbf(prediction, image_max_size, weights=None, iou_thr=0.55, skip_box_thr=0.75):
    boxes = [((prediction[0]['boxes'].clip(min=0, max=image_max_size - 1)/(image_max_size - 1))).tolist()]
    scores = [prediction[0]['scores'].tolist()]
    labels = [prediction[0]['labels'].tolist()]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = ((boxes * (image_max_size - 1)).clip(min=0, max=image_max_size-1)).astype(np.int32)

    return boxes, scores, labels



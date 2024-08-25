import argparse
import cv2
import numpy as np
import os
import torch
import time

from basenet.model import Model_factory
from loader import ListAppleDataset
from utils.post_processing import get_center_point_contour
from utils.post_processing import nms, topk, smoothing
from utils.util import APPLE_CLASSES, COLORS
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', 1)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help='Root directory of dataset')
parser.add_argument('--dataset', type=str, help='Dataset version name')
parser.add_argument('--checkpoint', type=str, help='Tranined checkpoint')
parser.add_argument('--experiment', default=1, type=int, help='number of experiment')
parser.add_argument('--input_size', default=512, type=int, help='input size')
parser.add_argument('--c_thresh', default=0.4, type=float, help='threshold for center point')
parser.add_argument('--backbone', type=str, default='gaussnet_cascade', 
                        help='[gaussnet_cascade, gaussnet]')
parser.add_argument('--kernel', default=3, type=int, help='kernel of max-pooling for center point')
parser.add_argument('--scale', default=1.3, type=float, help='scale factor')

arg = parser.parse_args()
print(arg)


result_img_path = 'img-out-240814/'
if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)
    
"""Data Loader"""
mean = (0.485, 0.456, 0.406)
var = (0.229, 0.224, 0.225)

test_dataset = ListAppleDataset('valid', arg.dataset, arg.root,
                            arg.input_size, transform=None, evaluation=True)
   
"""Network Backbone"""
NUM_CLASSES = {'version-1' : 2, 'version-2': 3, 'version-3': 2, 'split': 2}
num_classes = NUM_CLASSES[arg.dataset]
model = Model_factory(arg.backbone, num_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.cuda()

checkpoint = torch.load(arg.checkpoint)

model.load_state_dict(checkpoint['model'])
checkpoint = None

_ = model.eval()

dest_dir = f'/data/apple/results/{arg.checkpoint}_{arg.experiment}/Task1/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

filedict = {}
for cls in APPLE_CLASSES:
    fd = open(os.path.join(dest_dir, f'Task1_{cls}.txt'), 'a')
    filedict[cls] = fd
    
obj_write = "%s %.3f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" #imgname, conf, box*8

sum_total_time = 0
sum_infer_time = 0
sum_post_time = 0

num = 0

all_preds = []
all_targets = []


for idx in range(len(test_dataset)):
# for idx in range(50):
# id = 270
# for idx in [id]:
    image, image_path, ground_truth = test_dataset.__getitem__(idx)
    # import pdb; pdb.set_trace()
    image_name = os.path.basename(image_path).split('.')[0]
    print(image_name)    
    org_h, org_w, _ = image.shape

    h, w = arg.input_size, arg.input_size
    image = cv2.resize(np.array(image), (w, h))
    
    x = image.copy()
    x = x.astype(np.float32)
    x /= 255
    x -= mean
    x /= var
    x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
    x = x.unsqueeze(0)
    
    with torch.no_grad():
        x = x.to(device)
        
        t1 = time.time()
        out = model(x)
        
        if 'gaussnet' in arg.backbone:
            out = out[1]
            
        if 'hourglass' in arg.backbone:
            out = out[0]
        
        out = smoothing(out, arg.kernel)
        peak = nms(out, arg.kernel)
        c_ys, c_xs = topk(peak, k=2000)
    
    x = x[0].cpu().detach().numpy()
    out = out[0].cpu().detach().numpy()
    c_xs = c_xs[0].int().cpu().detach().numpy()
    c_ys = c_ys[0].int().cpu().detach().numpy()
    
    x = x.transpose(1, 2, 0)
    # x *= var
    # x += mean
    # x *= 255
    x = x.clip(0, 255).astype(np.uint8)
    
    t2 = time.time()
    
    results = get_center_point_contour(out, arg.c_thresh, arg.scale, (org_w, org_h))

    t3 = time.time()
    
    _img = image.copy()
    _img_gt = image.copy()
    overlay = image.copy()
    
    # Predicted value
    pred_boxes = []
    pred_labels = []
    
    for result in results:
        box = result['rbox']
        label = result['label']
        color = COLORS[label]

        target_wh = np.array([[w/org_w, h/org_h]], dtype=np.float32)
        box = box * np.tile(target_wh, (4,1))  # shape: (4, 2) [x1, y1], [x2, y2], [x3, y3], [x4, y4]
        
        _img = cv2.drawContours(_img, [box.astype(np.int0)], -1, color, 2)
        
        xmin = np.min(box[:, 0])
        ymin = np.min(box[:, 1])
        xmax = np.max(box[:, 0])
        ymax = np.max(box[:, 1])
        
        # box = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        
        pred_boxes.append(box)
        pred_labels.append(label)
        

    preds = {'boxes': torch.tensor(pred_boxes, dtype=torch.float32),
             'scores': torch.tensor([1.0]*len(pred_boxes)) ,
             'labels': torch.tensor(pred_labels)}
    
    all_preds.append(preds)
    
    # Ground truth value
    true_boxes = []
    true_labels = []
    
    for box in ground_truth:
        # import pdb; pdb.set_trace()
        label = int(box[8])
        color = COLORS[label]
        
        box = np.array(box[:8], dtype=np.float32).reshape(-1, 2)
        target_wh = np.array([[w/org_w, h/org_h]], dtype=np.float32)
        box = box * np.tile(target_wh, (4,1))
        
        _img_gt = cv2.drawContours(_img_gt, [box.astype(np.int0)], -1, color, 2)
        
        xmin = np.min(box[:, 0])
        ymin = np.min(box[:, 1])
        xmax = np.max(box[:, 0])
        ymax = np.max(box[:, 1])
        
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        
        true_boxes.append(box)
        true_labels.append(label)
        
    targets = {'boxes': torch.tensor(true_boxes, dtype=torch.float32),
               'labels': torch.tensor(true_labels)}
    
    all_targets.append(targets)
    
    
    # Plotting visualization after prediction
    merge_out = np.max(out, axis=-1)
    merge_out = np.clip(merge_out * 255, 0, 255)
    
    binary = (merge_out > 0.3*255) * 255
    
    merge_out = cv2.applyColorMap(merge_out.astype(np.uint8), cv2.COLORMAP_JET)
    binary = cv2.applyColorMap(binary.astype(np.uint8), cv2.COLORMAP_JET)
    # import pdb; pdb.set_trace()
    merge_out = cv2.resize(merge_out, (w, h))  # image with bounding box prediction
    binary = cv2.resize(binary, (w, h))  # binary image with thresholding
    
    result_img = cv2.hconcat([_img_gt[:, :, ::-1], _img[:, :, ::-1], merge_out, binary])

    overlay_img = cv2.addWeighted(overlay[:, :, ::-1], 1, merge_out, 0.5, 0)
    
    combine_img = cv2.hconcat([_img_gt[:, :, ::-1], overlay_img, _img[:, :, ::-1]])
    
    cv2.imwrite("%s/%s.jpg" % (result_img_path, image_name), combine_img)


# Calculate mAP
metric = MeanAveragePrecision(class_metrics=True)
metric.update(all_preds, all_targets)
mAP = metric.compute()

from sklearn.metrics import precision_recall_fscore_support

def calculate_iou(box1, box2):
    x1, y1 = torch.max(box1[:2], box2[:2])
    x2, y2 = torch.min(box1[2:], box2[2:])
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

iou_threshold = 0.5
y_true = []
y_pred = []

for pred, gt in zip(all_preds, all_targets):
    gt_boxes = gt['boxes']
    gt_labels = gt['labels']
    pred_boxes = pred['boxes']
    pred_labels = pred['labels']
    matched_gt = set()
    
    for p_box, p_label in zip(pred_boxes, pred_labels):
        found_match = False
        for i, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
            if i in matched_gt:
                continue
            if p_label == g_label:  # Check if labels match
                iou = calculate_iou(p_box, g_box)
                if iou >= iou_threshold:
                    y_true.append(1)  # True Positive
                    y_pred.append(1)
                    matched_gt.add(i)
                    found_match = True
                    break
        if not found_match:
            y_true.append(0)  # False Positive
            y_pred.append(1)
    
    for i, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
        if i not in matched_gt:
            y_true.append(1)  # False Negative
            y_pred.append(0)

# Calculate precision, recall, fscore
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
classification_report = {
    'precision': precision,
    'recall': recall,
    'f1-score': f1,
}

classification_report

from metric_module import ObjectDetectionMetric

metric = ObjectDetectionMetric([0, 1])

# Iterate through each prediction and ground truth pair and update the metric
for pred, gt in zip(all_preds, all_targets):
    bboxes_prediction = pred['boxes'].numpy()
    labels_prediction = pred['labels'].numpy()
    scores_prediction = pred['scores'].numpy()

    bboxes_groundtruth = gt['boxes'].numpy()
    labels_groundtruth = gt['labels'].numpy()

    # Update metric with current predictions and ground truths
    metric.update(
        bboxes_groundtruth=bboxes_groundtruth,
        labels_groundtruth=labels_groundtruth,
        bboxes_prediction=bboxes_prediction,
        labels_prediction=labels_prediction,
        scores_prediction=scores_prediction
    )

# Calculate mAP (mean Average Precision)
mAP = metric.get_mAP(type_mAP="VOC12", conclude=True)
print("mAP:", mAP)

# Get precision-recall curve for a specific class (e.g., class 0)
precisions, recalls = metric.get_precision_recall_curve(no_class=0, thresh_IOU=0.5)
print("Precision:", precisions)
print("Recall:", recalls)

# Get confusion matrix
confusion_matrix = metric.get_confusion(thresh_confidence=0.5, thresh_IOU=0.5, conclude=True)
print("Confusion Matrix:")
print(confusion_matrix)

import pdb; pdb.set_trace()
print("Mean Average Precision (mAP):", mAP)

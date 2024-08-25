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
import pandas as pd


def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', 1)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Tranined checkpoint')
parser.add_argument('--experiment', default=1, type=int, help='number of experiment')
parser.add_argument('--input_size', default=512, type=int, help='input size')
parser.add_argument('--c_thresh', default=0.1, type=float, help='threshold for center point')
parser.add_argument('--backbone', type=str, default='gaussnet_cascade', 
                        help='[gaussnet_cascade, gaussnet]')
parser.add_argument('--kernel', default=3, type=int, help='kernel of max-pooling for center point')
parser.add_argument('--scale', default=1, type=float, help='scale factor')

arg = parser.parse_args()
print(arg)


result_img_path = 'img-out-240215/'
if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)
    
"""Data Loader"""
mean = (0.485, 0.456, 0.406)
var = (0.229, 0.224, 0.225)
   
"""Network Backbone"""

model = Model_factory(arg.backbone, 2)
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
    

num = 0

normal_path = '/mnt/data/dataset/apple-defects/normal/valid/images/'

all_preds = {}

for image in os.listdir(normal_path):
    image_path = os.path.join(normal_path, image)
    image_name = os.path.basename(image_path).split('.')[0]
    print(image_name)    
    image = cv2.imread(image_path)
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
    x = x.clip(0, 255).astype(np.uint8)
    
    t2 = time.time()
    
    results = get_center_point_contour(out, arg.c_thresh, arg.scale, (org_w, org_h))
    
    t3 = time.time()
    
    _img = image.copy()
    _img_gt = image.copy()
    
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

        all_preds[image_name] = {'boxes': pred_boxes, 'labels': pred_labels}
    

    
    
    # Plotting visualization after prediction
    merge_out = np.max(out, axis=-1)
    merge_out = np.clip(merge_out * 255, 0, 255)

    binary = (merge_out > 0.3*255) * 255
    
    merge_out = cv2.applyColorMap(merge_out.astype(np.uint8), cv2.COLORMAP_JET)
    binary = cv2.applyColorMap(binary.astype(np.uint8), cv2.COLORMAP_JET)
    
    merge_out = cv2.resize(merge_out, (w, h))  # image with bounding box prediction
    binary = cv2.resize(binary, (w, h))  # binary image with thresholding
    
    # result_img = cv2.hconcat([_img_gt[:, :, ::-1], _img[:, :, ::-1], merge_out, binary])
    
    result_img = cv2.hconcat([_img_gt, merge_out, _img])

    cv2.imwrite("%s/%s.jpg" % (result_img_path, image_name), result_img)

pred_df = []
for key, pred in all_preds.items():
    for i, box in enumerate(pred['boxes']):
        pred_df.append({'image': key, 'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3], 'label': pred['labels'][i]})

df = pd.DataFrame(pred_df, columns=['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])        
df.to_csv('./assets/pred.csv', index=False)     
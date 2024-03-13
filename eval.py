import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T

import utils
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import math
import sys
import time

import torchvision.models.detection.mask_rcnn
from coco import CocoEvaluator,get_coco_api_from_dataset



from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import matplotlib.pyplot as plt
from torchvision.io import read_image


torch.multiprocessing.set_sharing_strategy('file_system')

# TensorBoard Writer
writer = SummaryWriter('runs/PennFudanPed')


def get_instance_segmentation_model(num_classes):
   """
   Load a pre-trained Mask R-CNN model and customize the head for the desired number of classes.
   """
   model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

   # Get the number of input features for the box predictor
   in_features_box = model.roi_heads.box_predictor.cls_score.in_features
   # Replace the pre-trained head with a new one
   model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

   # Get the number of input features for the mask classifier
   in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
   hidden_layer = 256
   # Replace the mask predictor with a new one
   model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

   return model

def get_transform(train):
   """
   Get the appropriate transformation for training or evaluation.
   """
   transforms = []
   if train:
       transforms.append(T.RandomHorizontalFlip(0.5))
   transforms.append(T.ToDtype(torch.float, scale=True))
   transforms.append(T.ToPureTensor())
   return T.Compose(transforms)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of classes
num_classes = 2



# Load model
model = get_instance_segmentation_model(num_classes)
model.to(device)


model.load_state_dict(torch.load("model.pth"))

# Evaluation
image = torchvision.io.read_image("data/PennFudanPed/ped_test.jpg")
eval_transform = get_transform(train=False)

model.eval()
with torch.no_grad():
   x = eval_transform(image)
   x = x[:3, ...].to(device)
   predictions = model([x, ])
   pred = predictions[0]

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]


confidence_threshold = 0.75
pred_labels = []
pred_boxes = []
pred_masks = []
for label, score, box, mask in zip(pred["labels"], pred["scores"], pred["boxes"], pred["masks"]):
    if(score >= confidence_threshold):
        pred_labels.append(f"pedestrian: {score:.3f}")
        pred_boxes.append(box)
        pred_masks.append(mask)

if(len(pred_boxes)>0):
    pred_boxes=torch.stack(pred_boxes)
else:
    pred_boxes=torch.Tensor(pred_boxes)

if(len(pred_masks)>0):
    pred_masks=torch.stack(pred_masks)
else:
    pred_masks=torch.Tensor(pred_masks)


if(len(pred_boxes)>0):
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="black")
    if(len(pred_masks)>0):
        masks = (pred_masks > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="red")
    writer.add_image('output_image', output_image)

writer.close()

print("Evaluation Complete")
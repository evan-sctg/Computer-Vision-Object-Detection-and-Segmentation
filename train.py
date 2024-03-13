import sys
import time
import os
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as T
import utils
import torchvision
from torchvision import tv_tensors
from torchvision.io import read_image
from coco import CocoEvaluator, get_coco_api_from_dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.boxes import masks_to_boxes
import torchvision.models.detection.mask_rcnn
from torchvision.transforms.v2 import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

torch.multiprocessing.set_sharing_strategy('file_system')

# TensorBoard Writer
writer = SummaryWriter('runs/PennFudanPed')

# Training and Evaluation Functions
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        writer.add_scalar('training loss',loss_value,epoch)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types




@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator




# Model Creation and Customization
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


# Data Transformation
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


# Custom Dataset
class PennFudanDataset(torch.utils.data.Dataset):
   """
   Custom dataset for the Penn-Fudan Pedestrian Database.
   """
   def __init__(self, root, transforms):
       self.root = root
       self.transforms = transforms
       self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
       self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

   def __getitem__(self, idx):
       img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
       mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
       img = read_image(img_path)
       mask = read_image(mask_path)

       obj_ids = torch.unique(mask)
       obj_ids = obj_ids[1:]  # Remove background
       num_objs = len(obj_ids)

       masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
       boxes = masks_to_boxes(masks)

       labels = torch.ones((num_objs,), dtype=torch.int64)
       image_id = idx
       area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
       iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

       img = tv_tensors.Image(img)
       target = {
           "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=T.functional.get_size(img)),
           "masks": tv_tensors.Mask(masks),
           "labels": labels,
           "image_id": image_id,
           "area": area,
           "iscrowd": iscrowd
       }

       if self.transforms is not None:
           img, target = self.transforms(img, target)

       return img, target

   def __len__(self):
       return len(self.imgs)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of classes
num_classes = 2

# Datasets and Data Loaders
dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader = torch.utils.data.DataLoader(
   dataset,
   batch_size=2,
   shuffle=True,
   num_workers=4,
   collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
   dataset_test,
   batch_size=1,
   shuffle=False,
   num_workers=4,
   collate_fn=utils.collate_fn
)

# Load Model
model = get_instance_segmentation_model(num_classes)
model.to(device)

# Optimizer and Learning Rate Scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training
num_epochs = 5
for epoch in range(num_epochs):
   train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
   lr_scheduler.step()
   evaluate(model, data_loader_test, device=device)

print("Training Complete")
# Save Model
torch.save(model.state_dict(), "model.pth")
print("Model Saved")

# Close TensorBoard Writer
writer.close()

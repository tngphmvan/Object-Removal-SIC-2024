import os
import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
from .detailRemove import DetailRemove
import importlib
from .aotgan import InpaintGenerator
def load_models(inpaint_model_path):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    rcnn = rcnn.eval()
    
    inpaint_model = InpaintGenerator(rates=[1, 2, 4, 8], block_num=8)
    inpaint_model.load_state_dict(torch.load(inpaint_model_path, map_location="cpu"))
    inpaint_model.eval()

    return rcnn, transforms, inpaint_model

def object_removal(image_path, rcnn, transforms, deepfill, rect):
    model = DetailRemove(segmentModel=rcnn, rcnn_transforms=transforms, inpaintModel=deepfill, image_path=image_path)
    rect[-1] = tuple(rect[-1])
    model.box = rect 
    print('Tọa độ: ', rect)
    output = model.run()
    img = cv2.cvtColor(model.image_orig[0].permute(1,2,0).numpy(),cv2.COLOR_RGB2BGR)
    boxed = cv2.rectangle(img, (model.box[0], model.box[1]),(model.box[2], model.box[3]), (0,255,0),2)
    boxed = cv2.cvtColor(boxed,cv2.COLOR_BGR2RGB)
    inpainted_image = output

    return inpainted_image

inpaint_model_path = r"D:\checkpoints\G0000000.pt"
def main_processing_detail(image_path, rect):
    rcnn, transforms, inpaint_model = load_models(inpaint_model_path)
    inpainted_image = object_removal(image_path, rcnn, transforms, inpaint_model, rect)
    
    
    inpainted_path = "inpainted_image.png"
    
    cv2.imwrite(inpainted_path, cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))
    
    return inpainted_path

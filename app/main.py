import os
import cv2
import torch
from .objRemove import ObjectRemove
from .objRemove2 import ObjectRemove2
from .deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt

def load_models(deepfill_weights_path):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    rcnn = rcnn.eval()
    
    deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)
    
    return rcnn, transforms, deepfill

def object_removal(image_path, rcnn, transforms, deepfill, rect):
    model = ObjectRemove(segmentModel=rcnn, rcnn_transforms=transforms, inpaintModel=deepfill, image_path=image_path)
    rect[-1] = tuple(rect[-1])
    model.box = rect 
    print('Tọa độ: ', rect)
    output = model.run()
    img = cv2.cvtColor(model.image_orig[0].permute(1,2,0).numpy(),cv2.COLOR_RGB2BGR)
    boxed = cv2.rectangle(img, (model.box[0], model.box[1]),(model.box[2], model.box[3]), (0,255,0),2)
    boxed = cv2.cvtColor(boxed,cv2.COLOR_BGR2RGB)
    original_image = boxed
    masked_image = model.image_masked.permute(1, 2, 0).detach().numpy()
    inpainted_image = output

    return original_image, masked_image, inpainted_image

def main_processing(image_path, rect, model_path):
    deepfill_weights_path = os.path.join(model_path, "states_pt_places2.pth")
    rcnn, transforms, deepfill = load_models(deepfill_weights_path)
    original_image, masked_image, inpainted_image = object_removal(image_path, rcnn, transforms, deepfill, rect)
    
    orig_path = "original_image.png"
    masked_path = "masked_image.png"
    inpainted_path = "inpainted_image.png"
    
    cv2.imwrite(orig_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(masked_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(inpainted_path, cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))
    
    return orig_path, masked_path, inpainted_path
def objRemove2_main_process(image_path, mask, mask_id, model_path):
    deepfill_weights_path = os.path.join(model_path, "states_pt_places2.pth")
    rcnn, transforms, deepfill = load_models(deepfill_weights_path)

    model = ObjectRemove2(segmentModel=rcnn, rcnn_transforms=transforms, inpaintModel=deepfill, image_path=image_path, mask_info=mask)
    output = model.run(mask_id)
    inpainted_image = output

    inpainted_path = "inpainted_image.png"

    cv2.imwrite(inpainted_path, cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))
    
    return inpainted_path



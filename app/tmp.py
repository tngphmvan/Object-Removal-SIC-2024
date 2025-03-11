import base64
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from io import BytesIO
from PIL import Image
import random
import requests
from urllib.request import urlopen

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold=0.5, url=False):
    if url:
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_path)
        
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().numpy()
    
    masks = torch.tensor(masks, dtype=torch.float32) 
    
    masks = masks.unsqueeze(1)  
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    mask_ids = list(range(len(masks))) 
    mask_info = {mask_id: {'mask': masks[i], 
                           'class': pred_class[i], 
                           'bbox': [pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], pred_boxes[i][1][1], (pred_boxes[i][1][0], pred_boxes[i][1][1])]}
                 for i, mask_id in enumerate(mask_ids)}
    
    mask_info1 = {mask_id: {'bbox': [pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], pred_boxes[i][1][1]]}
                  for i, mask_id in enumerate(mask_ids)}
    
    return mask_info, mask_info1

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    return image

def random_color_masks(image):
    colors = [
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
        [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]
    ]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    unique_labels = np.unique(image)
    for label in unique_labels:
        if label == 0: 
            continue
        color = colors[random.randrange(0, len(colors))]
        r[image == label], g[image == label], b[image == label] = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask

def instance_segmentation(img_path, threshold=0.5, url=False):
    mask_info, mask_info1 = get_prediction(img_path, threshold=threshold, url=url)
    
    if url:
        img = url_to_image(img_path)
    else:
        img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for mask_id, info in mask_info.items():
        mask = info['mask'].squeeze(0) 
        rgb_mask = random_color_masks(mask)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    
    
    return img, mask_info1
import copy
import cv2
import numpy as np
import torch
import torchvision.transforms as T  
from torchvision.io import read_image
from .inpaint_module import inpaint_image 

class DetailRemove():

    def __init__(self, segmentModel=None, rcnn_transforms=None, inpaintModel=None, image_path='') -> None:
        self.segmentModel = segmentModel
        self.inpaintModel = inpaintModel
        self.rcnn_transforms = rcnn_transforms
        self.image_path = image_path
        self.highest_prob_mask = None
        self.image_orig = None
        self.image_masked = None
        self.box = None

    def run(self):
        print('Reading in image')
        images = self.preprocess_image()
        self.image_orig = images
        print("segmentation")
        output = self.segment(images)
        out = output[0]
        self.highest_prob_mask = self.find_mask(out, self.box)
        self.highest_prob_mask[self.highest_prob_mask > 0.1]  = 1
        self.highest_prob_mask[self.highest_prob_mask <0.1] = 0
        self.image_masked = (images[0]*(1-self.highest_prob_mask))
        print('inpaint')
        output = self.inpaint()
        return output

    def inpaint(self):
        print("[**] inpainting ... ")
        comp_np = inpaint_image(self.image_path, self.inpaintModel, self.image_orig[0], self.highest_prob_mask)
        print("inpainting finish!")
        return comp_np

    def percent_within(self,nonzeros, rectangle):
        rect_ul, rect_br = rectangle
        inside_count = 0
        for _,y,x in nonzeros:
            if x >= rect_ul[0] and x<= rect_br[0] and y <= rect_br[1] and y>= rect_ul[1]:
                inside_count+=1
        return inside_count / len(nonzeros)

    def iou(self, boxes_a, boxes_b):
        x1 = np.array([boxes_a[:,0], boxes_b[:,0]]).max(axis=0)
        y1 = np.array([boxes_a[:,1], boxes_b[:,1]]).max(axis=0)
        x2 = np.array([boxes_a[:,2], boxes_b[:,2]]).min(axis=0)
        y2 = np.array([boxes_a[:,3], boxes_b[:,3]]).min(axis=0)
        w = x2-x1
        h = y2-y1
        w[w<0] = 0
        h[h<0] = 0
        intersect = w* h
        area_a = (boxes_a[:,2] - boxes_a[:,0]) * (boxes_a[:,3] - boxes_a[:,1])
        area_b = (boxes_b[:,2] - boxes_b[:,0]) * (boxes_b[:,3] - boxes_b[:,1])
        union = area_a + area_b - intersect
        return intersect / (union + 0.00001)

    def find_mask(self, rcnn_output, rectangle):
        print("Rect: ", rectangle)
        rectangle = ((rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]))
        _, h, w = rcnn_output['masks'].shape[1:]
        mask = torch.zeros((h, w), dtype=torch.float32)
        rect_ul, rect_br = rectangle
        rect_x1, rect_y1 = rect_ul
        rect_x2, rect_y2 = rect_br
        
        mask[rect_y1:rect_y2, rect_x1:rect_x2] = 1
        
        mask = mask.unsqueeze(0).expand_as(rcnn_output['masks'][0]) 
        
        return mask

    def preprocess_image(self):
        img= [read_image(self.image_path)]
        _, h, w = img[0].shape
        size = min(h, w)
        if size > 512:
            img[0] = T.Resize(512, max_size=680, antialias=True)(img[0])
        images_transformed = [self.rcnn_transforms(d) for d in img]
        return images_transformed

    def segment(self, images):
        out = self.segmentModel(images)
        return out

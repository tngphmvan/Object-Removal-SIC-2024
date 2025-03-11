import copy
import cv2
import numpy as np
import torchvision.transforms as T  
from torchvision.io import read_image

class ObjectRemove2():

    def __init__(self, segmentModel = None, rcnn_transforms = None, inpaintModel= None, image_path = '', mask_info = None) -> None:
        self.segmentModel = segmentModel
        self.inpaintModel = inpaintModel
        self.rcnn_transforms = rcnn_transforms
        self.image_path = image_path
        self.mask_info = mask_info 
        self.highest_prob_mask = None
        self.image_orig = None
        self.image_masked = None
        self.box = None

    def run(self, mask_id=None):
        print('Reading in image')
        images = self.preprocess_image()
        self.image_orig = images
        if mask_id is not None and mask_id in self.mask_info:
            mask = self.mask_info[mask_id]['mask']
            self.highest_prob_mask = mask
        else:
            print('No valid mask_id provided or mask_id not found.')
            return None
        print('Mask: ', self.highest_prob_mask)
        self.highest_prob_mask[self.highest_prob_mask > 0.1] = 1
        self.highest_prob_mask[self.highest_prob_mask < 0.1] = 0
        self.image_masked = (images[0] * (1 - self.highest_prob_mask))
        print('inpaint')
        output = self.inpaint()
        return output

    def percent_within(self,nonzeros, rectangle):
        '''
        Calculates percent of mask inside rectangle
        '''
        rect_ul, rect_br = rectangle
        inside_count = 0
        for _,y,x in nonzeros:
            if x >= rect_ul[0] and x<= rect_br[0] and y <= rect_br[1] and y>= rect_ul[1]:
                inside_count+=1
        return inside_count / len(nonzeros)

    def iou(self, boxes_a, boxes_b):
        '''
        Calculates IOU between all pairs of boxes

        boxes_a and boxes_b are matrices with each row representing the 4 coords of a box
        '''

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
        '''
        Finds the mask with highest probability in the rectangle given
        
        '''
        bounding_boxes= rcnn_output['boxes'].detach().numpy()
        masks = rcnn_output['masks']

        ref_boxes  = np.array([rectangle], dtype=object)
        ref_boxes = np.repeat(ref_boxes, bounding_boxes.shape[0], axis=0)

        ious= self.iou(ref_boxes, bounding_boxes)

        best_ind = np.argmax(ious)

        return masks[best_ind]


        #compare masks pixelwise
        '''
        masks = rcnn_output['masks']
        #go through each nonzero point in the mask and count how many points are within the rectangles
        highest_prob_mask = None
        percent_within,min_diff = 0,float('inf')
        #print('masks lenght:', len(masks))


        for m in range(len(masks)):
            #masks[m][masks[m] > 0.5] = 255.0
            #masks[m][masks[m] < 0.5] = 0.0
            nonzeros = np.nonzero(masks[m])
            #diff = rect_area - len(nonzeros)
            p = self.percent_within(nonzeros, rectangle)
            if p > percent_within:
                highest_prob_mask = masks[m]
                percent_within = p
            print(p)
        return highest_prob_mask
        '''

    def preprocess_image(self):
        img= [read_image(self.image_path)]
        _,h,w = img[0].shape
        size = min(h,w)
        if size > 512:
            img[0] = T.Resize(512, max_size=680, antialias=True)(img[0])

        images_transformed = [self.rcnn_transforms(d) for d in img]
        return images_transformed
   
    
    def segment(self,images):
        out = self.segmentModel(images)
        return out
    
    def inpaint(self):
        output = self.inpaintModel.infer(self.image_orig[0], self.highest_prob_mask, return_vals=['inpainted'])
        return output[0]



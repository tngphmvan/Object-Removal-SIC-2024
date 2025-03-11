import torch
import numpy as np
from torchvision.transforms import ToTensor
import cv2
import torch.nn.functional as F

def match_histograms(src, ref):
    matched = np.zeros_like(src)
    for i in range(3): 
        src_hist, bins = np.histogram(src[:, :, i].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref[:, :, i].flatten(), 256, [0, 256])
        cdf_src = src_hist.cumsum()
        cdf_ref = ref_hist.cumsum()
        cdf_src = cdf_src / cdf_src[-1]
        cdf_ref = cdf_ref / cdf_ref[-1]
        interp_map = np.interp(cdf_src, cdf_ref, np.arange(256))
        matched[:, :, i] = interp_map[src[:, :, i]]
    return matched

def postprocess(image, origional_image_path):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)

    orig_image = cv2.imread(origional_image_path)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  
    image = match_histograms(image, orig_image)

    return image
def resize_tensor(tensor, size):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

def inpaint_image(origional_image_path, model, image, mask):
    with torch.no_grad():
        if not isinstance(image, torch.Tensor):
            img_tensor = (ToTensor()(image) * 2.0 - 1.0).unsqueeze(0)
        else:
            img_tensor = image.unsqueeze(0)

        if not isinstance(mask, torch.Tensor):
            mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        else:
            mask_tensor = mask.unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()
        
        if img_tensor.shape[2:] != mask_tensor.shape[2:]:
            mask_tensor = resize_tensor(mask_tensor, img_tensor.shape[2:])

        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        
        if pred_tensor.shape[2:] != img_tensor.shape[2:]:
            pred_tensor = resize_tensor(pred_tensor, img_tensor.shape[2:])

        comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)
        
        pred_np = postprocess(pred_tensor[0], origional_image_path)
        masked_np = postprocess(masked_tensor[0], origional_image_path)
        comp_np = postprocess(comp_tensor[0], origional_image_path)
        
        return comp_np



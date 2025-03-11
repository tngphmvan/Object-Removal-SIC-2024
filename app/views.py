from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
import base64
import cv2
import numpy as np
from io import BytesIO
import os
from .main import main_processing
from .tmp import instance_segmentation
from .main2 import main_processing_detail
import json
from time import time


@csrf_exempt
@api_view(['POST'])
def detail_remove_process(request):
    data = request.data
    image_data = data.get('image')
    rects = data.get('rects', [])
    if not rects:
        return JsonResponse({'error': 'No rectangles provided'}, status=400)
    image_data = image_data.split(',')[1]
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if image is None:
        return JsonResponse({'error': 'Image cannot be loaded'}, status=400)
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, image)
    start_time = time()
    inpainted_path = main_processing_detail(temp_image_path, rects[0])
    end_time = time()
    print("Time: ", end_time-start_time)
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    result = {
        'inpainted': encode_image_to_base64(inpainted_path)
    }
    os.remove(temp_image_path)
    os.remove(inpainted_path)

    return JsonResponse(result)


mask_info = None
temp_path_for_removing = ''
@csrf_exempt
def segment_remove(request):
    global mask_info, temp_path_for_removing
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image file provided'}, status=400)
        try:
            mask_id = int(request.POST.get('mask_id'))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return JsonResponse({'error': 'Invalid mask_info or mask_id'}, status=400)

        if mask_id not in mask_info:
            return JsonResponse({'error': 'Invalid mask_id'}, status=400)

        image = Image.open(image_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        temp_image_path = "temp_image.png"
        cv2.imwrite(temp_image_path, image)

        rect = mask_info[mask_id]['bbox']
        rect.append([rect[2], rect[3]])
        if not rect:
            return JsonResponse({'error': 'Invalid mask bounding box'}, status=400)

        model_path = r'C:\Users\TUAN PC\Documents\MainProject\main_project\app\models'
       
        orig_path, masked_path, inpainted_path = main_processing(temp_path_for_removing, rect, model_path)

        def encode_image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        result = {
            'inpainted': encode_image_to_base64(inpainted_path)
        }

        os.remove(temp_image_path)
        os.remove(orig_path)
        os.remove(masked_path)
        os.remove(inpainted_path)
        os.remove(temp_path_for_removing)

        return JsonResponse(result)

    return JsonResponse({'error': 'Invalid request'}, status=400)
@csrf_exempt
def process_segment_api(request):
    global mask_info, temp_path_for_removing
    if request.method == 'POST':
        image_file = request.FILES['image']
        image = Image.open(image_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        temp_image_path = "temp_image.png"
        temp_path_for_removing= 'temp_path_for_removing.png'
        cv2.imwrite(temp_image_path, image)
        cv2.imwrite(temp_path_for_removing, image)
        start_time = time()
        segmented_image_np, mask_info = instance_segmentation(temp_image_path, threshold=0.5, url=False)
        end_time = time()
        print("Time: ", end_time-start_time)
        segmented_image = Image.fromarray(segmented_image_np)
        buffered = BytesIO()
        segmented_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        result = {'segmented_image': img_str,
                  'mask_info': mask_info}
        return JsonResponse(result)

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
@api_view(['POST'])
def process_image_api(request):
    data = request.data
    image_data = data.get('image')
    rects = data.get('rects', [])

    if not rects:
        return JsonResponse({'error': 'No rectangles provided'}, status=400)
    image_data = image_data.split(',')[1]
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if image is None:
        return JsonResponse({'error': 'Image cannot be loaded'}, status=400)

    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, image)

    model_path = r'C:\Users\TUAN PC\Documents\MainProject\main_project\app\models'

    start_time = time()
    orig_path, masked_path, inpainted_path = main_processing(temp_image_path, rects[0], model_path)
    end_time = time()
    print("Time: ", end_time-start_time)

    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    result = {
        'original': encode_image_to_base64(orig_path),
        'masked': encode_image_to_base64(masked_path),
        'inpainted': encode_image_to_base64(inpainted_path)
    }

    os.remove(temp_image_path)
    os.remove(orig_path)
    os.remove(masked_path)
    os.remove(inpainted_path)

    return JsonResponse(result)

def index(request):
    return render(request, 'app/home.html')

def removal(request):
    return render(request, 'app/removal.html')

def tutorial(request):
    return render(request, 'app/tutorial.html')

def autoseg(request):
    return render(request, 'app/autoseg.html')

def detail_remove(request):
    return render(request, 'app/detail_remove.html')






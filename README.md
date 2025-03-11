# Object-Removal-SIC-2024

## 📌 Introduction
Object-Removal-SIC-2024 is an **AI-powered object removal application** that leverages **object segmentation** and **GAN-based inpainting** to seamlessly remove unwanted objects from images and replace them with realistic background textures. Users can provide images either via **file upload** or **webcam capture**.

This project is developed as part of the **SIC Final Project (Aug 2024 – Sep 2024)** with a team of **5 members**.

## 🛠️ Tech Stack
- **Frameworks & Libraries:** Python, PyTorch, Tensorboard, Django, OpenCV
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Django
- **Deep Learning Models:** U-Net, Mask R-CNN, AOT-GAN

## 🖼️ Object Segmentation
For accurate segmentation, the following models are utilized:
- **U-Net**
- **Mask R-CNN**

### 📊 Dataset
- **2000 images** from Pascal-VOC-2007
- **2500 images** from Places2
- **20 object categories**

### 🎯 Performance Metrics
- **U-Net Dice Score:** 0.34
- **Mask R-CNN Dice Score:** 0.94

## 🎨 Inpainting Algorithm
The **AOT-GAN** model is used for inpainting, ensuring high-quality background reconstruction after object removal.

### 📊 Training Performance
- **MAE:** 0.098
- **PSNR:** 21.24
- **FID:** 68.27
- **SSIM:** 0.61

## 🚀 Application Workflow
1. **Upload an image** or **capture via webcam**.
2. **Segment the object** using U-Net or Mask R-CNN.
3. **Remove the object** and fill the missing region using AOT-GAN.
4. **Display and download** the processed image.

## ⚙️ Installation & Usage
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/yourusername/Object-Removal-SIC-2024.git
cd Object-Removal-SIC-2024
```

### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application:
```bash
python manage.py runserver
```
Then, open **http://127.0.0.1:8000/** in your browser.

## 🎯 Future Enhancements
- Improve **segmentation accuracy** using transformer-based models.
- Optimize **inpainting quality** with advanced GAN architectures.
- Deploy as a **web service/API** for real-world use cases.

## 🤝 Contributions
Contributions are welcome! Feel free to fork, create pull requests, or open issues.

## 📜 License
This project is licensed under the **MIT License**.


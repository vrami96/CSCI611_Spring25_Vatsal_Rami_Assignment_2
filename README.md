# **README: CNN Implementation and Image Filtering**

## **1. Overview**
This project implements a **Convolutional Neural Network (CNN)** for image classification using **PyTorch** and explores **image filtering techniques** such as **Sobel Edge Detection, Harris Corner Detection, and Scaling** using **OpenCV**. The CNN was trained on a dataset, optimized, and evaluated for accuracy.

---

## **2. Installation Requirements**
Before running the code, ensure you have the following dependencies installed:

```bash
pip install torch torchvision torchaudio torchsummary torchviz numpy matplotlib opencv-python
```

If you use Jupyter Notebook, install `notebook`:

```bash
pip install notebook
```

---

## **3. Running the CNN Model**
### **3.1 Training the CNN**
To train the CNN, run:

```bash
python train_cnn.py
```
or execute the corresponding Jupyter Notebook cell.

### **3.2 Code Breakdown**
The CNN model is implemented in **PyTorch** with multiple convolutional layers, batch normalization, dropout, and fully connected layers.

To initialize and train the model:

```python
from model import Net  # Import the CNN model
import torch

model = Net()  # Instantiate the model

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

To start training:

```bash
python train_cnn.py
```

---

## **4. Running Image Filtering**
### **4.1 Edge Detection**
To apply Sobel edge detection:

```bash
python image_filtering.py --filter sobel
```

or modify the script:

```python
from image_processing import apply_sobel

image = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)
sobel_edges = apply_sobel(image)
cv2.imshow("Sobel Edge Detection", sobel_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### **4.2 Harris Corner Detection**
To run Harris Corner Detection:

```bash
python image_filtering.py --filter harris
```

### **4.3 Scaling**
To downscale an image:

```bash
python image_filtering.py --scale 2
```

---

## **5. Model Evaluation**
Once the model is trained, evaluate it using:

```bash
python evaluate_cnn.py
```

You can also visualize the training performance using:

```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

---

## **6. Results**
- The CNN model achieved **77% accuracy** after optimizations.
- Image filtering techniques effectively detected **edges, corners, and scaled images** for preprocessing.

---

## **7. Future Improvements**
- Implement **ResNet** for better accuracy.
- Apply **data augmentation** techniques.
- Use **hyperparameter tuning** for optimization.

---

## **8. Contribution & Support**
If you wish to contribute:
1. Fork this repository.
2. Make modifications.
3. Create a pull request.

For issues, open a discussion thread.

---

This README file provides a **step-by-step guide to using the CNN model and image filtering tools**. 🚀 Let me know if you need any modifications!

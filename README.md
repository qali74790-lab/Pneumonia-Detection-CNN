# 🫁 Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Deep Learning](https://img.shields.io/badge/Skill-CNN-yellow)

## 📌 Project Overview
This project uses *Deep Learning (Convolutional Neural Networks)* to automatically detect Pneumonia from Chest X-Ray images. The model classifies images into two categories: *Normal* vs. *Pneumonia*, helping medical professionals make faster and more accurate diagnoses.

The model was trained on the *Chest X-Ray Images (Pneumonia)* dataset from Kaggle and achieved an accuracy of *~94%* on the validation set.

---

## 🚀 Key Features
* *Data Augmentation:* Used ImageDataGenerator (Zoom, Shear, Flip) to prevent overfitting and improve model generalization.
* *CNN Architecture:* Built a custom Sequential CNN model with Conv2D, MaxPooling, and Dropout layers.
* *Medical Metrics:* Focused on *Recall (Sensitivity)* rather than just Accuracy, ensuring that sick patients are not misdiagnosed as healthy (False Negatives).

---

## 📊 Performance & Results
* *Accuracy:* 94%
* *Loss:* Low validation loss indicating good generalization.

### Why "Recall" Matters More Than Accuracy
In medical diagnosis, a *False Negative* (telling a sick patient they are healthy) is far more dangerous than a *False Positive* (flagging a healthy patient for further checks).
* *Goal:* Maximize Recall to ensure we catch every Pneumonia case.
* *Result:* The model successfully identifies positive cases with high sensitivity.

---

## 🛠 Tech Stack
* *Language:* Python
* *Libraries:* TensorFlow, Keras, NumPy, Matplotlib
* *Environment:* Jupyter Notebook / Google Colab

---

## 📂 Dataset
The dataset is organized into 3 folders (train, test, val) and contains 5,863 X-Ray images (JPEG).
* *Source:* [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 💻 How to Run This Project
1.  *Clone the repository:*
    bash
    git clone [https://github.com/your-username/pneumonia-detection.git](https://github.com/your-username/pneumonia-detection.git)
    
2.  *Install dependencies:*
    bash
    pip install tensorflow numpy matplotlib
    
3.  *Run the Notebook:*
    Open Pneumonia_Detection_CNN.ipynb in Jupyter Notebook or Google Colab and run all cells.

---

## 🔮 Future Improvements
* *Grad-CAM:* Implement Class Activation Maps to visualize where in the lungs the AI is looking.
* *Web App:* Deploy the model using *Streamlit* to allow users to upload their own X-rays.

---

### 👨‍💻 Author 
*Qasim Ali*
* [LinkedIn Profile](https://www.linkedin.com/in/qasam-ali-743286333)
* [Email](mailto:qali74790@gmail.com)# Pneumonia-Detection-CNN
AI system to detect pneumonia from chest X-RAY using TensorFlow 

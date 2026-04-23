# 🛡️ AI-Enhanced Blind Steganalysis for Image Security

## Overview
This project presents a machine learning-based blind steganalysis system designed to detect hidden data in digital images. Unlike traditional approaches, the system does not require prior knowledge of the embedding technique and instead relies on statistical and spatial feature analysis.

This system achieves approximately 88% accuracy in detecting steganographic content using handcrafted features and machine learning techniques.

This project was developed as part of my AI/ML internship experience.

---

## Key Features

- Detection of hidden (stego) content in images without prior embedding knowledge  
- Extraction of statistical and spatial features  
- Optimized Support Vector Machine (SVM) classifier  
- Real-time prediction using a Flask-based web application  
- End-to-end pipeline from dataset preprocessing to deployment  

---

## Feature Engineering

The model uses the following features to identify steganographic artifacts:

- Mean Intensity  
- Standard Deviation  
- Entropy  
- LSB (Least Significant Bit) Ratio  
- Horizontal Pixel Difference Ratio  
- Vertical Pixel Difference Ratio  
- Noise Variance  
- Even-Odd Pixel Ratio  

These features capture both statistical irregularities and spatial inconsistencies introduced during data embedding.

---

## Model & Training

- Algorithm: Support Vector Machine (SVM)  
- Kernel: RBF  
- Hyperparameter tuning: GridSearchCV  
- Validation: 5-fold Stratified Cross Validation  

### Performance

- Accuracy: ~88%  
- Evaluated using Confusion Matrix, Precision, Recall, and F1-score  

---

## Web Application

A lightweight web interface was developed using Flask, allowing users to:

- Upload an image  
- Analyze it in real-time  
- Receive classification output:
  - Cover Image  
  - Stego Image  

---

## Project Structure

- train_model.py  
- app.py  
- generate_stego.py  
- check_dataset.py  
- verify_dataset.py  
- check_stego.py  
- stego_model.pkl  
- templates/index.html  
- static/style.css  
- screenshots/home.png  
- screenshots/result.png  
- requirements.txt  

---

## Tech Stack

- Python  
- OpenCV  
- NumPy  
- Scikit-learn  
- Flask  

---

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train model:
python train_model.py

3. Run application:
python app.py

4. Open in browser:
http://127.0.0.1:5000/

---

## Screenshots

### 🏠 Home Interface
![Home](home.png)

### ✅ Cover Image Detection
![Cover](cover.png)

### ⚠️ Stego Image Detection
![Stego](stego.png)

---

## Key Highlights

- End-to-end ML system (data → model → deployment)  
- Focus on explainable feature engineering  
- Computationally efficient compared to deep learning-based approaches  
- Practical real-world usability via web interface  

---

## Future Improvements

- Integration of deep learning models (CNN-based steganalysis)  
- Support for larger datasets  
- Performance optimization for real-time applications  

---

## Author

Nandana Santhosh

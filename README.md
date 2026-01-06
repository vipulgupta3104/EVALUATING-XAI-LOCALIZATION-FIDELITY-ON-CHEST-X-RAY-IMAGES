# EVALUATING-XAI-LOCALIZATION-FIDELITY-ON-CHEST-X-RAY-IMAGES
This project evaluates the **localization fidelity of Explainable AI (XAI) techniques** applied to deep learning models for chest X-ray image classification. The objective is to analyze how accurately different XAI methods highlight infected regions and improve interpretability in medical imaging systems.

## 📌 Project Overview

- Applied **Explainable AI (XAI)** techniques to interpret CNN-based chest X-ray classifiers.
- Generated **infection localization masks** using **LIME** and **Grad-CAM**.
- Evaluated spatial alignment of explanations against **ground-truth infection masks**.
- Benchmarked localization quality using **IoU, Dice Score, and Hausdorff Distance**.
- Demonstrated that **LIME provides superior spatial fidelity** compared to gradient-based explanations.

## 🧰 Tech Stack & Tools

- Python  
- PyTorch  
- LIME  
- Grad-CAM  
- OpenCV, NumPy  
- Pandas  
- Matplotlib  
- Jupyter Notebook  

## 📊 Datasets Used

- **COVID-QU-Ex Dataset**  
  https://www.kaggle.com/datasets/anasmohammedtahir/covidqu  

- **CheXpert Dataset (Stanford)**  
  https://aimi.stanford.edu/datasets/chexpert-chest-x-rays  
  https://www.kaggle.com/datasets/ashery/chexpert  

These datasets provide chest X-ray images along with **ground-truth infection segmentation masks**, enabling quantitative evaluation of XAI localization performance. To use these datasets, code is also provided in their respective folders in order to preprocess and get guidance over their usage.

## 🔍 Key Features

- Application of **LIME and Grad-CAM** on PyTorch-based CNN models
- Pixel-level infection localization mask generation
- Quantitative evaluation using:
  - Intersection over Union (IoU)
  - Dice Score
  - Hausdorff Distance
- Comparative analysis of **perturbation-based vs gradient-based** explainability methods
- Support for **binary and multi-class classification** settings

## 📁 Workflow

1. Load trained CNN models and chest X-ray images  
2. Apply XAI techniques (LIME / Grad-CAM) to generate saliency maps  
3. Post-process saliency maps into infection localization masks  
4. Compare predictions with ground-truth segmentation masks  
5. Evaluate localization fidelity using standard segmentation metrics  

## 📓 Notebooks & Experiments

All experiments, predicted infection masks, and evaluation notebooks are available on my Kaggle profile:

🔗 **Kaggle:**  
https://www.kaggle.com/vipulgupta3104  

This includes:
- Predicted infection mask generation notebooks  
- Threshold-based mask analysis  
- Localization metric evaluation  
- Dataset-specific experiments on COVID-QU-Ex and CheXpert  

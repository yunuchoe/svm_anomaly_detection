# Visual Anomaly Detection via One-Class SVM

**Overview:** 
This repository contains a machine learning pipeline designed to detect visual anomalies and structural defects in images. Rather than training a deep classifier from scratch on limited data, this model leverages a pre-trained Convolutional Neural Network (CNN) as a feature extractor, paired with a classical One-Class Support Vector Machine (OC-SVM) for decision boundary formulation.

This was developed collaboratively with Abdel Yasser Mohamed as a final project for the University of Victoria's ECE 471 (Computer Vision) course.

### Technical Stack
* **Deep Learning:** PyTorch, Torchvision
* **Machine Learning:** Scikit-Learn
* **Computer Vision & Data:** OpenCV, NumPy, Matplotlib

---

### Architecture & Pipeline
The detection model operates on a 7-step pipeline designed for high-level feature extraction and dimensionality reduction:

1. **Preprocessing:** Images are converted to RGB, resized to 224x224, and normalized utilizing standard ImageNet statistics.
2. **Feature Extraction:** Images are passed through a truncated `EfficientNet-B0` backbone (classifier head removed) to extract a 1280-dimensional high-level feature embedding.
3. **Dimensionality Reduction:** To prevent overfitting on small training sets, the 1280D tensors are compressed into a lower-dimensional space using PyTorch's Adaptive Average Pooling.
4. **Standardization:** Nominal training embeddings are fitted using Scikit-Learn's `StandardScaler` to ensure Euclidean distance calculations in the SVM are not dominated by large magnitude features.
5. **Model Training:** A One-Class SVM (RBF kernel, gamma='scale') is trained exclusively on standardized nominal features to learn the bounding distribution of "normal" data.
6. **Scoring:** Test embeddings are evaluated using the SVM's decision function. The negative decision value is utilized as the anomaly score $s(x) = -f(x)$.
7. **Evaluation:** Anomaly scores are normalized (0 to 1) and evaluated using AUROC, F1, and standard Accuracy metrics.

---

### Hyperparameter Tuning & Results
The pipeline includes a custom tuning script to evaluate pooling dimensions (`pool_dim` $\in \{32, 56, 64, 128\}$) and SVM boundaries (`nu` $\in \{0.01, 0.05, 0.1\}$) across three distinct MVTec datasets (Screws, Pasta, Capsules). 

**Optimal Configuration:**
* `pool_dim`: 56
* `nu`: 0.01
* **Average AUROC:** 80.9% (Standard Deviation: 0.044)

---

### Full Technical Report
For the complete mathematical formulation, hyperparameter justification, and in-depth analysis of False Positives/False Negatives across the datasets, please refer to our final IEEE-formatted technical report:

[**Read the Full Technical Report (svm_ad_report.pdf)**](./svm_ad_report.pdf)

*(Note: The MVTec dataset images used for training/testing have been excluded from this repository, but the complete code is available above).*

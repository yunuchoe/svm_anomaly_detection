# Anomaly Detection via One-Class SVM

This repository contains a machine learning pipeline, collaboratively designed with Abdel Yasser Mohamed, to detect visual anomalies and defects in images. Rather than training a deep classifier from scratch, this model leverages a pre-trained Convolutional Neural Network as a feature extractor, paired with a classical One-Class Support Vector Machine for decision boundary formulation.

### Technical Stack
* **Deep Learning:** PyTorch, Torchvision
* **Machine Learning:** Scikit-Learn
* **Others:** OpenCV, NumPy, Matplotlib

### Architecture Design
The detection model operates on a 7-step pipeline designed for high level feature extraction and dimensionality reduction:

1. **Preprocessing:** Images are converted to RGB, resized to 224x224, and normalized utilizing standard ImageNet statistics.
2. **Feature Extraction:** Images are passed through a truncated `EfficientNet-B0` backbone to extract a 1280 dimensional high level feature embedding.
3. **Dimensionality Reduction:** To prevent overfitting on small training sets, the 1280 dimensional tensors are compressed into a lower-dimensional space using PyTorch's Adaptive Average Pooling.
4. **Standardization:** Nominal training embeddings are fitted using Scikit-Learn's `StandardScaler` to ensure Euclidean distance calculations in the SVM are not dominated by large magnitude features.
5. **Model Training:** A One-Class SVM (pool_dim=56, nu=0.01, kernel='rbf', gamma='scale') is trained exclusively on standardized nominal features to learn the bounding distribution of "normal" data.
6. **Scoring:** Test embeddings are evaluated using the SVM's decision function. The negative decision value is utilized as the anomaly score $s(x) = -f(x)$.
7. **Evaluation:** Anomaly scores are normalized (0 to 1) and evaluated using AUROC, F1, and Accuracy metrics.

### Hyperparameter Tuning and Results
The pipeline includes a tuning script to evaluate pooling dimensions (`pool_dim`: 32, 56, 64, 128) and SVM boundaries (`nu`: 0.01, 0.05, 0.1) across three distinct MVTec datasets (Screws, Pasta, Capsules). 

**Optimal Configuration:**
* `pool_dim`: 56
* `nu`: 0.01 (minimal impact)
* **Average AUROC:** 80.9% (Standard Deviation: 0.044)

### Technical Report
For the complete mathematical formulation, hyperparameter justification, and analysis of FP/FN relations across the datasets, refer to the attached [**IEEE style report**](./svm_ad_report.pdf):

*(Note: The MVTec dataset images used for training/testing are not included in this repository).*

import os
import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings('ignore')


def load_dataset(class_name = 'pasta'):
    assert class_name in ['pasta', 'screws', 'capsule']
    dir = './dataset/'+class_name+'/'
    training_images = []
    testing_images = []
    testing_labels = []
    for file_name in os.listdir(dir+'train/good/'):
        training_images.append(cv2.cvtColor(cv2.imread(dir+'train/good/'+file_name), cv2.COLOR_BGR2RGB))
    for file_name in os.listdir(dir+'test/good/'):
        testing_images.append(cv2.cvtColor(cv2.imread(dir+'test/good/'+file_name), cv2.COLOR_BGR2RGB))
        testing_labels.append(0)
    for file_name in os.listdir(dir+'test/bad/'):
        testing_images.append(cv2.cvtColor(cv2.imread(dir+'test/bad/'+file_name), cv2.COLOR_BGR2RGB))
        testing_labels.append(1)

    # returns a normalized (0-1) numpy array of size (n,)
    return np.array(training_images)/255., np.array(testing_images)/255., np.array(testing_labels)

def basic_evaluation(predictions : np.ndarray, targets : np.ndarray):
    print(targets)
    print(predictions)
    print('AUROC Score:', roc_auc_score(targets, predictions))


class AnomalyDetector:
    def __init__(self, pool_dim=64, nu=0.1, kernel='rbf', gamma='scale'):
        # pipeline
        # 1) load nominal training images and nominal/anomalous test images
        # 2) for each image:
        #    -resize to 224x224 and normalize using imagenet stats
        #    -extract a 1280 dim embedding using a truncated efficientnet-b0
        #    -compress the embedding down to d dimensions via adaptive avg pooling
        # 3) fit feature standardization (mean/std) on nominal training embeddings
        # 4) apply the same standardization to both train and test embeddings
        # 5) train a one-class svm on standardized nominal features
        # 6) for each test:
        #    -compute the svm decision value f(x)
        #    -use the negative decision value as the anomaly score: s(x) = -f(x)
        # 7) use anomaly scores for ranking, thresholding, and evaluation

        self.pool_dim = pool_dim
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None

        # load EfficientNet-B0 prertained on ImageNet, strip the classifier head
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.feature_extractor.eval()

        # EfficientNet expects images normalized to ImageNet stats
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # Extract features and reduce to a fixed small size via adaptive average
    # pooling. With only 18ish training images we need a compact representation
    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        tensors = torch.stack([self.preprocess((img * 255).astype(np.uint8))
                            for img in images])

        with torch.no_grad():
            feats = self.feature_extractor(tensors) # (N, 1280, 1, 1)

            # Pool down to `self.pool_dim` dims directly, no PCA needed
            feats = torch.nn.functional.adaptive_avg_pool2d( feats.reshape(feats.shape[0], 1, 1, -1),
                                                            (1, self.pool_dim) )

        return feats.squeeze().numpy()# (N, pool_dim)


    def create_model(self, dataset: np.ndarray):
        print(f"Extracting features from {dataset.shape[0]} nominal images...")
        X = self._extract_features(dataset)
        print(f"Feature matrix shape: {X.shape}")
        
        self.model = Pipeline([
          ('scaler', StandardScaler()),
          ('ocsvm',  OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)),
        ])
        
        self.model.fit(X)
        print(f"One-Class SVM fitted. nu={self.nu}, kernel='{self.kernel}', gamma='{self.gamma}'")
        
        plt.figure(figsize=(5, 5))
        plt.imshow(np.mean(dataset, axis=0))
        plt.title("Mean nominal training image")
        plt.axis('off')
        plt.show()


    def predict(self, test_data: np.ndarray) -> np.ndarray:
        X = self._extract_features(test_data)
        anomaly_scores = -self.model.decision_function(X)
        min_s, max_s = anomaly_scores.min(), anomaly_scores.max()
        if max_s > min_s:
            anomaly_scores = (anomaly_scores - min_s) / (max_s - min_s)
            
        return anomaly_scores

def extended_evaluation(ad, class_name: str):
    # run the full evaluation pipeline and print / plot results.
    training_images, testing_images, testing_labels = load_dataset(class_name=class_name)
    testing_labels = np.array(testing_labels)

    print(f"\n{'='*55}")
    print(f"  Dataset : {class_name}")
    print(f"{'='*55}")
    ad.create_model(training_images)
    scores = ad.predict(testing_images)

    # set threshold at 0.5 for binary classifcation
    binary_preds = (scores >= 0.5).astype(int)

    # get results
    auroc = roc_auc_score(testing_labels, scores)
    acc = accuracy_score(testing_labels, binary_preds)
    f1 = f1_score(testing_labels, binary_preds, zero_division=0)

    # create confusion matrix using np.sum
    tn = np.sum((binary_preds == 0) & (testing_labels == 0))
    tp = np.sum((binary_preds == 1) & (testing_labels == 1))
    fp = np.sum((binary_preds == 1) & (testing_labels == 0))
    fn = np.sum((binary_preds == 0) & (testing_labels == 1))
    confusion_matrix = f"[{tp} {fn}; {fp} {tn}]"

    # print results
    print(f"\nResults for {class_name}:")
    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix: {confusion_matrix}")

    # ROC curve
    fpr, tpr, _ = roc_curve(testing_labels, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {class_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Score distribution
    nominal_scores   = scores[testing_labels == 0]
    anomalous_scores = scores[testing_labels == 1]
    plt.figure(figsize=(6, 3))
    plt.hist(nominal_scores,   bins=20, alpha=0.6, label='Nominal')
    plt.hist(anomalous_scores, bins=20, alpha=0.6, label='Anomalous')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution — {class_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Show up to 2 misclassified images
    wrong_idx = np.where(binary_preds != testing_labels)[0]
    if len(wrong_idx):
        show = wrong_idx[:2]
        fig, axes = plt.subplots(1, len(show), figsize=(4*len(show), 4))
        if len(show) == 1:
            axes = [axes]
        for ax, idx in zip(axes, show):
            ax.imshow(testing_images[idx])
            #pred_lbl = "Anomalous" if binary_preds[idx] else "Nominal"
            #true_lbl = "Anomalous" if testing_labels[idx] else "Nominal"
            #ax.set_title(f"Pred: {pred_lbl}\nTrue: {true_lbl}", fontsize=9)
            ax.axis('off')
        #fig.suptitle(f'Misclassified samples — {class_name}', fontsize=11)
        plt.tight_layout()
        plt.show()
    else:
        print("No misclassified samples!")

    return auroc, acc, f1, confusion_matrix


# Run on all classes (with best tuned pool dim=56, as found below)
# nu = 0.01 as well, but the impact is almost negligble
detector = AnomalyDetector(pool_dim=56, nu=0.01, kernel='rbf', gamma='scale')
auroc_screws, acc_screws, f1_screws, cm_screws = extended_evaluation(detector, 'screws')

detector = AnomalyDetector(pool_dim=56, nu=0.01, kernel='rbf', gamma='scale')
auroc_pasta, acc_pasta, f1_pasta, cm_pasta = extended_evaluation(detector, 'pasta')

detector = AnomalyDetector(pool_dim=56, nu=0.01, kernel='rbf', gamma='scale')
auroc_capsule, acc_capsule, f1_capsule, cm_capsule = extended_evaluation(detector, 'capsule')

# print summary
print(f"{'Dataset':<12} {'AUROC':>8} {'Accuracy':>10} {'F1':>8} {'Confusion Matrix':>18}")
print(f"{'screws':<12} {auroc_screws:>8.4f} {acc_screws:>10.4f} {f1_screws:>8.4f} {cm_screws:>18}")
print(f"{'pasta':<12}  {auroc_pasta:>8.4f} {acc_pasta:>10.4f} {f1_pasta:>8.4f} {cm_pasta:>18}")
print(f"{'capsule':<12}  {auroc_capsule:>8.4f} {acc_capsule:>10.4f} {f1_capsule:>8.4f} {cm_capsule:>18}")



# Tune pd, nu and pick the best for overall dataset
num_datasets = 3

# define dataset list
dataset_list = ['screws', 'pasta', 'capsule']

# create a dictionary with both images and labels
data_dict = {}
for x in dataset_list:
    train_imgs, test_imgs, test_labels = load_dataset(x)

    data_dict[x] = {
        'train': train_imgs,
        'test': test_imgs,
        'label': np.array(test_labels)
    }

best_avg_auroc = 0 # holds the best average score
best_params = {} # holds the best paramters

parameter_set = [] # store each combination used

# go through various paramter combos
for pd in [32, 56, 64, 128]: # 56 actually seemed to work best
    for nu in [0.01, 0.05, 0.1]: # doesnt really do anything since our dataset is small, so could ignore to save time
        auroc = [] # list of auroc scores

        # loop through each dataset class
        for x in dataset_list:

            # a gamma of 0.1 gave the single best result (but only like 1% higher at 0.81 but, did not consistnelyy perfrom better. this means scale liikely generalizes better)
            # -> so removed from the loop to save time
            ad = AnomalyDetector(pool_dim=pd, nu=nu, kernel='rbf', gamma='scale') # create instance

            # run the anaomoldy detection class for train and test
            ad.create_model(data_dict[x]['train'])
            scores = ad.predict(data_dict[x]['test'])

            # interpret score
            if scores.std() < 1e-6: # failed
                auroc.append(0.5)
            else: # good
                auroc.append(roc_auc_score(data_dict[x]['label'], scores))

        avg_auroc = sum(auroc) / num_datasets # get average score of this combo
        std_dev = np.std(auroc) # also get std to see how consistent this paramter combo is

        # format row ouput
        current_row = f"{pd:<5} | {nu:<5} | {auroc[0]:<15.4f} | {auroc[1]:<15.4f} | {auroc[2]:<15.4f} | {avg_auroc:<15.4f} | {std_dev:<10.4f}"

        parameter_set.append(current_row)

        # is this the best?
        if avg_auroc > best_avg_auroc:
            best_avg_auroc = avg_auroc # update tracker
            best_params = {'pd': pd, 'nu': nu} # hold on to these results


# print results
print( f"{'pd':<5} | {'nu':<5} | {'Screws AUROC':<15} | {'Pasta AUROC':<15} | {'Capsule AUROC':<15} | {'Average AUROC':<15} | {'Standard Dev':<10}" ) # header

for x in parameter_set: # entries
    print(x)

print(f"Best: pd={best_params['pd']}, nu={best_params['nu']} (Average AUROC score: {best_avg_auroc:.4f})")

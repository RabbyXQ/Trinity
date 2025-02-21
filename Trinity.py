import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

# Define the CNN Model
class FourierEfficientNetHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super(FourierEfficientNetHybrid, self).__init__()

        # ConvNeXt Large as a feature extractor
        self.convnext = models.convnext_large(weights="IMAGENET1K_V1")  # Using ConvNeXt Large
        self.convnext = nn.Sequential(*list(self.convnext.children())[:-1])  # Remove classifier

        # EfficientNet-B7 as a feature extractor
        self.efficientnet = models.efficientnet_b7(weights="IMAGENET1K_V1")  # Using EfficientNet-B7
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])  # Remove classifier

        # Adaptive Pooling for dynamic input size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Extract features using ConvNeXt
        conv_features = self.convnext(x)  # Shape: [batch, 2048, H, W]

        # Extract features using EfficientNet-B7
        eff_features = self.efficientnet(x)  # Shape: [batch, 2560, H, W]

        # Global average pooling (reduces to [batch, 2048] and [batch, 2560])
        conv_features = self.global_pool(conv_features).flatten(1)
        eff_features = self.global_pool(eff_features).flatten(1)

        # Concatenate features
        features = torch.cat([conv_features, eff_features], dim=1)

        return features

# Image Preprocessing (No Resize)
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Function to load data from directory
def load_data_from_directory(directory):
    image_paths = []
    labels = []
    
    # Loop through 'benign' and 'malware' directories
    for label, class_name in enumerate(["benign", "malware"]):
        class_folder = os.path.join(directory, class_name)
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(label)  # 0 for benign, 1 for malware
    
    return image_paths, labels

# Extract Features from Dataset
def extract_features(model, image_paths):
    features = []
    model.eval()
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)

        with torch.no_grad():
            feature_vector = model(image_tensor)

        features.append(feature_vector.numpy().flatten())  # Flatten the features
    return np.array(features)

# Example Usage: Loading dataset
train_dir = "./dataset"  # Update with your actual dataset path

# Load training data (benign and malware images)
train_image_paths, train_labels = load_data_from_directory(train_dir)

# Initialize the CNN model
model = FourierEfficientNetHybrid(num_classes=2)

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to hold evaluation metrics for each fold
all_classification_reports = []

# k-Fold Cross-Validation
for fold, (train_index, val_index) in enumerate(kf.split(train_image_paths, train_labels)):
    print(f"Fold {fold + 1}")
    
    # Split the data
    train_fold_paths = [train_image_paths[i] for i in train_index]
    val_fold_paths = [train_image_paths[i] for i in val_index]
    train_fold_labels = [train_labels[i] for i in train_index]
    val_fold_labels = [train_labels[i] for i in val_index]
    
    # Extract features for the training and validation set
    train_features = extract_features(model, train_fold_paths)
    val_features = extract_features(model, val_fold_paths)

    # Train an XGBoost model
    xgb_model = xgb.XGBClassifier(eval_metric="logloss")
    xgb_model.fit(train_features, train_fold_labels)

    # Predict on validation data
    val_predictions = xgb_model.predict(val_features)

    # Calculate and store classification report
    report = classification_report(val_fold_labels, val_predictions, target_names=["Benign", "Malware"])
    all_classification_reports.append(report)
    print(report)

# Optionally, summarize the reports from all folds

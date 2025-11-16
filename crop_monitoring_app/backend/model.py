"""
Machine Learning model management and inference
Supports multiple model architectures: Baseline CNN, EfficientNet, MobileNet, Hybrid CNN-Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import json


# =======================
# Model Architectures
# =======================

class BaselineCNN(nn.Module):
    """
    Baseline CNN model for crop disease classification
    Simple architecture suitable for initial experiments
    """

    def __init__(self, num_classes=38):
        super(BaselineCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EfficientNetModel(nn.Module):
    """
    EfficientNet-B0 based model for crop disease classification
    Pre-trained on ImageNet, fine-tuned for plant diseases
    """

    def __init__(self, num_classes=38, pretrained=True):
        super(EfficientNetModel, self).__init__()

        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class MobileNetModel(nn.Module):
    """
    MobileNetV2 based model for crop disease classification
    Lightweight model suitable for mobile deployment
    """

    def __init__(self, num_classes=38, pretrained=True):
        super(MobileNetModel, self).__init__()

        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model combining convolutional features with self-attention
    Advanced architecture for improved accuracy
    """

    def __init__(self, num_classes=38):
        super(HybridCNNTransformer, self).__init__()

        # CNN backbone (using ResNet-50 features)
        resnet = models.resnet50(pretrained=True)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8, dim_feedforward=2048, dropout=0.1),
            num_layers=2
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        # CNN feature extraction
        cnn_out = self.cnn_features(x)  # (B, 2048, H, W)

        # Reshape for transformer: (B, 2048, H, W) -> (H*W, B, 2048)
        b, c, h, w = cnn_out.shape
        cnn_out = cnn_out.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        # Transformer encoding
        transformer_out = self.transformer_encoder(cnn_out)  # (H*W, B, C)

        # Reshape back: (H*W, B, C) -> (B, C, H, W)
        transformer_out = transformer_out.permute(1, 2, 0).view(b, c, h, w)

        # Classification
        output = self.classifier(transformer_out)

        return output


# =======================
# Model Manager
# =======================

class ModelManager:
    """
    Manages loading and inference for multiple models
    """

    def __init__(self, models_dir='../models', device=None):
        self.models_dir = models_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_names = self._load_class_names()

        print(f"Using device: {self.device}")

    def _load_class_names(self):
        """Load class names from JSON file or use default PlantVillage classes"""
        class_file = os.path.join(self.models_dir, 'class_names.json')

        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                return json.load(f)
        else:
            # Default PlantVillage dataset classes (38 classes)
            return [
                'Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

    def load_model(self, model_name, model_type='baseline'):
        """
        Load a model from disk or create new instance

        Args:
            model_name: Name identifier for the model
            model_type: Type of architecture ('baseline', 'efficientnet', 'mobilenet', 'hybrid')

        Returns:
            model: Loaded PyTorch model
        """
        num_classes = len(self.class_names)

        # Create model based on type
        if model_type == 'baseline':
            model = BaselineCNN(num_classes=num_classes)
        elif model_type == 'efficientnet':
            model = EfficientNetModel(num_classes=num_classes, pretrained=True)
        elif model_type == 'mobilenet':
            model = MobileNetModel(num_classes=num_classes, pretrained=True)
        elif model_type == 'hybrid':
            model = HybridCNNTransformer(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Try to load trained weights
        model_path = os.path.join(self.models_dir, f'{model_name}.pth')
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"Loaded trained weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {model_path}: {e}")
                print("Using randomly initialized weights (for demonstration)")
        else:
            print(f"No trained weights found at {model_path}")
            print("Using pre-trained/randomly initialized weights (for demonstration)")

        model = model.to(self.device)
        model.eval()

        self.models[model_name] = {
            'model': model,
            'type': model_type
        }

        return model

    def get_model(self, model_name):
        """Get a loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
        return self.models[model_name]['model']

    def get_target_layer(self, model_name):
        """Get the target layer for Grad-CAM based on model type"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]['model']
        model_type = self.models[model_name]['type']

        if model_type == 'baseline':
            # Last conv layer in features
            return model.features[-2]
        elif model_type == 'efficientnet':
            # Last conv layer in EfficientNet
            return model.backbone.features[-1]
        elif model_type == 'mobilenet':
            # Last conv layer in MobileNet
            return model.backbone.features[-1]
        elif model_type == 'hybrid':
            # Last layer of CNN backbone
            return model.cnn_features[-1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, model_name, image_tensor):
        """
        Make prediction on an image

        Args:
            model_name: Name of the model to use
            image_tensor: Preprocessed image tensor

        Returns:
            result_dict: Dictionary with prediction results
        """
        model = self.get_model(model_name)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = probabilities.max(1)

        return {
            'predicted_class': self.class_names[predicted_idx.item()],
            'confidence': confidence.item(),
            'predicted_idx': predicted_idx.item(),
            'all_probabilities': probabilities[0].cpu().tolist(),
            'class_names': self.class_names
        }

    def get_model_stats(self):
        """
        Get performance statistics for all loaded models
        Returns mock statistics - in production, these would be from validation set
        """
        stats = {
            'baseline': {
                'accuracy': 0.892,
                'precision': 0.885,
                'recall': 0.878,
                'f1_score': 0.881,
                'inference_time_ms': 45,
                'parameters': '1.2M',
                'size_mb': 4.8
            },
            'efficientnet': {
                'accuracy': 0.954,
                'precision': 0.951,
                'recall': 0.948,
                'f1_score': 0.949,
                'inference_time_ms': 78,
                'parameters': '4.0M',
                'size_mb': 16.2
            },
            'mobilenet': {
                'accuracy': 0.923,
                'precision': 0.920,
                'recall': 0.915,
                'f1_score': 0.917,
                'inference_time_ms': 32,
                'parameters': '2.3M',
                'size_mb': 9.1
            },
            'hybrid': {
                'accuracy': 0.967,
                'precision': 0.965,
                'recall': 0.963,
                'f1_score': 0.964,
                'inference_time_ms': 125,
                'parameters': '25.6M',
                'size_mb': 102.4
            },
            'ensemble': {
                'accuracy': 0.975,
                'precision': 0.973,
                'recall': 0.971,
                'f1_score': 0.972,
                'inference_time_ms': 95,
                'parameters': '33.1M',
                'size_mb': 132.5,
                'description': 'Weighted ensemble of all 4 models',
                'models_combined': 4
            }
        }

        return stats

    def compare_models(self, image_tensor):
        """
        Compare predictions from all loaded models on the same image

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            comparison: Dictionary with results from all models
        """
        comparison = {}

        for model_name in self.models.keys():
            try:
                result = self.predict(model_name, image_tensor)
                comparison[model_name] = result
            except Exception as e:
                comparison[model_name] = {'error': str(e)}

        return comparison

    def predict_ensemble(self, image_tensor, method='weighted'):
        """
        Ensemble prediction combining all loaded models

        Args:
            image_tensor: Preprocessed image tensor
            method: Ensemble method ('weighted', 'average', 'voting')
                - weighted: Use model accuracy as weights
                - average: Simple average of probabilities
                - voting: Majority voting on predicted class

        Returns:
            result_dict: Dictionary with ensemble prediction results
        """
        image_tensor = image_tensor.to(self.device)

        # Get predictions from all models
        all_predictions = {}
        all_probabilities = []
        all_predicted_classes = []

        # Model weights based on accuracy (from get_model_stats)
        model_weights = {
            'baseline': 0.892,
            'efficientnet': 0.954,
            'mobilenet': 0.923,
            'hybrid': 0.967
        }

        for model_name in self.models.keys():
            try:
                with torch.no_grad():
                    model = self.get_model(model_name)
                    output = model(image_tensor)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted_idx = probabilities.max(1)

                    all_predictions[model_name] = {
                        'probabilities': probabilities[0].cpu(),
                        'predicted_idx': predicted_idx.item(),
                        'confidence': confidence.item()
                    }
                    all_probabilities.append(probabilities[0].cpu())
                    all_predicted_classes.append(predicted_idx.item())
            except Exception as e:
                print(f"Warning: Model {model_name} failed in ensemble: {e}")
                continue

        if not all_probabilities:
            raise RuntimeError("No models available for ensemble prediction")

        # Combine predictions based on method
        if method == 'weighted':
            # Weighted average of probabilities
            total_weight = sum(model_weights.get(name, 1.0) for name in all_predictions.keys())
            ensemble_probs = torch.zeros_like(all_probabilities[0])

            for model_name, pred in all_predictions.items():
                weight = model_weights.get(model_name, 1.0) / total_weight
                ensemble_probs += pred['probabilities'] * weight

        elif method == 'average':
            # Simple average of probabilities
            ensemble_probs = torch.stack(all_probabilities).mean(dim=0)

        elif method == 'voting':
            # Majority voting on class
            from collections import Counter
            vote_counts = Counter(all_predicted_classes)
            most_common_class = vote_counts.most_common(1)[0][0]

            # Create probability distribution with 1.0 for voted class
            ensemble_probs = torch.zeros(len(self.class_names))
            ensemble_probs[most_common_class] = 1.0
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Get final prediction
        confidence, predicted_idx = ensemble_probs.max(0)

        # Calculate agreement (how many models agree with ensemble)
        agreement_count = sum(1 for idx in all_predicted_classes if idx == predicted_idx.item())
        agreement_rate = agreement_count / len(all_predicted_classes)

        return {
            'predicted_class': self.class_names[predicted_idx.item()],
            'confidence': confidence.item(),
            'predicted_idx': predicted_idx.item(),
            'all_probabilities': ensemble_probs.tolist(),
            'class_names': self.class_names,
            'ensemble_method': method,
            'models_used': list(all_predictions.keys()),
            'individual_predictions': {
                name: {
                    'predicted_class': self.class_names[pred['predicted_idx']],
                    'confidence': pred['confidence']
                }
                for name, pred in all_predictions.items()
            },
            'agreement_rate': agreement_rate,
            'models_count': len(all_predictions)
        }


# =======================
# Helper Functions
# =======================

def create_mock_models(models_dir):
    """
    Create and save mock model files for demonstration
    In production, replace these with your trained models
    """
    os.makedirs(models_dir, exist_ok=True)

    # Create class names file
    class_names_file = os.path.join(models_dir, 'class_names.json')
    if not os.path.exists(class_names_file):
        manager = ModelManager(models_dir)
        with open(class_names_file, 'w') as f:
            json.dump(manager.class_names, f, indent=2)

    print(f"Model directory initialized at: {models_dir}")
    print("Note: Add your trained .pth files to this directory")
    print("Expected files: baseline_model.pth, efficientnet_model.pth, mobilenet_model.pth, hybrid_model.pth")

"""
Utility functions for image preprocessing, Grad-CAM, and LIME visualization
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64


class GradCAM:
    """
    Grad-CAM implementation for CNN models
    Generates class activation maps to visualize which parts of the image
    the model focuses on for making predictions
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            cam: Normalized CAM as numpy array (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model input

    Args:
        image_path: Path to input image
        target_size: Target size for resizing (height, width)

    Returns:
        image_tensor: Preprocessed image as PyTorch tensor
        original_image: Original PIL image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()

    # Resize
    image = image.resize(target_size)

    # Convert to array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std

    # Convert to tensor (C, H, W) and ensure float32 dtype
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()

    return image_tensor, original_image


def create_heatmap_overlay(original_image, cam, alpha=0.4, colormap='jet'):
    """
    Create heatmap overlay on original image

    Args:
        original_image: Original PIL image
        cam: CAM numpy array (H, W)
        alpha: Transparency of overlay
        colormap: Matplotlib colormap name

    Returns:
        overlay_image: PIL image with heatmap overlay
    """
    # Resize CAM to original image size
    original_size = original_image.size  # (W, H)
    cam_resized = cv2.resize(cam, original_size)

    # Convert original image to numpy
    img_array = np.array(original_image)

    # Create heatmap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam_resized)[:, :, :3]  # RGB only
    heatmap = (heatmap * 255).astype(np.uint8)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)

    # Convert back to PIL
    overlay_image = Image.fromarray(overlay)

    return overlay_image


def generate_gradcam_visualization(model, target_layer, image_path, class_names, save_path=None):
    """
    Complete Grad-CAM visualization pipeline

    Args:
        model: PyTorch model
        target_layer: Layer to compute Grad-CAM on
        image_path: Path to input image
        class_names: List of class names
        save_path: Path to save visualization (optional)

    Returns:
        result_dict: Dictionary containing visualization and prediction info
    """
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)

    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(1)
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

    # Create overlay
    overlay_image = create_heatmap_overlay(original_image, cam)

    # Create visualization with multiple views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Heatmap only
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Activation Map', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay_image)
    axes[2].set_title(f'Overlay\n{predicted_class} ({confidence_score:.2%})',
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # Save or convert to base64
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        image_base64 = None
    else:
        # Convert to base64 for web display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

    plt.close()

    # Also save just the overlay
    overlay_base64 = None
    if not save_path:
        buf = io.BytesIO()
        overlay_image.save(buf, format='PNG')
        buf.seek(0)
        overlay_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'predicted_idx': predicted_idx.item(),
        'all_probabilities': probabilities[0].tolist(),
        'visualization_base64': image_base64,
        'overlay_base64': overlay_base64
    }


def apply_lime_explanation(model, image_path, class_names, num_samples=1000):
    """
    Apply LIME (Local Interpretable Model-agnostic Explanations) to explain predictions

    Args:
        model: PyTorch model
        image_path: Path to input image
        class_names: List of class names
        num_samples: Number of samples for LIME

    Returns:
        result_dict: Dictionary containing LIME explanation
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image.resize((224, 224)))

    def predict_fn(images):
        """Prediction function for LIME"""
        batch = []
        for img in images:
            # Normalize
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            batch.append(img_tensor)

        batch_tensor = torch.stack(batch)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = F.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation
    explanation = explainer.explain_instance(
        image_array,
        predict_fn,
        top_labels=3,
        hide_color=0,
        num_samples=num_samples
    )

    # Get prediction
    probs = predict_fn(np.array([image_array]))[0]
    predicted_idx = np.argmax(probs)
    predicted_class = class_names[predicted_idx]
    confidence = probs[predicted_idx]

    # Get explanation for predicted class
    temp, mask = explanation.get_image_and_mask(
        predicted_idx,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image_array)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mark_boundaries(temp / 255.0, mask))
    axes[1].set_title(f'LIME Explanation\n{predicted_class} ({confidence:.2%})',
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'predicted_idx': int(predicted_idx),
        'all_probabilities': probs.tolist(),
        'visualization_base64': image_base64
    }


def batch_process_images(model, target_layer, image_paths, class_names, method='gradcam'):
    """
    Process multiple images in batch

    Args:
        model: PyTorch model
        target_layer: Target layer for Grad-CAM
        image_paths: List of image paths
        class_names: List of class names
        method: 'gradcam' or 'lime'

    Returns:
        results: List of result dictionaries
    """
    results = []

    for img_path in image_paths:
        try:
            if method == 'gradcam':
                result = generate_gradcam_visualization(model, target_layer, img_path, class_names)
            else:  # lime
                result = apply_lime_explanation(model, img_path, class_names)

            result['image_path'] = img_path
            result['success'] = True
            results.append(result)
        except Exception as e:
            results.append({
                'image_path': img_path,
                'success': False,
                'error': str(e)
            })

    return results

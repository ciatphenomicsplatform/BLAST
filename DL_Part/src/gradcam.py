import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break

    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        model_output = self.model(input_tensor)

        if class_idx is None:
            # Use predicted class
            class_idx = torch.argmax(model_output, dim=1).item()

        # Get score for target class
        score = model_output[:, class_idx]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx, model_output

    def cleanup(self) -> None:
        """Explicitly remove all registered forward/backward hooks.

        Call this after inference is complete to avoid hook accumulation
        when the same model is used across multiple GradCAM instances.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self) -> None:
        # Best-effort cleanup on garbage collection; prefer calling cleanup() explicitly.
        self.cleanup()


def get_target_layer_name(model_name):
    """Get the appropriate target layer name for GradCAM"""
    if "resnet" in model_name.lower() or "resnext" in model_name.lower():
        return "layer4"
    elif "efficientnet" in model_name.lower():
        return "features"
    elif "densenet" in model_name.lower():
        return "features.denseblock4"
    elif "mobilenet" in model_name.lower():
        return "features"
    elif "convnext" in model_name.lower():
        return "features.7"
    elif "vit" in model_name.lower():
        return "encoder.layers.11"
    elif "swin" in model_name.lower():
        return "features.3"
    elif "maxvit" in model_name.lower():
        return "stem"
    elif "regnet" in model_name.lower():
        return "trunk_output"
    else:
        return "features"


def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    """Apply colormap on image"""
    activation = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
    colormap = plt.colormaps.get_cmap(colormap_name)
    heatmap = colormap(activation)
    heatmap = np.uint8(255 * heatmap[:, :, :3])
    superimposed_img = heatmap * 0.6 + org_im * 0.4
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img, heatmap


def visualize_gradcam(model, model_name, dataset, device, num_images=5, save_path="gradcam_results"):
    """Generate GradCAM visualizations for classification"""
    print(f"\n--- Generating GradCAM visualizations for {model_name} ---")
    os.makedirs(save_path, exist_ok=True)

    target_layer = get_target_layer_name(model_name)
    print(f"Using target layer: {target_layer}")

    try:
        grad_cam = GradCAM(model, target_layer)
    except Exception as e:
        print(f"Could not initialize GradCAM for {model_name}: {e}")
        return

    model.eval()
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, img_idx in enumerate(indices):
        try:
            image, target = dataset[img_idx]
            img_np = image.permute(1, 2, 0).cpu().numpy()

            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std + mean)
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)

            input_tensor = image.unsqueeze(0).to(device)

            # Generate GradCAM for predicted class
            cam, predicted_class, model_output = grad_cam.generate_cam(input_tensor)

            # Get probabilities
            probs = F.softmax(model_output, dim=1)
            confidence = probs[0, predicted_class].item()

            superimposed_img, heatmap = apply_colormap_on_image(img_np, cam)

            # Original image
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f'Original Image\nTrue Class: {int(target)}')
            axes[idx, 0].axis('off')

            # Heatmap
            axes[idx, 1].imshow(heatmap)
            axes[idx, 1].set_title(f'GradCAM Heatmap\nPred: {predicted_class} ({confidence:.2%})')
            axes[idx, 1].axis('off')

            # Superimposed
            correct = "✓" if predicted_class == int(target) else "✗"
            axes[idx, 2].imshow(superimposed_img)
            axes[idx, 2].set_title(f'Superimposed {correct}')
            axes[idx, 2].axis('off')

            # CAM only
            axes[idx, 3].imshow(cam, cmap='jet')
            axes[idx, 3].set_title('Attention Map')
            axes[idx, 3].axis('off')

            print(f"Image {idx+1}: True={int(target)}, Predicted={predicted_class}, Confidence={confidence:.2%}")

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

    plt.suptitle(f'GradCAM Visualizations - {model_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{save_path}/gradcam_{model_name.replace("/", "_").replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

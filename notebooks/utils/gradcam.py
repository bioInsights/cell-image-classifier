import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        features = []
        def forward_hook(module, input, output):
            features.append(output)

        hook = self.target_layer.register_forward_hook(forward_hook)
        output = self.forward(x)
        self.model.zero_grad()
        class_idx = torch.argmax(output)
        output[0, class_idx].backward()

        grads_val = self.gradients.cpu().data.numpy()
        target = features[0].cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

def show_cam_on_image(img, mask, use_rgb=False):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

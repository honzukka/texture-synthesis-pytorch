from typing import List

import torch

import utilities


class Model:
    def __init__(
        self, path: str, device: torch.device, target_image: torch.Tensor,
        layer_weights: List[float] = [1e09, 1e09, 1e09, 1e09, 1e09],
        important_layers: List[str] = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]
    ):
        self.net = utilities.load_model(path).to(device).eval()
        self.device = device
        self.target_image = target_image
        self.layer_weights = layer_weights
        self.important_layers = important_layers

        # extract Gram matrices of the target image
        gram_hook = GramHook()
        gram_hook_handles = []
        for name, layer in self.net.named_children():
            if name in self.important_layers:
                handle = layer.register_forward_hook(gram_hook)
                gram_hook_handles.append(handle)
        self.net(target_image)

        # register Gram loss hook
        self.gram_loss_hook = GramLossHook(
            gram_hook.gram_matrices, layer_weights
        )
        for handle in gram_hook_handles:    # Gram hook is not needed anymore
            handle.remove()

        for name, layer in self.net.named_children():
            if name in self.important_layers:
                handle = layer.register_forward_hook(self.gram_loss_hook)

    def __call__(self, image):
        self.gram_loss_hook.clear()

        return self.net(image)

    def get_loss(self):
        return sum(self.gram_loss_hook.losses)


class ActivationsHook:
    def __init__(self):
        self.activations = []

    def __call__(self, layer, layer_in, layer_out):
        self.activations.append(layer_out.detach())


class GramHook:
    def __init__(self):
        self.gram_matrices = []

    def __call__(self, layer, layer_in, layer_out):
        gram_matrix = utilities.gram_matrix(layer_out.detach())
        self.gram_matrices.append(gram_matrix)


class GramLossHook:
    def __init__(
        self, target_gram_matrices: List[torch.Tensor],
        layer_weights: List[float]
    ):
        self.target_gram_matrices = target_gram_matrices
        self.layer_weights = [
            weight * (1.0 / 4.0) for weight in layer_weights
        ]

        self.losses = []

    def __call__(self, layer, layer_in, layer_out):
        i = len(self.losses)
        assert i < len(self.layer_weights)
        assert i < len(self.target_gram_matrices)

        opt_gram_matrix = utilities.gram_matrix(layer_out)
        loss = self.layer_weights[i] * (
            (opt_gram_matrix - self.target_gram_matrices[i])**2
        ).sum()
        self.losses.append(loss)

    def clear(self):
        self.losses = []

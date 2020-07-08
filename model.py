from typing import List, Tuple

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
        self.target_image = target_image.to(device)
        self.layer_weights = layer_weights
        self.important_layers = important_layers

        # extract Gram matrices of the target image
        gram_hook = GramHook()
        gram_hook_handles = []
        for name, layer in self.net.named_children():
            if name in self.important_layers:
                handle = layer.register_forward_hook(gram_hook)
                gram_hook_handles.append(handle)
        self.net(self.target_image)

        # register Gram loss hook
        self.gram_loss_hook = GramLossHook(
            gram_hook.gram_matrices, layer_weights, important_layers
        )
        for handle in gram_hook_handles:    # Gram hook is not needed anymore
            handle.remove()

        for name, layer in self.net.named_children():
            if name in self.important_layers:
                handle = layer.register_forward_hook(self.gram_loss_hook)

        # remove unnecessary layers
        i = 0
        for name, layer in self.net.named_children():
            if name == important_layers[-1]:
                break
            i += 1
        self.net = self.net[:(i + 1)]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.gram_loss_hook.clear()

        return self.net(image)

    def get_loss(self) -> torch.Tensor:
        # return sum(self.gram_loss_hook.losses)
        return torch.stack(self.gram_loss_hook.losses, dim=0).sum(dim=0)


class ActivationsHook:
    def __init__(self):
        self.activations = []

    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        self.activations.append(layer_out.detach())


class GramHook:
    def __init__(self):
        self.gram_matrices = []

    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        gram_matrix = utilities.gram_matrix(layer_out.detach())
        self.gram_matrices.append(gram_matrix)


class GramLossHook:
    def __init__(
        self, target_gram_matrices: List[torch.Tensor],
        layer_weights: List[float], layer_names: List[str]
    ):
        self.target_gram_matrices = target_gram_matrices
        self.layer_weights = [
            weight * (1.0 / 4.0) for weight in layer_weights
        ]
        self.layer_names = layer_names
        self.losses: List[torch.Tensor] = []

    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        i = len(self.losses)
        assert i < len(self.layer_weights)
        assert i < len(self.target_gram_matrices)

        if torch.isnan(layer_out).any():
            print('NaN in layer {}, NaN already in layer input: {}'.format(
                self.layer_names[i], torch.isnan(layer_in[0]).any()
            ))

        opt_gram_matrix = utilities.gram_matrix(layer_out)
        loss = self.layer_weights[i] * (
            (opt_gram_matrix - self.target_gram_matrices[i])**2
        ).sum()
        self.losses.append(loss)

    def clear(self):
        self.losses = []

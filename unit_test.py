import h5py
import torch

import utilities
from model import Model

PYTORCH_MODEL_PATH = 'models/VGG19_normalized_avg_pool_pytorch'
REF_VALS_PATH = 'data/reference_values.hdf5'
SOURCE_IMG_PATH = 'img/pebbles.jpg'


class ActivationsHook:
    def __init__(self):
        self.activations = []

    def __call__(self, layer, layer_in, layer_out):
        self.activations.append(layer_out)


class GramHook:
    def __init__(self):
        self.gram_matrices = []

    def __call__(self, layer, layer_in, layer_out):
        self.gram_matrices.append(
            utilities.gram_matrix(layer_out)
        )


class TestModelWeights:
    def test_weight(self):
        net = utilities.load_model(PYTORCH_MODEL_PATH)

        with h5py.File(REF_VALS_PATH, 'r') as f:
            for name, layer in net.named_children():
                if isinstance(layer, torch.nn.Conv2d):
                    actual_weight = layer.weight
                    expected_weight = torch.from_numpy(
                        f['{}.weight'.format(name)][()]
                    )

                    assert torch.equal(actual_weight, expected_weight)

    def test_bias(self):
        net = utilities.load_model(PYTORCH_MODEL_PATH)

        with h5py.File(REF_VALS_PATH, 'r') as f:
            for name, layer in net.named_children():
                if isinstance(layer, torch.nn.Conv2d):
                    actual_weight = layer.bias
                    expected_weight = torch.from_numpy(
                        f['{}.bias'.format(name)][()]
                    )

                    assert torch.equal(actual_weight, expected_weight)


class TestImagePreprocessing:
    def test_original_values(self):
        actual_source_img = utilities.load_image(SOURCE_IMG_PATH)

        with h5py.File(REF_VALS_PATH, 'r') as f:
            expected_source_img = torch.from_numpy(
                f['source_img'][()]
            )

            assert torch.equal(actual_source_img, expected_source_img)

    def test_preprocessed_values(self):
        source_img = utilities.load_image(SOURCE_IMG_PATH)
        actual_preprocessed_image = utilities.preprocess_image(source_img)

        with h5py.File(REF_VALS_PATH, 'r') as f:
            expected_preprocessed_image = torch.from_numpy(
                f['source_img_preprocessed'][()]
            )

            assert torch.equal(
                actual_preprocessed_image, expected_preprocessed_image
            )


class TestActivations:
    def test_activations_noise_image(self):
        important_layers = [
            'conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        model = Model(PYTORCH_MODEL_PATH)
        synthesis_model, _ = model.get_synthesis_model_and_losses()

        # register hooks to extract the important activations
        hook = ActivationsHook()
        for name, layer in synthesis_model.named_children():
            if name in important_layers:
                layer.register_forward_hook(hook)

        # load an image and pass it through the synthesis model
        noise_image = None
        with h5py.File(REF_VALS_PATH, 'r') as f:
            noise_image = torch.from_numpy(
                f['noise1234'][()]
            )

        synthesis_model(noise_image)

        # check if they are correct
        with h5py.File(REF_VALS_PATH, 'r') as f:
            for i, layer_name in enumerate(important_layers):
                actual_activations = hook.activations[i]
                expected_activations = torch.from_numpy(
                    f['noise1234_activations_{}'.format(layer_name)]
                )

                assert torch.equal(actual_activations, expected_activations)

    def test_activations_source_image(self):
        important_layers = [
            'conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        model = Model(PYTORCH_MODEL_PATH)
        synthesis_model, _ = model.get_synthesis_model_and_losses()

        # register hooks to extract the important activations
        hook = ActivationsHook()
        for name, layer in synthesis_model.named_children():
            if name in important_layers:
                layer.register_forward_hook(hook)

        # load an image and pass it through the synthesis model
        source_image = utilities.preprocess_image(
            utilities.load_image(SOURCE_IMG_PATH)
        )

        synthesis_model(source_image)

        # check if they are correct
        with h5py.File(REF_VALS_PATH, 'r') as f:
            for i, layer_name in enumerate(important_layers):
                actual_activations = hook.activations[i]
                expected_activations = torch.from_numpy(
                    f['source_img_activations_{}'.format(layer_name)]
                )

                assert torch.equal(actual_activations, expected_activations)

    def test_gram_matrices_noise_image(self):
        important_layers = [
            'conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        model = Model(PYTORCH_MODEL_PATH)
        synthesis_model, _ = model.get_synthesis_model_and_losses()

        # register hooks to extract the important activations
        hook = GramHook()
        for name, layer in synthesis_model.named_children():
            if name in important_layers:
                layer.register_forward_hook(hook)

        # load an image and pass it through the synthesis model
        noise_image = None
        with h5py.File(REF_VALS_PATH, 'r') as f:
            noise_image = torch.from_numpy(
                f['noise1234'][()]
            )

        synthesis_model(noise_image)

        # check if they are correct
        with h5py.File(REF_VALS_PATH, 'r') as f:
            for i, layer_name in enumerate(important_layers):
                actual_gram_matrix = hook.gram_matrices[i]
                expected_gram_matrix = torch.from_numpy(
                    f['noise1234_gram_{}'.format(layer_name)]
                )

                assert torch.equal(actual_gram_matrix, expected_gram_matrix)

    def test_gram_matrices_source_image(self):
        important_layers = [
            'conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        model = Model(PYTORCH_MODEL_PATH)
        synthesis_model, _ = model.get_synthesis_model_and_losses()

        # register hooks to extract the important activations
        hook = GramHook()
        for name, layer in synthesis_model.named_children():
            if name in important_layers:
                layer.register_forward_hook(hook)

        # load an image and pass it through the synthesis model
        source_image = utilities.preprocess_image(
            utilities.load_image(SOURCE_IMG_PATH)
        )

        synthesis_model(source_image)

        # check if they are correct
        with h5py.File(REF_VALS_PATH, 'r') as f:
            for i, layer_name in enumerate(important_layers):
                actual_gram_matrix = hook.gram_matrices[i]
                expected_gram_matrix = torch.from_numpy(
                    f['source_img_gram_{}'.format(layer_name)]
                )

                assert torch.equal(actual_gram_matrix, expected_gram_matrix)


class TestLosses:
    def test_noise_image(self):
        noise_image = None
        with h5py.File(REF_VALS_PATH, 'r') as f:
            noise_image = torch.from_numpy(
                f['noise1234'][()]
            )

        model = Model(PYTORCH_MODEL_PATH)
        synthesis_model, losses = model.get_synthesis_model_and_losses()

        synthesis_model(noise_image)

        actual_loss_value = 0.0
        for loss in losses:
            actual_loss_value += loss.loss.item()

        with h5py.File(REF_VALS_PATH, 'r') as f:
            expected_loss_value = f['loss_value'][()]

            assert actual_loss_value == expected_loss_value

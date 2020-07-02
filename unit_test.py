import h5py
import torch
import torchvision

import utilities
from model import Model, ActivationsHook, GramHook

PYTORCH_MODEL_PATH = 'models/VGG19_normalized_avg_pool_pytorch'
REF_VALS_PATH = 'data/reference_values.hdf5'
SOURCE_IMG_PATH = 'img/pebbles.jpg'

# TODO: make the tests device-aware


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
        source_img = utilities.load_image(SOURCE_IMG_PATH)
        transform = torchvision.transforms.ToTensor()
        actual_source_img = transform(source_img).permute(1, 2, 0)

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
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        synthesis_model = utilities.load_model(PYTORCH_MODEL_PATH)

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
                if layer_name == 'relu1_1':
                    layer_name = 'conv1_1'

                actual_activations = hook.activations[i]
                expected_activations = torch.from_numpy(
                    f['noise1234_activations_{}'.format(layer_name)][()]
                )

                assert torch.allclose(
                    actual_activations, expected_activations, atol=1e-07
                )

    def test_activations_source_image(self):
        important_layers = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        synthesis_model = utilities.load_model(PYTORCH_MODEL_PATH)

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
                if layer_name == 'relu1_1':
                    layer_name = 'conv1_1'

                actual_activations = hook.activations[i]
                expected_activations = torch.from_numpy(
                    f['source_img_activations_{}'.format(layer_name)][()]
                )

                assert torch.allclose(
                    actual_activations, expected_activations, atol=1e-05
                )

    def test_gram_matrices_noise_image(self):
        important_layers = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        synthesis_model = utilities.load_model(PYTORCH_MODEL_PATH)

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
                if layer_name == 'relu1_1':
                    layer_name = 'conv1_1'

                gram_matrix = hook.gram_matrices[i]
                actual_gram_matrix = gram_matrix * gram_matrix.shape[0]

                expected_gram_matrix = torch.from_numpy(
                    f['noise1234_gram_{}'.format(layer_name)][()]
                ).to(torch.float32)

                assert torch.allclose(
                    actual_gram_matrix, expected_gram_matrix, atol=1e-07
                )

    def test_gram_matrices_source_image(self):
        important_layers = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        synthesis_model = utilities.load_model(PYTORCH_MODEL_PATH)

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
                if layer_name == 'relu1_1':
                    layer_name = 'conv1_1'

                gram_matrix = hook.gram_matrices[i]
                actual_gram_matrix = gram_matrix * gram_matrix.shape[0]

                expected_gram_matrix = torch.from_numpy(
                    f['source_img_gram_{}'.format(layer_name)][()]
                ).to(torch.float32)

                assert torch.allclose(
                    actual_gram_matrix, expected_gram_matrix, atol=1e-05
                )

    # TODO: test this on the GPU as well!
    def test_activations_two_consecutive_runs(self):
        important_layers = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ]

        # load an image
        source_image = utilities.preprocess_image(
            utilities.load_image(SOURCE_IMG_PATH)
        )

        activations = []
        for j in range(2):
            # load a model and pass the image through it
            synthesis_model = utilities.load_model(PYTORCH_MODEL_PATH)

            # register hooks to extract the important activations
            hook = ActivationsHook()
            for name, layer in synthesis_model.named_children():
                if name in important_layers:
                    layer.register_forward_hook(hook)

            synthesis_model(source_image)

            # save the activations
            activations.append(hook.activations[-1])

        assert torch.equal(activations[0], activations[1]), \
            'mean error: {}'.format(
                (activations[0] - activations[1]).abs().mean()
            )


class TestLosses:
    def test_noise_image(self):
        noise_image = None
        with h5py.File(REF_VALS_PATH, 'r') as f:
            noise_image = torch.from_numpy(
                f['noise1234'][()]
            )

        target_image = utilities.preprocess_image(
            utilities.load_image(SOURCE_IMG_PATH)
        )

        model = Model(PYTORCH_MODEL_PATH, torch.device('cpu'), target_image)

        model(noise_image)
        actual_loss_value = model.get_loss().detach().cpu()

        with h5py.File(REF_VALS_PATH, 'r') as f:
            expected_loss_value = torch.tensor(
                f['loss_value'][()], dtype=torch.float32
            )

            assert torch.allclose(actual_loss_value, expected_loss_value)

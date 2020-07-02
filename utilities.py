import pickle
import sys
import imp
import inspect
import importlib

import PIL
import torch
import numpy as np


def save_model(model, path):
    """
    Saves the model(s), including the definitions in its containing module.
    Restore the model(s) with load_model. References to other modules
    are not chased; they're assumed to be available when calling load_model.
    The state of any other object in the module is not stored.
    Written by Pauli Kemppinen.
    """
    model_pickle = pickle.dumps(model)

    # Handle dicts, lists and tuples of models.
    model = list(model.values()) if isinstance(model, dict) else model
    model = (
        (model,)
        if not (isinstance(model, list) or isinstance(model, tuple))
        else model
    )

    # Create a dict of modules that maps from name to source code.
    module_names = {m.__class__.__module__ for m in model}
    modules = {
        name:
            inspect.getsource(importlib.import_module(name))
            for name in module_names
    }

    pickle.dump((modules, model_pickle), open(path, 'wb'))


def load_model(path):
    """
    Loads the model(s) stored by save_model.
    Written by Pauli Kemppinen.
    """
    modules, model_pickle = pickle.load(open(path, 'rb'))

    # Temporarily add or replace available modules with stored ones.
    sys_modules = {}
    for name, source in modules.items():
        module = imp.new_module(name)
        exec(source, module.__dict__)
        if name in sys.modules:
            sys_modules[name] = sys.modules[name]
        sys.modules[name] = module

    # Map pytorch models to cpu if cuda is not available.
    if imp.find_module('torch'):
        import torch
        original_load = torch.load

        def map_location_cpu(*args, **kwargs):
            kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)
        torch.load = (
            original_load
            if torch.cuda.is_available()
            else map_location_cpu
        )

    model = pickle.loads(model_pickle)

    if imp.find_module('torch'):
        torch.load = original_load  # Revert monkey patch.

    # Revert sys.modules to original state.
    for name in modules.keys():
        if name in sys_modules:
            sys.modules[name] = sys_modules[name]
        else:
            # Just to make sure nobody else depends on these existing.
            sys.modules.pop(name)

    return model


def load_image(path):
    return PIL.Image.open(path)


def preprocess_image(
    image: PIL.Image.Image,
    new_size: int = 256,
    mean: np.ndarray = np.array([0.40760392,  0.45795686,  0.48501961])
) -> torch.Tensor:
    assert isinstance(image, PIL.Image.Image)

    # use PIL here because it resamples properly
    # (https://twitter.com/jaakkolehtinen/status/1258102168176951299)
    image = image.resize((new_size, new_size), resample=PIL.Image.LANCZOS)

    # RGB to BGR
    r, g, b = image.split()
    image_bgr = PIL.Image.merge('RGB', (b, g, r))

    # normalization
    image_numpy = np.array(image_bgr, dtype=np.float32) / 255.0
    image_numpy -= mean
    image_numpy *= 255.0

    # [H, W, C] -> [N, C, H, W]
    image_numpy = np.transpose(image_numpy, (2, 0, 1))[None, :, :, :]

    return torch.from_numpy(
        image_numpy
    ).to(torch.float32)


def gram_matrix(activations):
    b, n, x, y = activations.size()
    activation_matrix = activations.view(b * n, x * y)
    G = torch.mm(activation_matrix, activation_matrix.t())    # gram product
    return G.div(b * n * x * y)     # normalization

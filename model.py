import utilities


class Model:
    def __init__(self, path):
        self.net = utilities.load_model(path)

    def get_synthesis_model_and_losses(self):
        raise NotImplementedError()

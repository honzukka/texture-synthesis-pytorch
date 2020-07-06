import time

import torch
import torchvision

import utilities


class Optimizer:
    def __init__(self, model, n_iter=100, checkpoint_every=1):
        self.model = model
        self.n_iter = n_iter
        self.checkpoint_every = checkpoint_every

        self.opt_image = torch.randn_like(model.target_image)
        self.reparam_func = self.get_reparam_func(model.target_image)
        self.last_checkpoint_time = time.time()
        self.to_pil = torchvision.transforms.ToPILImage()

    def optimize(self):
        optimizer = torch.optim.LBFGS([self.opt_image.requires_grad_()])

        iteration = 0
        self.losses = []
        self.opt_images_pil = []
        while iteration < self.n_iter:
            def closure():
                optimizer.zero_grad()
                self.model(self.reparam_func(self.opt_image))
                loss = self.model.get_loss()
                loss.backward()

                return loss

            optimizer.step(closure)

            if iteration % self.checkpoint_every == self.checkpoint_every - 1:
                self.checkpoint(iteration + 1, self.model.get_loss().item())
            iteration += 1

        return self.reparam_func(self.opt_image).detach().cpu()

    def checkpoint(self, iteration, loss):
        time_delta = time.time() - self.last_checkpoint_time
        print('iteration: {}, loss: {} ({:.2f}s)'.format(
            iteration, loss, time_delta
        ))

        self.losses.append(loss)
        opt_image_normalized = utilities.normalize_image(
            self.reparam_func(self.opt_image.clone().detach().cpu())
        ).squeeze()
        self.opt_images_pil.append(
            self.to_pil(opt_image_normalized)
        )

        self.last_checkpoint_time = time.time()

    def get_reparam_func(self, target_image):
        minimum = target_image.min()
        maximum = target_image.max()
        value_range = maximum - minimum

        return lambda x: (utilities.sigmoid(x) * value_range) + minimum

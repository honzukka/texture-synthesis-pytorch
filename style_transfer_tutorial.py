# from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

import copy

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128

loader = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(imsize), torchvision.transforms.ToTensor()]
)
unloader = torchvision.transforms.ToPILImage()


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float32)


style_img = image_loader('./img/picasso.jpg')
content_img = image_loader('./img/dancing.jpg')

assert style_img.size() == content_img.size()


def imshow(tensor, title=None):
    plt.figure()
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


# NOTE: not a true PyTorch loss function - missing backward()!
class ContentLoss(torch.nn.Module):
    def __init__(self, target_feature_maps):
        super(ContentLoss, self).__init__()
        self.target_feature_maps = target_feature_maps.detach()

    def forward(self, opt_feature_maps):
        self.loss = F.mse_loss(opt_feature_maps, self.target_feature_maps)
        return opt_feature_maps


def gram_matrix(feature_maps):
    b, n, x, y = feature_maps.size()
    feature_matrix = feature_maps.view(b * n, x * y)
    G = torch.mm(feature_matrix, feature_matrix.t())    # gram product
    return G.div(b * n * x * y)     # normalization


# NOTE: not a true PyTorch loss function - missing backward()!
class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature_maps):
        super(StyleLoss, self).__init__()
        self.target_gram_matrix = gram_matrix(target_feature_maps).detach()

    def forward(self, opt_feature_maps):
        opt_gram_matrix = gram_matrix(opt_feature_maps)
        self.loss = F.mse_loss(opt_gram_matrix, self.target_gram_matrix)
        return opt_feature_maps


net = torchvision.models.vgg19(pretrained=True).features.to(device).eval()

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(
    net, normalization_mean, normalization_std, style_img, content_img,
    content_layers=content_layers_default, style_layers=style_layers_default
):
    net_copy = copy.deepcopy(net)
    normalization_module = Normalization(
        normalization_mean, normalization_std
    ).to(device)

    content_losses = []
    style_losses = []

    model = torch.nn.Sequential(normalization_module)

    conv_counter = 0
    for layer in net_copy.children():
        if isinstance(layer, torch.nn.Conv2d):
            conv_counter += 1
            name = 'conv_{}'.format(conv_counter)
        elif isinstance(layer, torch.nn.ReLU):
            name = 'relu_{}'.format(conv_counter)
            layer = torch.nn.ReLU(inplace=False)    # replace in-place
        elif isinstance(layer, torch.nn.MaxPool2d):
            # TODO: replace by average pooling?
            name = 'pool_{}'.format(conv_counter)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = 'bn_{}'.format(conv_counter)
        else:
            raise RuntimeError(
                'Unrecognized layer: {}'.format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        # insert losses
        if name in content_layers:
            target_feature_maps = model(content_img).detach()
            content_loss = ContentLoss(target_feature_maps)
            model.add_module(
                'content_loss_{}'.format(conv_counter), content_loss
            )
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature_maps = model(style_img).detach()
            style_loss = StyleLoss(target_feature_maps)
            model.add_module(
                'style_loss_{}'.format(conv_counter), style_loss
            )
            style_losses.append(style_loss)

    # get rid of layers after the last loss
    for i in range(len(model) - 1, -1, -1):
        if (
            isinstance(model[i], ContentLoss) or
            isinstance(model[i], StyleLoss)
        ):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses


opt_img = content_img.clone()
# opt_img = torch.randn(content_img.data.size(), device=device)


def get_optimizer(opt_img):
    return torch.optim.LBFGS([opt_img.requires_grad_()])


def run_style_transfer(
    net, normalization_mean, normalization_std,
    content_img, style_img, opt_img,
    n_steps=20, style_weight=1*1000*1000, content_weight=1
):
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(
        net, normalization_mean, normalization_std, style_img, content_img
    )
    optimizer = get_optimizer(opt_img)

    print('Optimizing...')
    run = [0]
    while run[0] <= n_steps:
        def closure():
            opt_img.data.clamp_(0, 1)    # TODO: sigmoid better?

            optimizer.zero_grad()
            model(opt_img)
            style_score = 0
            content_score = 0

            for style_loss in style_losses:
                style_score += style_loss.loss
            for content_loss in content_losses:
                content_score += content_loss.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 10 == 0:
                print('run {}:'.format(run))
                print('Style Loss: {:.4f}, Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()
                ))

            return style_score + content_score

        optimizer.step(closure)

    opt_img.data.clamp_(0, 1)
    return opt_img


output = run_style_transfer(
    net, imagenet_mean, imagenet_std, content_img, style_img, opt_img
)

imshow(style_img, title='Style Image')
imshow(content_img, title='Content Image')
imshow(output, title='Output Image')

unloader(output[0]).save('output.png')

import argparse
from collections import OrderedDict
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torchvision          # type: ignore

import utilities
import caffemodel2pytorch   # type: ignore


# the original Caffe model can be found at
# http://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel
def main(args: Optional[argparse.Namespace] = None):
    return_value = True
    if args is None:
        args = parse_arguments()
        return_value = False

    caffe_model_path = os.path.join(args.model_dir, args.caffe_model)
    pytorch_model_out_path = os.path.join(args.model_dir, args.out_name)
    tmp_path = os.path.join(args.model_dir, 'temp.pt')

    print('Converting normalized Caffe weights to a PyTorch state_dict...')
    caffe_args = SimpleNamespace(
        model_caffemodel=caffe_model_path,
        output_path=tmp_path,
        caffe_proto=''
    )
    caffemodel2pytorch.main(caffe_args)
    state_dict = torch.load(tmp_path)

    # reshape caffe bias to match the PyTorch one
    for learnables_key in state_dict:
        if 'bias' in learnables_key:
            state_dict[learnables_key] = state_dict[learnables_key].squeeze()

    print('Loading VGG19 PyTorch model...')

    # only features are needed
    net = torchvision.models.vgg19(pretrained=True).features

    # rename VGG19's feature layers, so that we can refer to them easily later
    new_layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
        'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
        'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'conv5_4', 'relu5_4', 'pool5'
    ]
    net = torch.nn.Sequential(
        OrderedDict(zip(new_layer_names, net))
    )

    # replace max pooling by average pooling
    for i in range(len(net)):
        if isinstance(net[i], torch.nn.MaxPool2d):
            net[i] = torch.nn.AvgPool2d(2, stride=2)

    print('Loading normalized weights into the PyTorch model...')
    net.load_state_dict(state_dict)

    # remove intermediate files
    os.remove(tmp_path)

    if return_value:
        return net
    else:
        print('Saving the converted model..')
        utilities.save_model(net, pytorch_model_out_path)
        return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model_dir',
        default='./models',
        help='Directory containing models.'
    )

    parser.add_argument(
        '--caffe_model',
        default='vgg_normalised.caffemodel',
        help='Caffe model weights file name.'
    )

    parser.add_argument(
        '--out_name',
        default='VGG19_normalized_avg_pool_pytorch',
        help='PyTorch model weights output file name.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()

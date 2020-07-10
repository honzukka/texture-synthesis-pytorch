from typing import Optional
import argparse
import os

import torch
import matplotlib.pyplot as plt     # type: ignore

import utilities
import model
import optimize


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model & data
    target_image = utilities.preprocess_image(
        utilities.load_image(args.img_path)
    )
    net = model.Model(args.model_path, device, target_image)

    # synthesize
    optimizer = optimize.Optimizer(net, args)
    result = optimizer.optimize()

    # save result
    final_image = utilities.postprocess_image(
        result, utilities.load_image(args.img_path)
    )
    final_image.save(os.path.join(args.out_dir, 'output.png'))

    # plot loss
    x = list(range(
        args.checkpoint_every - 1,
        len(optimizer.losses) * args.checkpoint_every,
        args.checkpoint_every
    ))
    plt.plot(x, optimizer.losses)
    plt.savefig(os.path.join(args.out_dir, 'loss_plot.png'))
    plt.close()

    # save intermediate images
    for i, image in enumerate(optimizer.opt_images):
        image.save(
            os.path.join(args.out_dir, 'intermediate_image_{}.png'.format(
                i * args.checkpoint_every
            ))
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-o',
        dest='out_dir',
        default='output',
        help=(
            'Output directory. All output will be save into a subdirectory '
            'within this directory which will be called after the input image '
            'and will be created if it does not exist yet.'
        )
    )

    parser.add_argument(
        '-i',
        dest='img_path',
        default='img/pebbles.jpg',
        help='Path to the target texture image.'
    )

    parser.add_argument(
        '-m',
        dest='model_path',
        default='models/VGG19_normalized_avg_pool_pytorch',
        help='Path to the model file.'
    )

    parser.add_argument(
        '--check',
        dest='checkpoint_every',
        type=int,
        default=1,
        help=(
            'The number of iterations between storing progress information '
            'and intermediate values.'
        )
    )

    parser.add_argument(
        '-n',
        dest='n_steps',
        type=int,
        default=100,
        help='The maximum number of optimizer steps to be performed.'
    )

    parser.add_argument(
        '--iter',
        dest='max_iter',
        type=int,
        default=20,
        help='The maximum number of iterations within one optimization step.'
    )

    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=1.0,
        help='Optimizer learning rate.'
    )

    args = parser.parse_args()

    # create output subfolder
    input_name = os.path.splitext(os.path.basename(args.img_path))[0]
    args.out_dir = os.path.join(args.out_dir, input_name)
    os.makedirs(args.out_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    main()

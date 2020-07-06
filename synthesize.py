import argparse
import os

import torch
import matplotlib.pyplot as plt
import PIL
import numpy as np

import utilities
import model
import optimize


def main(args=None):
    if args is None:
        args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model & data
    target_image = utilities.preprocess_image(
        utilities.load_image(args.img_path)
    )
    net = model.Model(args.model_path, device, target_image)

    # synthesize
    optimizer = optimize.Optimizer(
        net,
        n_iter=args.iter_limit,
        checkpoint_every=args.checkpoint_every
    )
    result = optimizer.optimize()

    # save result
    final_image = utilities.postprocess_image(result).numpy()
    final_image_pil = PIL.Image.fromarray((final_image * 255).astype(np.uint8))
    final_image_pil.save(os.path.join(args.out_dir, 'output.png'))

    # plot loss
    plt.plot(optimizer.losses)
    plt.savefig(os.path.join(args.out_dir, 'loss_plot.png'))
    plt.close()

    # save intermediate images
    for i, image in enumerate(optimizer.opt_images_pil):
        image.save(
            os.path.join(args.out_dir, 'intermediate_image_{}.png'.format(
                i * args.checkpoint_every
            ))
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-o',
        dest='out_dir',
        default='output',
        help='Output directory.'
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
        default=2,
        help=(
            'The number of iterations between storing progress information '
            'and intermediate values.'
        )
    )

    parser.add_argument(
        '--lim',
        dest='iter_limit',
        type=int,
        default=10,
        help=(
            'The maximum number of iterations to be performed '
            'even if loss keeps falling. -1 means no maximum.'
        )
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()

import sys

from sklearn.utils import shuffle
import torch

from main import get_model, str2bool
import argparse
import pathlib
from torch import load as torch_load
import numpy as np

from generate_adversarial_images import get_adversarial_patch_pictures_by_art, \
    get_adversarial_patch_pictures_by_custom, \
    get_adversarial_patch_pictures_by_custom_with_batch
from transformations.experiments import custom_seq_transformation
from constant import MODEL_MAPPING, DATA_MAPPING, ADVERSARIAL_IMAGES_MAPPING, \
    OUTPUT_PATH, OUTPUT_PATH_TRANSFORMATION_IMAGE, PREFIX_NAME, RANDOM_STATE


def parse_arguments():
    parser = argparse.ArgumentParser()

    # EXPS
    parser.add_argument(
        '--patch-scale',
        type=int,
        default=100,
        help='scale of patch for attack',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generation', 'transformation', 'transfer'],
        default='generation',
        help='experiment type: generation adversarial images for model,'
             'transformation adversarial images or transfer attack',
    )
    parser.add_argument(
        '--type-attack',
        type=str,
        choices=['custom', 'art', 'batch_custom'],
        default='art',
        help='experiment type of attack',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=MODEL_MAPPING['resnet'],
        help='path to model',
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['cbn', 'resnet', 'bagnet', 'resnet_patch', 'cbn_patch', 'adv_cbn'],
        default='resnet',
        help='type of model',
    )
    # DATA
    parser.add_argument(
        '--images-arr-path',
        type=pathlib.Path,
        default=(
            DATA_MAPPING['test_images']
        ),
        help='path to the images array (*.npy)',
    )
    parser.add_argument(
        '--labels-arr-path',
        type=pathlib.Path,
        help='path to the labels array (*.npy)',
        default=(
            DATA_MAPPING['test_labels']
        ),
    )
    parser.add_argument(
        '--adversarial-images-arr-path',
        type=pathlib.Path,
        default=(
            ADVERSARIAL_IMAGES_MAPPING[100]['resnet']
        ),
        help='path to the adversarial images array (*.npy)',
    )
    parser.add_argument(
        '--im-shape',
        type=int,
        default=224,
        help='size of image',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=10,
        help='number of classes in data',
    )
    parser.add_argument(
        '--renumber',
        type=str2bool,
        default=False,
        help='if number of classes != 1000, then need renumber labels',
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='test',
        help='path with filename for saving output files',
    )
    parser.add_argument(
        '--pre-trained',
        type=str2bool,
        default=False,
        help='load pretrained model(true) or load clean model',
    )
    # OUTPUT
    parser.add_argument(
        '--output-path',
        type=pathlib.Path,
        default=OUTPUT_PATH,
        help='path with output results',
    )
    parser.add_argument(
        '--output-trans-image',
        type=pathlib.Path,
        default=OUTPUT_PATH_TRANSFORMATION_IMAGE,
        help='path with output results for transformation images',
    )
    parser.add_argument(
        '--prefix-name',
        type=str,
        default=PREFIX_NAME,
        help='prefix for name for transformation images',
    )
    parser.add_argument(
        '--load-from-constant',
        type=str2bool,
        default=False,
        help='if True, then loading model from path in constant.py, else need enter path with key --model-path',
    )
    parser.add_argument(
        '--count-images-from-first',
        type=int,
        default=4000,
        help='count of images for generation and transformation from first image to *this value*',
    )
    parser.add_argument(
        '--random-shuffle',
        type=str2bool,
        default=False,
        help='if True, then images will be shuffle',
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=40,
        help='maximum iterations for attack',
    )
    parser.add_argument(
        '--patch-type',
        type=str,
        default='square',
        help='',
    )
    parser.add_argument(
        '--target',
        type=str2bool,
        default=False,
        help='',
    )
    parser.add_argument(
        '--poln',
        type=str2bool,
        default=False,
        help='',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='',
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')
    print('Preprocessing data')

    X_test = np.load(args.images_arr_path)
    y_test = np.load(args.labels_arr_path)

    if args.random_shuffle:
        print('With random shuffle!')
        ind, X_test, y_test = shuffle(np.arange(y_test.shape[0]), X_test, y_test, random_state=RANDOM_STATE)
        file_path_arr = args.filename.split('.')
        np.save(f'{file_path_arr[0]}_shuffle_index', ind[:args.count_images_from_first])

    print('Loading model')
    if (args.mode == 'generation') or (args.mode == 'transformation'):
        if (args.model == 'cbn') or (args.model == 'cbn_patch'):
            aggregation = 'cbn'
        elif (args.model == 'resnet_patch') or (args.model == 'bagnet'):
            aggregation = 'mean'
        else:
            aggregation = None
        print(args.model, args.num_classes, aggregation)
        model = get_model(model_key=args.model,
                          num_classes=args.num_classes,
                          pretrained=args.pre_trained,
                          aggregation=aggregation)
        if not args.pre_trained:
            if args.load_from_constant:
                print(MODEL_MAPPING[args.model])
                weights = torch_load(MODEL_MAPPING[args.model])
            else:
                weights = torch_load(args.model_path)
            model.load_state_dict(weights)
    model.eval().to(device)

    if args.mode == 'generation':
        print('Generation adversarial images')
        if args.type_attack == 'art':
            if args.max_iters != 500:
                print(f'{args.max_iters} ITERATIONS!')
            get_adversarial_patch_pictures_by_art(model=model,
                                                  x=X_test[:args.count_images_from_first],
                                                  y=y_test[:args.count_images_from_first],
                                                  filename=args.filename,
                                                  patch_scale=args.patch_scale,
                                                  nb_classes=args.num_classes,
                                                  batch_size=1, max_iter=args.max_iters,
                                                  patch_type=args.patch_type,
                                                  targeted=args.target, poln=args.poln)

        if args.type_attack == 'custom':
            if args.max_iters != 40:
                print(f'{args.max_iters} ITERATIONS!')
            get_adversarial_patch_pictures_by_custom(model=model,
                                                     images=X_test[:args.count_images_from_first],
                                                     labels=y_test[:args.count_images_from_first],
                                                     filename=args.filename,
                                                     sticker_size=args.patch_scale,
                                                     device='cuda', max_iters=args.max_iters)
        if args.type_attack == 'batch_custom':
            print('Generation adversarial pictures with batch!')
            get_adversarial_patch_pictures_by_custom_with_batch(model=model,
                                                                images=X_test[:args.count_images_from_first],
                                                                labels=y_test[:args.count_images_from_first],
                                                                filename=args.output_path,
                                                                sticker_size=args.patch_scale,
                                                                device='cuda',
                                                                batch_size=args.batch_size,
                                                                max_iters=args.max_iters)


    if args.mode == 'transformation':
        print('Transformation images')
        adversarial_images = np.load(args.adversarial_images_arr_path)
        custom_seq_transformation(model=model,
                                  images=adversarial_images[:args.count_images_from_first],
                                  vanilla_images=X_test[:args.count_images_from_first],
                                  labels=y_test[:args.count_images_from_first],
                                  output=args.output_trans_image,
                                  prefix_name=args.prefix_name)


if __name__ == '__main__':
    main()

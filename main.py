import os
import argparse
import pathlib
import time
import torchvision
from torch import Tensor
import torch
from torch import load as torch_load
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from bagnet import ClippedBagNet
from attacks import SpsaSticker, PgdSticker, get_adv_images
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from preprocess import (
    preprocess_data_short_cbn,
    get_dataloader,
    get_target_label,
)

from add_lib.PatchGuard.nets.resnet import resnet18 as resnet18_patch
from add_lib.PatchGuard.nets.bagnet import bagnet33
from constant import MODEL_MAPPING, DATA_MAPPING, \
    OUTPUT_PATH, OUTPUT_MODEL_NAME


DEFAULT_TRAIN_LR = 0.01

def parse_arguments():
    parser = argparse.ArgumentParser()

    # MODEL
    parser.add_argument(
        '--model',
        type=str,
        choices=['cbn', 'bagnet', 'resnet'],
        default='cbn',
        help='model type',
    )
    parser.add_argument(
        '--model-path',
        type=pathlib.Path,
        default=MODEL_MAPPING['cbn'],
        help='path to the model if train is False',
    )
    parser.add_argument(
        '--train-lr',
        type=float,
        default=None,
        help='lr for training',
    )
    parser.add_argument(
        '--train',
        type=str2bool,
        default=False,
        help='train model(true) before attack or use pretrained model',
    )
    parser.add_argument(
        '--pre-trained',
        type=str2bool,
        default=False,
        help='load pretrained model(true) or load clean model',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=40,
        help='number of epochs for training',
    )

    # DATA
    parser.add_argument(
        '--images-arr-path',
        type=pathlib.Path,
        default=(
            DATA_MAPPING['images']
        ),
        help='path to the images array (*.npy)',
    )
    parser.add_argument(
        '--labels-arr-path',
        type=pathlib.Path,
        help='path to the labels array (*.npy)',
        default=(
            DATA_MAPPING['labels']
        ),
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
        default=False, #True, сейчас же данные уже готовенькие
        help='if number of classes != 1000, then need renumber labels',
    )

    # ATTACK
    parser.add_argument(
        '--sticker-size',
        type=int,
        default=50,
        help='size of attack sticker for image',
    )
    parser.add_argument(
        '--targeted',
        type=str2bool,
        default=False,
        help='type of attack: target(true) or false',
    )
    parser.add_argument(
        '--attack',
        type=str,
        choices=['pgd', 'spsa'],
        default='pgd',
        help='type of attack',
    )
    parser.add_argument(
        '--attack-iters',
        type=int,
        default=40,
        help='number of iters for attack',
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1.0,
        help='eps for sticker attack',
    )
    parser.add_argument(
        '--clip-min',
        type=float,
        default=0.0,
        help='clip-min is min value from array of image',
    )
    parser.add_argument(
        '--clip-max',
        type=float,
        default=1.0,
        help='clip-max is max value from array of image',
    )

    # PGD
    parser.add_argument(
        '--pgd-alpha',
        type=float,
        default=0.8, #2 / 255,
        help='alpha for pgd attack',
    )

    # SPSA
    parser.add_argument(
        '--attack-lr',
        type=float,
        default=0.01,
        help='lr for attack',
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=0.1,
        help='delta for spsa attack',
    )
    parser.add_argument(
        '--classic',
        type=str2bool,
        default=False,
        help='type of spsa attack',
    )
    parser.add_argument(
        '--spsa-alpha',
        type=float,
        default=0.5,
        help='alpha for default spsa attack',
    )

    # OUTPUT
    parser.add_argument(
        '--output-path',
        type=pathlib.Path,
        default=OUTPUT_PATH,
        help='path with output results',
    )
    parser.add_argument(
        '--output-model-name',
        type=str,
        default=OUTPUT_MODEL_NAME,
        help='model name for output',
    )

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_model(model_key, num_classes, aggregation='cbn', pretrained=False):
    if model_key == 'cbn':
        model = ClippedBagNet(num_classes=num_classes,
                              aggregation=aggregation)
    elif model_key == 'bagnet':
        model = ClippedBagNet(num_classes=num_classes,
                              aggregation=aggregation)
    elif model_key == 'resnet':
        model = torchvision.models.resnet18(pretrained=pretrained,
                                            num_classes=num_classes)
    elif model_key == 'resnet_patch':
        model = resnet18_patch(aggregation=aggregation,
                               num_classes=num_classes)
    elif model_key == 'cbn_patch':
        model = bagnet33(pretrained=pretrained,
                         aggregation=aggregation,
                         num_classes=num_classes)
    elif model_key == 'adv_cbn':
        model = bagnet33(pretrained=pretrained,
                         aggregation='adv',
                         num_classes=num_classes)
    else:
        raise ValueError('Wrong model type!')
    return model


def remetric_accuracy(cou, true_labels, pred_labels):
    k = 0
    for i in range(0, len(true_labels), cou):
        if true_labels[i : i + cou] != pred_labels[i : i + cou]:
            k += 1

    return 1 - k / (len(true_labels) // cou)


def attack_step(
    model,
    attack,
    dataloader,
    acceptable_labels,
    sticker_size,
    im_shape,
    targeted,
    device,
    output_path,
    writer: SummaryWriter,
    prefix: ""
):
    pred_labels = []
    adv_count = 0
    target_labels = []
    adv_images = []

    for image_i, (x, y) in enumerate(dataloader):
        if targeted:
            y = get_target_label(model, x, acceptable_labels)
            target_labels.append(y.cpu().detach().numpy()[0])
            y = y.to(device)
        result = get_adv_images(
            images=x,
            labels=y,
            sticker_size=sticker_size,
            im_shape=(im_shape, im_shape),
            attack=attack,
        )
        success = result['success']
        pred = result['pred_top']
        n = 10
        if image_i < n:
            writer.add_image(f"test_image/{prefix}/im={image_i}", result['adv_images'][0])
        adv_images.append(result['adv_images'].cpu().detach().numpy()[0])# for 4-dimensional weight [64, 3, 1, 1], but got 5-dimensional input of size [1, 1, 3, 224, 224] instead

        if success:
            print(f'Image №{image_i + 1} successfully attacked')
            adv_count += 1
        else:
            print(f'Image №{image_i + 1} UNsuccessfully attacked')

        pred_labels.append(pred)

    try:
        np.save(output_path, adv_images)
    except:
        output_path = f'exception_saving_{time.time()}.npy'
        print(f"Exception saving images in {output_path}")
        np.save(output_path, adv_images)
    print(f'====>: {adv_count} of {image_i} pictures changed label in attack')
    return pred_labels, target_labels, adv_images


def main():
    args = parse_arguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Preprocessing data')
    images, labels = preprocess_data_short_cbn(
        image_arr_path=args.images_arr_path,
        lab_arr_path=args.labels_arr_path,
        renumber=args.renumber,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )
    writer = SummaryWriter(log_dir=args.output_path)

    if args.train:
        if args.train_lr is None:
            if args.model == 'cbn':
                args.train_lr = 0.001
            else:
                args.train_lr = DEFAULT_TRAIN_LR
        os.makedirs(args.output_path, exist_ok=True)
        print('Training model')
        model = get_model(model_key=args.model,
                          num_classes=args.num_classes,
                          pretrained=args.pre_trained)
        model.to(device)
        trainer = Trainer(
            model=model,
            writer=writer,
            epochs=args.epochs,
            lr=args.train_lr,
            output_path=args.output_path,
        )
        dataloader = get_dataloader(X_train, y_train, device=device)
        test_dataloader = get_dataloader(X_test, y_test, device=device)
        trainer.train(
            dataloader=dataloader, validloader=test_dataloader, device=device
        )

        print('Testing model')
        acc = trainer.test(dataloader=test_dataloader, device=device)
        print(f'Test finished with accuracy={acc}')
        writer.add_scalar(tag="accuracy/test", scalar_value=acc)

        model_path = os.path.join(args.output_path, args.output_model_name)
        torch.save(trainer.model, model_path)
        model = trainer.model

    else:
        model = get_model(model_key=args.model,
                          num_classes=args.num_classes,
                          pretrained=args.pre_trained)
        if not args.pre_trained:
            weights = torch_load(args.model_path)
            model.load_state_dict(weights)
        model.to(device)

    dataloader = get_dataloader(
        X_train, y_train, device=device, batch_size=1, shuffle=False
    )
    test_dataloader = get_dataloader(
        X_test, y_test, device=device, batch_size=1, shuffle=False
    )

    print('Attack preprocess')
    if args.attack == 'spsa':
        eps = Tensor([args.eps])
        attack = SpsaSticker(
            predict=model,
            eps=eps,
            targeted=args.targeted,
            nb_iter=args.attack_iters,
            classic=args.classic,
            lr=args.attack_lr,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            delta=args.delta,
            alpha=args.spsa_alpha,
        )

    elif args.attack == 'pgd':
        attack = PgdSticker(
            model=model,
            eps=args.eps,
            alpha=args.pgd_alpha,
            iters=args.attack_iters,
            target=args.targeted,
        )

    if args.targeted:
        acceptable_labels = np.unique(y_train)
    else:
        acceptable_labels = None

    output_path = f'train_{args.model}_{args.output_path}_{args.attack}_{args.sticker_size}.npy'
    print('Attack loop for train-part of dataset')
    pred_labels, target_labels, adv_images = attack_step(
        model=model,
        attack=attack,
        dataloader=dataloader,
        acceptable_labels=acceptable_labels,
        sticker_size=args.sticker_size,
        im_shape=args.im_shape,
        targeted=args.targeted,
        device=device,
        output_path=output_path,
        writer=writer,
        prefix="train"
    )

    acc = metrics.accuracy_score(y_train, np.array(pred_labels))
    print(f'Accuracy for train part = {acc}')
    writer.add_scalar(tag="attack_accuracy/train", scalar_value=acc)
    if args.targeted:
        target_acc = metrics.accuracy_score(np.array(target_labels),
                                            np.array(pred_labels))
        print(f'Target accuracy for train part = {target_acc}')
        writer.add_scalar(tag="attack_target_accuracy/train", scalar_value=target_acc)

    output_path = f'test_{args.model}_{args.output_path}_{args.attack}_{args.sticker_size}.npy'
    print('Attack loop for test-part of dataset')
    pred_labels, target_labels, adv_images = attack_step(
        model=model,
        attack=attack,
        dataloader=test_dataloader,
        acceptable_labels=acceptable_labels,
        sticker_size=args.sticker_size,
        im_shape=args.im_shape,
        targeted=args.targeted,
        device=device,
        output_path=output_path,
        writer=writer,
        prefix="test"
    )

    acc = metrics.accuracy_score(y_test, np.array(pred_labels))
    print(f'Accuracy for test part = {acc}')
    writer.add_scalar(tag="attack_accuracy/test", scalar_value=acc)
    if args.targeted:
        target_acc = metrics.accuracy_score(np.array(target_labels),
                                            np.array(pred_labels))
        print(f'Target accuracy for test part = {target_acc}')
        writer.add_scalar(tag="attack_target_accuracy/test", scalar_value=target_acc)


if __name__ == '__main__':
    main()

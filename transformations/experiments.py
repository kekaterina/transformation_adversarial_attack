import os

from bagnet import *
import torch
import numpy as np
from progress.bar import IncrementalBar

import cv2
import matplotlib.pyplot as plt
from torch import Tensor

from .transformations import get_transformation_choise, reshape_image, \
    rotate_image, dark_image, gauss_transformer

SHORT_MAPPING_DICT = {5: 'loaf',
                      4: 'wallet',
                      9: 'finch',
                      6: 'GrannySmith',
                      1: 'bassoon',
                      2: 'dog/rig',
                      3: 'lizard',
                      7: 'SaintBernard,',
                      0: 'boa',
                      8: 'space/rockets'}


def revert_and_get_predict(model, new_im):
    tmp = np.moveaxis(new_im, 2, 0)
    pred = np.argmax(model(torch.Tensor([tmp]).cuda()).cpu().detach().numpy(), axis=1)[0]
    return pred, tmp


def get_transform_predict(model, img, dark_one, rotation_one, scale_one):
    reshape_im = reshape_image(img)
    new_im = rotate_image(reshape_im, angle=float(rotation_one), scale=float(scale_one), center=None)
    pred_rotate, rotate_new_im = revert_and_get_predict(model, new_im)

    new_im = dark_image(reshape_im, delta=dark_one)
    pred_dark, dark_new_im = revert_and_get_predict(model, new_im)

    new_im = gauss_transformer(reshape_im)
    pred_gauss, gauss_new_im = revert_and_get_predict(model, new_im)

    new_im = rotate_image(reshape_im, angle=float(rotation_one), scale=float(scale_one), center=None)
    new_im = dark_image(new_im, delta=dark_one)
    new_im = gauss_transformer(new_im)
    pred_combo, combo_new_im = revert_and_get_predict(model, new_im)


    return {'pred_rotate': pred_rotate,
            'pred_dark': pred_dark,
            'pred_gauss': pred_gauss,
            'pred_combo': pred_combo,
            'rotate_image': rotate_new_im,
            'dark_image': dark_new_im,
            'gauss_image': gauss_new_im,
            'adversarial_transform_img': combo_new_im}


def printing_success_rate(labels, pred_list, name):
    m = labels[pred_list != labels].shape[0]
    n = labels.shape[0]
    cou = m / n * 100
    print(f'{name}: succsess rate {m} of {n} ({cou}%)')
    return cou


def print_with_sticker_vs_without_sticker_success_rate(labels, transform_predict, vanilla_transform_predict):
    print('WITH STICKER')
    cou = printing_success_rate(labels, transform_predict['pred_list'], 'pred_list')
    cou_rotate = printing_success_rate(labels, transform_predict['pred_list_rotate'], 'pred_list_rotate')
    cou_dark = printing_success_rate(labels, transform_predict['pred_list_dark'], 'pred_list_dark')
    cou_gauss = printing_success_rate(labels, transform_predict['pred_list_gauss'], 'pred_list_gauss')
    cou_combo = printing_success_rate(labels, transform_predict['pred_list_combo'], 'pred_list_combo')

    print('WITHOUT STICKER')
    cou_rotate = printing_success_rate(labels, vanilla_transform_predict['pred_list_rotate'], 'pred_list_rotate')
    cou_dark = printing_success_rate(labels, vanilla_transform_predict['pred_list_dark'], 'pred_list_dark')
    cou_gauss = printing_success_rate(labels, vanilla_transform_predict['pred_list_gauss'], 'pred_list_gauss')
    cou_combo = printing_success_rate(labels, vanilla_transform_predict['pred_list_combo'], 'pred_list_combo')

    return {'cou_rotate': cou_rotate,
            'cou_dark': cou_dark,
            'cou_gauss': cou_gauss,
            'cou_combo': cou_combo}


def custom_seq_transformation(model, images, vanilla_images, labels, output='test/', prefix_name='test'):
    if not os.path.exists(output):
        os.makedirs(output)
    transformations = get_transformation_choise(labels.shape[0])

    scale_one = np.random.choice(transformations['scale_trans'], size=labels.shape[0], replace=True)
    rotation_one = np.random.choice(transformations['rotation_trans'], size=labels.shape[0], replace=True)
    dark_one = np.random.choice(transformations['dark_trans'], size=labels.shape[0], replace=True)

    print('Transformation adversarial images')
    transform_predict = transformation_predict_loop(model, images, dark_one, rotation_one, scale_one)
    print('Transformation clean images')
    vanilla_transform_predict = transformation_predict_loop(model, vanilla_images, dark_one, rotation_one, scale_one)

    mapping_list_name = {'rotate_image_list': 'rotate',
                         'dark_image_list': 'dark',
                         'gauss_image_list': 'gauss',
                         'adversarial_transformation_img': 'combo'}

    print('Saving transformation images')
    for key in ['rotate_image_list', 'dark_image_list',
                'gauss_image_list', 'adversarial_transformation_img']:

        np.save(f'{str(output)}/{prefix_name}_adv_{mapping_list_name[key]}_images.npy',
                np.array(transform_predict[key]))
        np.save(f'{str(output)}/{prefix_name}_{mapping_list_name[key]}_images.npy',
                np.array(vanilla_transform_predict[key]))

    success_rate = print_with_sticker_vs_without_sticker_success_rate(labels,
                                                                      transform_predict,
                                                                      vanilla_transform_predict)


    return {'pred_list': transform_predict['pred_list'],
            'pred_list_gauss': transform_predict['pred_list_gauss'],
            'pred_list_dark': transform_predict['pred_list_dark'],
            'pred_list_rotate': transform_predict['pred_list_rotate'],
            'pred_list_combo': transform_predict['pred_list_combo'],
            'rate_r_d_g_c': (success_rate['cou_rotate'],
                             success_rate['cou_dark'],
                             success_rate['cou_gauss'],
                             success_rate['cou_combo']),
            'adversarial_transformation_img': transform_predict['adversarial_transformation_img'], }


def transformation_predict_loop(model, images, dark_one, rotation_one, scale_one):
    bar = IncrementalBar('Count of transform images', max=images.shape[0])

    pred_list_rotate = []
    pred_list_dark = []
    pred_list_gauss = []
    pred_list_combo = []
    adversarial_transformation_img = []
    pred_list = []

    rotate_image_list = []
    dark_image_list = []
    gauss_image_list = []

    for num, img in enumerate(images):
        pred = np.argmax(model(torch.Tensor(np.array([img])).cuda()).cpu().detach().numpy(), axis=1)
        pred_list.append(pred[0])

        adv_predictions = get_transform_predict(model, img, dark_one[num], rotation_one[num], scale_one[num])
        pred_list_rotate.append(adv_predictions['pred_rotate'])
        pred_list_dark.append(adv_predictions['pred_dark'])
        pred_list_gauss.append(adv_predictions['pred_gauss'])
        pred_list_combo.append(adv_predictions['pred_combo'])
        adversarial_transformation_img.append(adv_predictions['adversarial_transform_img'])

        rotate_image_list.append(adv_predictions['rotate_image'])
        dark_image_list.append(adv_predictions['dark_image'])
        gauss_image_list.append(adv_predictions['gauss_image'])
        bar.next()
    bar.finish()

    return {'pred_list': pred_list,
            'pred_list_rotate': pred_list_rotate,
            'pred_list_dark': pred_list_dark,
            'pred_list_gauss': pred_list_gauss,
            'pred_list_combo': pred_list_combo,
            'adversarial_transformation_img': adversarial_transformation_img,
            'rotate_image_list': rotate_image_list,
            'dark_image_list': dark_image_list,
            'gauss_image_list': gauss_image_list}


# тут отрисовка состязательных картинок
def show_images(images, patch_images, true_lab, pred_lab,
                pred_lab_trans, col, name, short_mapping_dict=SHORT_MAPPING_DICT) -> None:
    n = len(images)
    f = plt.figure(figsize=(20, 20))

    for i in range(0, 2 * n, 2):
        if i+2 > (n // col)*(2*col):
            break
        f.add_subplot(n // col, 2 * col, i + 1)
        plt.imshow(images[i // 2])
        plt.title(f'True label: {short_mapping_dict[true_lab[i // 2]]},\n'
                  f'predict label: {short_mapping_dict[pred_lab[i // 2]]}',
                  fontdict={'fontsize': 8, 'fontweight': 'medium'})
        plt.axis('off')
        f.add_subplot(n // col, 2 * col, i + 2)
        plt.imshow(patch_images[i // 2])
        plt.axis('off')

        plt.title(f'True label: {short_mapping_dict[true_lab[i // 2]]},\n'
                  f'predict label: {short_mapping_dict[pred_lab_trans[i // 2]]}',
                  fontdict={'fontsize': 8, 'fontweight': 'medium'})
    f.subplots_adjust(left=None,
                      bottom=None,
                      right=None,
                      top=None,
                      wspace=0.,
                      hspace=0.2)
    plt.savefig(f'{name}.png')
    plt.show()


def get_clipped_heatmap(model, img, true_label, pred_label):
    alpha = 0.05
    beta = -1
    x = model(Tensor(img))[0]
    x = Tensor.tanh(x * alpha + beta)
    heat2_true = x.cpu().detach().numpy()[:, :, true_label]

    heat2_pred = x.cpu().detach().numpy()[:, :, pred_label]

    return {'heat_true': heat2_true, 'heat_pred': heat2_pred}


def get_simple_heatmap(model, img, true_label, pred_label):
    x = model(Tensor(img))[0]
    heat2_true = x.cpu().detach().numpy()[:, :, true_label]

    heat2_pred = x.cpu().detach().numpy()[:, :, pred_label]

    return {'heat_true': heat2_true, 'heat_pred': heat2_pred}


def plot_cool_heatmap(model_cut, img, true_label, pred_label,
                      filename='default_pic.png',
                      short_mapping_dict=SHORT_MAPPING_DICT):

    clipped_heatmap = get_clipped_heatmap(model_cut, img, true_label, pred_label)
    simple_heatmap = get_simple_heatmap(model_cut, img, true_label, pred_label)
    pixel_img = cv2.resize(np.moveaxis(img[0], 0, 2), dsize=(24, 24))
    original_img = np.moveaxis(img[0], 0, 2)

    fig = plt.figure(figsize=(15, 10))

    vmin_cbn = min(simple_heatmap['heat_pred'].min(), simple_heatmap['heat_true'].min())
    vmax_cbn = max(simple_heatmap['heat_pred'].max(), simple_heatmap['heat_true'].max())

    a = fig.add_subplot(2, 3, 1)
    an = plt.imshow(original_img)

    a = fig.add_subplot(2, 3, 2)
    an = plt.imshow(simple_heatmap['heat_true'], vmin=vmin_cbn, vmax=vmax_cbn)
    plt.colorbar(an, shrink=0.8)
    plt.title(f"Bagnet Heat Map: true class {short_mapping_dict[true_label]}")

    a = fig.add_subplot(2, 3, 3)
    an = plt.imshow(simple_heatmap['heat_pred'], vmin=vmin_cbn, vmax=vmax_cbn)
    plt.colorbar(an, shrink=0.8)
    plt.title(f"Bagnet Heat Map: predict class {short_mapping_dict[pred_label]}")

    a = fig.add_subplot(2, 3, 4)
    an = plt.imshow(pixel_img)

    a = fig.add_subplot(2, 3, 5)
    an = plt.imshow(clipped_heatmap['heat_true'], vmin=-1, vmax=1)
    plt.colorbar(an, shrink=0.8)
    plt.title(f"CBN Heat Map: true class {short_mapping_dict[true_label]}")

    a = fig.add_subplot(2, 3, 6)
    an = plt.imshow(clipped_heatmap['heat_pred'], vmin=-1, vmax=1)
    plt.colorbar(an, shrink=0.8)
    plt.title(f"CBN Heat Map: predict class {short_mapping_dict[pred_label]}")

    plt.savefig(filename)
    plt.show()

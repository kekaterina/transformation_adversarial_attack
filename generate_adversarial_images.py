import sys

sys.path.append('../')

from preprocess import get_dataloader

from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch import Tensor, FloatTensor, LongTensor, device, cuda, no_grad

import numpy as np

from attacks import PgdSticker
from main import attack_step
from torch.utils.tensorboard import SummaryWriter


def get_adversarial_patch_pictures_by_art(model, x, y, filename='test', patch_scale=30,
                                  nb_classes=10, batch_size=5, max_iter=500,
                                  patch_type='circle', rotation_max=22.5, patch_location=None, learning_rate=5.0):
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=nb_classes,
        device_type='cpu'
    )

    #dataloader = get_dataloader(x, y, device='cpu', batch_size=batch_size, shuffle=False)
    adversarial_res = []

    for num, im in enumerate(x):
        la = y[num]
        attack = AdversarialPatchPyTorch(classifier, batch_size=batch_size, patch_shape=(3, patch_scale, patch_scale),
                                         max_iter=max_iter, patch_type=patch_type,
                                         rotation_max=rotation_max, learning_rate=learning_rate,
                                         patch_location=patch_location)
        patch, patch_mask = attack.generate(x=im, y=la)

        res = attack.apply_patch(x=im, scale=0.1)#, patch_external=Tensor(patch))
        adversarial_res.append(res)

    results_all = np.concatenate(adversarial_res)
    np.save(filename, results_all)

    return results_all


def get_adversarial_patch_pictures_by_custom(model, images, labels, filename,
                                             sticker_size, device='cuda', max_iters=40):
    writer = SummaryWriter(log_dir='output_exp')
    dataloader = get_dataloader(images, labels, device='cuda', batch_size=1, shuffle=False)

    attack = PgdSticker(model=model,
                        eps=1.,
                        alpha=0.8, #2/255,
                        iters=max_iters,
                        target=False,
                        )
    pred_labels, target_labels, adv_images = attack_step(
        model=model,
        attack=attack,
        dataloader=dataloader,
        acceptable_labels=None,
        sticker_size=sticker_size,
        im_shape=224,
        targeted=False,
        device=device,
        output_path=filename,
        writer=writer,
        prefix='',
    )

    return {'adv_images': adv_images,
            'pred_labels': pred_labels,
            'target_labels': target_labels}

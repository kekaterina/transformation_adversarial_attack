from copy import deepcopy
from advertorch.attacks.spsa import spsa_grad, linf_clamp_
from advertorch.attacks import LinfSPSAAttack
from advertorch.attacks.utils import MarginalLoss
from torch.nn import CrossEntropyLoss
from torch import Tensor
import torch
import numpy as np


def get_adv_images(images, labels, sticker_size, im_shape, attack, shift=3):
    #sticker_size*shift, sticker_size*shift это надо потом добавить, пока что результаты без шифта получены
    for row in range(0, im_shape[0] - sticker_size, sticker_size):
        for col in range(0, im_shape[1] - sticker_size, sticker_size):
            place = (row, col)

            adv_images, success, pred_top = attack.perturb(
                x=images, sticker_size=sticker_size, y=labels, place=place,
            )
            if success:
                return {
                    'adv_images': adv_images,
                    'success': success,
                    'pred_top': pred_top,
                }
    return {'adv_images': adv_images, 'success': success, 'pred_top': pred_top}


class PgdSticker:
    def __init__(self, model, eps, alpha, iters, target, loss_fn=None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.target = target
        if loss_fn is None:
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def perturb(self, x, y, sticker_size, place, is_batch=False):
        if self.target:

            def loss_fn(*args):
                return self.loss_fn(*args)

        else:

            def loss_fn(*args):
                return -self.loss_fn(*args)
        if is_batch:
            adv_images, success, pred_top = self.sticker_pgd_attack_with_batch(
                images=x,
                y=y,
                loss=loss_fn,
                sticker_size=sticker_size,
                place=place,
            )
        else:
            adv_images, success, pred_top = self.sticker_pgd_attack(
                images=x,
                y=y,
                loss=loss_fn,
                sticker_size=sticker_size,
                place=place,
            )
        return adv_images, success, pred_top

    def sticker_pgd_attack(
        self,
        images,
        y,
        loss,
        sticker_size=20,
        place=(0, 0),
        top=1, #5,
    ):
        place_i, place_j = place

        new_pic = deepcopy(images.data)
        true_log = self.model(new_pic)
        true_log = true_log.cpu().detach().numpy()[0]
        top_lab = np.argsort(true_log)[-1 * top :]

        for i in range(self.iters):
            new_pic = self.pgd_step(
                images=new_pic,
                y=y,
                loss_fn=loss,
                place_i=place_i,
                place_j=place_j,
                sticker_size=sticker_size,
            )

            pred = self.model(new_pic)
            pred = pred[0].cpu().detach().numpy()

            # сделать так, чтобы возвращался список успехов. если есть хотя бы один, то закончили подбирать.
            if self.target:
                success = succsess_metric_top_target(
                    y=y[0],
                    pred=pred,
                )
                if success or i == (self.iters - 1):
                    print(f'iter return {i}')
                    return new_pic, success, np.argmax(pred)

            else:
                success = without_top_succsess_metric_top_untarget(
                    y=y[0],
                    pred=pred,
                )
                if success or i == (self.iters - 1):
                    print(f'iter return {i}')
                    return new_pic, success, np.argmax(pred)

    def pgd_step(self, images, y, loss_fn, place_i, place_j, sticker_size):
        images.requires_grad = True
        outputs = self.model(images)

        self.model.zero_grad()
        loss = loss_fn(outputs, y)
        loss.backward()
        images.requires_grad = False

        adv_images = images - self.alpha * images.grad.sign()
        eta = Tensor.clamp(adv_images - images, min=-self.eps, max=self.eps)
        sticker = Tensor.clamp(images + eta, min=0, max=1)[
            :,
            :,
            place_i : place_i + sticker_size,
            place_j : place_j + sticker_size,
        ]

        images[
            :,
            :,
            place_i : place_i + sticker_size,
            place_j : place_j + sticker_size,
        ] = sticker

        return images

    def pgd_step_with_batch(self, images, y, loss_fn, place, sticker_size):
        images.requires_grad = True
        outputs = self.model(images)

        self.model.zero_grad()
        loss = loss_fn(outputs, y)
        loss.backward()
        images.requires_grad = False

        adv_images = images - self.alpha * images.grad.sign()
        eta = Tensor.clamp(adv_images - images, min=-self.eps, max=self.eps)
        print(eta.shape, images.shape)
        for num, place_coor in enumerate(place):
            print(place_coor)
            print(num)
            place_i = place_coor[0].cpu().numpy()
            place_j = place_coor[1].cpu().numpy()
            
            print(images[num].shape, place_i, place_j)
            sticker = Tensor.clamp(images[num] + eta[num], min=0, max=1)[
                :,
                place_i : place_i + sticker_size,
                place_j : place_j + sticker_size,
            ]

            images[
                num,
                :,
                place_i : place_i + sticker_size,
                place_j : place_j + sticker_size,
            ] = sticker

        return images

    def sticker_pgd_attack_with_batch(
        self,
        images,
        y,
        loss,
        sticker_size=20,
        place=(0, 0),
        top=1, #5,
    ):

        new_pic = deepcopy(images.data)
        true_log = self.model(new_pic)
        true_log = true_log.cpu().detach().numpy()[0]
        top_lab = np.argsort(true_log)[-1 * top :]
        preds = []

        for i in range(self.iters):
            new_pic = self.pgd_step_with_batch(
                images=new_pic,
                y=y,
                loss_fn=loss,
                place=place,
                sticker_size=sticker_size,
            )

        pred = self.model(new_pic)
        pred = pred.cpu().detach().numpy()
        preds.append(np.argmax(pred, axis=1))

            # сделать так, чтобы возвращался список успехов. если есть хотя бы один, то закончили подбирать.
            #if self.target:
            #    success = succsess_metric_top_target(
            #        y=y[0],
            #        pred=pred,
            #    )
            #    if success or i == (self.iters - 1):
            #        print(f'iter return {i}')
            #        return new_pic, success, np.argmax(pred)

            #else:
            #    success = without_top_succsess_metric_top_untarget(
            #        y=y[0],
            #        pred=pred,
            #    )
            #    if success or i == (self.iters - 1):
            #        print(f'iter return {i}')
            #        return new_pic, success, np.argmax(pred)
        return new_pic, None, preds

class SpsaSticker(LinfSPSAAttack):
    """
    predict: model for prediction
    eps: epsilon for perturbation in attack
    classic: type of perturb function. If classic is True, then use function
    with margina loss and spsa gradient. Else use function with combo of
    `pgd-step` and spsa gradient.
    alpha: scaling parameter.
    delta scaling parameter for spsa gradient:
    """

    def __init__(
        self,
        predict,
        eps,
        delta=0.1,
        lr=0.01,
        nb_iter=40,
        nb_sample=128,
        max_batch_size=64,
        targeted=False,
        loss_fn=None,
        clip_min=0.0,
        clip_max=1.0,
        classic=False,
        alpha=0.05,
    ):
        super().__init__(
            predict,
            eps,
            delta=delta,
            lr=lr,
            nb_iter=nb_iter,
            nb_sample=nb_sample,
            max_batch_size=max_batch_size,
            targeted=targeted,
            loss_fn=loss_fn,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.classic = classic
        self.alpha = alpha

        if classic:
            self.loss_fn = MarginalLoss(reduction='none')

    def perturb(self, x, y=None, sticker_size=20, place=(0, 0)):
        """Perturbs the input `x` based on SPSA attack.
        :param x: input tensor.
        :param y: label tensor (default=`None`). if `self.targeted` is `False`,
                  `y` is the ground-truth label. if it's `None`, then `y` is
                  computed as the predicted label of `x`.
                  if `self.targeted` is `True`, `y` is the target label.
        x, y = self._verify_and_process_inputs(x, y)
        """

        if self.targeted:

            def loss_fn(*args):
                return self.loss_fn(*args)

        else:

            def loss_fn(*args):
                return -self.loss_fn(*args)

        if self.classic:
            adv_images, success, pred = self.classic_spsa_perturb(
                loss_fn=loss_fn,
                x=x,
                y=y,
                sticker_size=sticker_size,
                place=place,
            )
        else:
            adv_images, success, pred = self.spsa_perturb(
                loss_fn=loss_fn,
                x=x,
                y=y,
                sticker_size=sticker_size,
                place=place,
            )
        return adv_images, success, pred

    def spsa_perturb(
        self, loss_fn, x, y, sticker_size=100, place=(0, 0), top=1
    ):
        """Perturbs the input `x` based on SPSA attack.
        :param predict: predict function (single argument: input).
        :param loss_fn: loss function (dual arguments: output, target).
        :param x: input argument for function `predict`.
        :param y: target argument for function `loss_fn`.
        :param eps: the L_inf budget of the attack.
        :param delta: scaling parameter of SPSA.
        :param lr: the learning rate of the `Adam` optimizer.
        :param nb_iter: number of iterations of the attack.
        :param nb_sample: number of samples for the SPSA gradient approximation.
        :param max_batch_size: maximum batch size to be evaluated at once.
        :param clip_min: upper bound of image values.
        :param clip_max: lower bound of image values.
        :return: the perturbated input.
        """

        place_i, place_j = place

        sticker = x[
            :,
            :,
            place_i : place_i + sticker_size,
            place_j : place_j + sticker_size,
        ]
        dx = torch.zeros_like(sticker)
        optimizer = torch.optim.Adam([dx], lr=self.lr)
        new_x = deepcopy(
            x.data[
                :,
                :,
                place_i : place_i + sticker_size,
                place_j : place_j + sticker_size,
            ]
        )
        new_pic = deepcopy(x.data)
        true_log = self.predict(new_pic)
        true_log = true_log.cpu().detach().numpy()[0]
        ind = np.argpartition(true_log, -1 * top)[-1 * top :]
        top_lab = ind[np.argsort(true_log[ind])]

        for it in range(self.nb_iter):
            new_pic, new_x = self.spsa_step(
                new_x=new_x,
                new_pic=new_pic,
                y=y,
                optimizer=optimizer,
                loss_fn=loss_fn,
                place_i=place_i,
                place_j=place_j,
                sticker_size=sticker_size,
            )

            pred = self.predict(new_pic)[0]
            pred = pred.cpu().detach().numpy()

            if not self.targeted:
                success = without_top_succsess_metric_top_untarget(
                    y=y[0],
                    pred=pred,
                )
                if success or (it == self.nb_iter - 1):
                    print(f'iter return {it}')
                    return new_pic, success, np.argmax(pred)

            else:
                success = succsess_metric_top_target(
                    y=y[0],
                    pred=pred,
                )
                if success or (it == self.nb_iter - 1):
                    print(f'iter return {it}')
                    return new_pic, success, np.argmax(pred)

    def spsa_step(
        self,
        new_x,
        new_pic,
        y,
        optimizer,
        loss_fn,
        place_i,
        place_j,
        sticker_size,
    ):
        optimizer.zero_grad()
        grad = spsa_grad(
            self.predict,
            loss_fn,
            new_x,
            y,
            self.delta,
            self.nb_sample,
            self.max_batch_size,
        )

        dx = self.alpha * grad
        new_x -= dx
        new_x = Tensor.clamp(new_x, self.clip_min, self.clip_max)

        new_pic[
            :,
            :,
            place_i : place_i + sticker_size,
            place_j : place_j + sticker_size,
        ] = new_x

        return new_pic, new_x

    def classic_spsa_step(
        self,
        dx,
        x,
        y,
        optimizer,
        loss_fn,
        place_i,
        place_j,
        sticker_size,
    ):
        optimizer.zero_grad()
        dx.grad = spsa_grad(
            self.predict,
            loss_fn,
            dx,
            y,
            self.delta,
            self.nb_sample,
            self.max_batch_size,
        )
        optimizer.step()

        dx = linf_clamp_(dx, x, self.eps, self.clip_min, self.clip_max)

        tmp = deepcopy(
            dx[
                :,
                :,
                place_i : place_i + sticker_size,
                place_j : place_j + sticker_size,
            ]
        )
        dx = torch.zeros_like(dx)
        dx[
            :,
            :,
            place_i : place_i + sticker_size,
            place_j : place_j + sticker_size,
        ] = tmp

        return x + dx

    def classic_spsa_perturb(
        self, loss_fn, x, y, sticker_size=100, place=(0, 0), top=1
    ):
        place_i, place_j = place
        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)

        true_log = self.predict(x)
        true_log = true_log.cpu().detach().numpy()[0]
        ind = np.argpartition(true_log, -1 * top)[-1 * top :]
        top_lab = ind[np.argsort(true_log[ind])]

        for it in range(self.nb_iter):

            new_x = self.classic_spsa_step(
                dx=dx,
                x=x,
                y=y,
                optimizer=optimizer,
                loss_fn=loss_fn,
                place_i=place_i,
                place_j=place_j,
                sticker_size=sticker_size,
            )
            pred = self.predict(new_x)[0]
            pred = pred.cpu().detach().numpy()

            if not self.targeted:
                success = without_top_succsess_metric_top_untarget(
                    y=y[0],
                    pred=pred,
                )
                if success or (it == self.nb_iter - 1):
                    return new_x, success, np.argmax(pred)

            else:
                success = succsess_metric_top_target(
                    y=y[0],
                    pred=pred,
                )
                if success or (it == self.nb_iter - 1):
                    return new_x, success, np.argmax(pred)


def succsess_metric_top_target(y, pred):
    succsess = False
    if y == np.argmax(pred):
        succsess = True
    return succsess


def success_metric_top_untarget(pred, top_lab, top=1):
    true_label = top_lab[-1]
    ind = np.argpartition(pred, -1 * top)[-1 * top :]
    pred_top = ind[np.argsort(pred[ind])]
    success = True

    if true_label in pred_top:
        success = False
    return success, pred_top[-1]


def without_top_succsess_metric_top_untarget(y, pred):
    succsess = False
    if y != np.argmax(pred):
        succsess = True
    return succsess


def get_adv_images_with_batch(images, labels, sticker_size, attack, place, id):

    adv_images, success, pred_top = attack.perturb(
        x=images, sticker_size=sticker_size, y=labels, place=place, is_batch=True,
    )
    #if success:
    #    return {
    #        'adv_images': adv_images,
    #        'success': success,
    #        'pred_top': pred_top,
    #    }
    return {'adv_images': adv_images, 'success': success, 'pred_top': pred_top}

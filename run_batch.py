from main import attack_step_with_batch
from torch import Tensor, FloatTensor, LongTensor, device, cuda, no_grad
import time
import torchvision
from torch import load as torch_load
import numpy as np
from attacks import SpsaSticker, PgdSticker, get_adv_images, get_adv_images_with_batch

path = 'models/resnet18_10_classes_state_dict.pth'
#path = '/dump/ekurdenkova/article_data_dump/cbn_10_classes_state_dict.pth'
#model = ClippedBagNet(num_classes=10, aggregation='cbn')
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
weights = torch_load(path)
model.load_state_dict(weights)
model.eval()
model.cuda()

images = np.load('my_imagenet/images_test_part_imagenet_10_classes_1000_pic.npy')
#patch_images = np.load('/dump/ekurdenkova/gitlab/No_fork/guardiann/guardiann/bagnet/transformations/customPGD/resnet_32_combo_images.npy')
true_lab = np.load('my_imagenet/labels_test_part_imagenet_10_classes_1000_pic.npy')

im_shape = (224, 224)
sticker_size = 32
batch_size = 20
device ='cuda'
targeted = False
attack_iters = 40
pgd_alpha = 0.8

attack = PgdSticker(
    model=model,
    eps=1,
    alpha=pgd_alpha,
    iters=attack_iters,
    target=targeted,
)

attack_step_with_batch(
    model=model,
    attack=attack,
    images=images,
    labels=true_lab,
    acceptable_labels=None,
    sticker_size=sticker_size,
    im_shape=im_shape,
    targeted=targeted,
    device=device,
    batch_size=40)

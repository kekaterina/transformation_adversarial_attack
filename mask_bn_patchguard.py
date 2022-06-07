import torch
import torch.backends.cudnn as cudnn

import add_lib.PatchGuard.nets.bagnet as patch_bagnet
from add_lib.PatchGuard.utils.defense_utils import *

import os
import argparse
from tqdm import tqdm
import numpy as np
from math import ceil

from add_lib.PatchGuard.mask_bn import get_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", default='checkpoints', type=str, help="path to checkpoints")
parser.add_argument('--data_dir', default='data', type=str, help="path to data")
parser.add_argument('--dataset', default='imagenette', choices=('imagenette', 'imagenet', 'cifar'), type=str,
                    help="dataset")
parser.add_argument("--model", default='bagnet17', type=str, help="model name")
parser.add_argument("--clip", default=-1, type=int,
                    help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr", default='none', type=str, help="aggregation methods. set to none for local feature")
parser.add_argument("--skip", default=1, type=int, help="number of example to skip")
parser.add_argument("--thres", default=0.0, type=float, help="detection threshold for robust masking")
parser.add_argument("--patch_size", default=-1, type=int, help="size of the adversarial patch")
parser.add_argument("--m", action='store_true', help="use robust masking")
parser.add_argument("--cbn", action='store_true', help="use cbn")

args = parser.parse_args()

MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, args.dataset)
DATASET = args.dataset

val_dataset_, class_names = get_dataset(DATASET, DATA_DIR)
skips = list(range(0, len(val_dataset_), args.skip))
val_dataset = torch.utils.data.Subset(val_dataset_, skips)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False)

# build and initialize model
device = 'cuda'  # if torch.cuda.is_available() else 'cpu'

if args.clip > 0:
    clip_range = [0, args.clip]
else:
    clip_range = None

if 'bagnet17' in args.model:
    model = patch_bagnet.bagnet17(pretrained=True, clip_range=clip_range, aggregation=args.aggr)
    rf_size = 17
elif 'bagnet33' in args.model:
    model = patch_bagnet.bagnet33(pretrained=True, clip_range=clip_range, aggregation=args.aggr)
    rf_size = 33
elif 'bagnet9' in args.model:
    model = patch_bagnet.bagnet9(pretrained=True, clip_range=clip_range, aggregation=args.aggr)
    rf_size = 9

if DATASET == 'imagenet':
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR, args.model + '_net.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    args.patch_size = args.patch_size if args.patch_size > 0 else 32

rf_stride = 8
window_size = ceil((args.patch_size + rf_size - 1) / rf_stride)
print("window_size", window_size)

model = model.to(device)
model.eval()
cudnn.benchmark = True

accuracy_list = []
result_list = []
clean_corr = 0

for data, labels in tqdm(val_loader):

    data = data.to(device)
    labels = labels.numpy()
    output_clean = model(data).detach().cpu().numpy()  # logits

    # note: the provable analysis of robust masking is cpu-intensive and can take some time to finish
    # you can dump the local feature and do the provable analysis with another script so that GPU mempry is not always occupied
    for i in range(len(labels)):
        if args.m:  # robust masking
            local_feature = output_clean[i]
            result = provable_masking(local_feature, labels[i], thres=args.thres,
                                      window_shape=[window_size, window_size])
            result_list.append(result)
            clean_pred = masking_defense(local_feature, thres=args.thres, window_shape=[window_size, window_size])
            clean_corr += clean_pred == labels[i]

        elif args.cbn:  # cbn
            # note that cbn results reported in the paper is obtained with vanilla BagNet (without provable adversrial training), since
            # the provable adversarial training is proposed in our paper. We will find that our training technique also benifits CBN
            result = provable_clipping(output_clean[i], labels[i], window_shape=[window_size, window_size])
            result_list.append(result)
            clean_pred = clipping_defense(output_clean[i])
            clean_corr += clean_pred == labels[i]
    acc_clean = np.sum(np.argmax(np.mean(output_clean, axis=(1, 2)), axis=1) == labels)
    accuracy_list.append(acc_clean)

cases, cnt = np.unique(result_list, return_counts=True)
print("Provable robust accuracy:", cnt[-1] / len(result_list) if len(cnt) == 3 else 0)
print("Clean accuracy with defense:", clean_corr / len(result_list))
print("Clean accuracy without defense:", np.sum(accuracy_list) / len(val_dataset))
print("------------------------------")
print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):", cases)
print("Provable analysis breakdown", cnt / len(result_list))
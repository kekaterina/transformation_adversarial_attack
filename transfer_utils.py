import pandas as pd
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, FloatTensor, LongTensor, device, cuda, no_grad
from torch import load as torch_load

from sklearn import metrics
import numpy as np

from preprocess import ImagenetDataset

from add_lib.PatchGuard.utils.defense_utils import masking_defense, provable_masking, clipping_defense, provable_clipping
from math import ceil
from torchvision.models import resnet18

from constant import CUSTOM_MODEL_LOAD_PARAMS, DEFAULT_MODEL_LIST


def predict_accuracy(model, x, y, num_classes=10, batch_size=1, device='cuda'):
    cuda.empty_cache()
    preds = np.zeros((y.shape[0], num_classes))
    test_dataset = ImagenetDataset(Tensor(x), LongTensor(y))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    with torch.no_grad():
        for i, (im, la) in enumerate(test_dataloader):
            im = Tensor(im).to(device)
            pred = model(im)
            preds[i * batch_size:(i + 1) * batch_size, :] = pred.cpu().detach().numpy()

        acc = metrics.accuracy_score(y, np.argmax(preds, axis=1))
    del test_dataloader
    return acc


def predict_patchguard(model, x, y, batch_size=1, kind='m', rf_size=33, thres=0.0, rf_stride=8,
                       patch_size=100, device='cuda'):
    if patch_size==60:
        window_size = 5
    else:
        window_size = int(ceil((patch_size + rf_size - 1) / rf_stride))
    cuda.empty_cache()

    test_dataset = ImagenetDataset(Tensor(x), LongTensor(y))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    result_list = []
    pred_list = []
    clean_pred_list = []

    accuracy_list = []
    clean_corr = 0

    for i, (im, la) in enumerate(test_dataloader):
        im = Tensor(im).to(device)
        output_clean = model(im).detach().cpu().numpy()
        print(f'output_clean {output_clean.shape}')
        print(f'window_size {window_size}')
        print(f'la {la}')

        for j in range(len(la)):
            print(f'lai {la[j]}')
            if kind == 'm':  # robust masking
                local_feature = output_clean[j]
                result = provable_masking(local_feature,
                                          la[j],
                                          thres=thres,
                                          window_shape=[window_size, window_size])
                result_list.append(result)
                clean_pred = masking_defense(local_feature,
                                             thres=thres,
                                             window_shape=[window_size, window_size])
                clean_corr += clean_pred == la[j]

            elif kind == 'cbn':  # cbn
                # note that cbn results reported in the paper is obtained with vanilla BagNet (without provable adversrial training), since
                # the provable adversarial training is proposed in our paper. We will find that our training technique also benifits CBN
                result = provable_clipping(output_clean[j],
                                           la[j],
                                           window_shape=[window_size, window_size])
                result_list.append(result)
                clean_pred = clipping_defense(output_clean[j])
                clean_corr += clean_pred == la[j]

            pred = np.argmax(np.mean(output_clean, axis=(1, 2)), axis=1)
            pred_list.append(pred)
            clean_pred_list.append(clean_pred)

            acc_clean = np.sum(np.argmax(np.mean(output_clean, axis=(1, 2)), axis=1) == la)
            accuracy_list.append(acc_clean)

    del test_dataloader

    d = {}
    print(f'result_list {result_list}')
    print(f'clean_pred {kind}, {clean_corr, clean_pred_list}')
    cases, cnt = np.unique(result_list, return_counts=True)
    print("Provable robust accuracy:", cnt[-1] / len(result_list) if len(cnt) == 3 else 0)
    print("Clean accuracy with defense:", clean_corr / len(result_list))
    print("Clean accuracy without defense:", np.sum(accuracy_list) / len(test_dataset))
    print("------------------------------")
    print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):", cases)
    print("Provable analysis breakdown", cnt / len(result_list))

    d['Provable robust accuracy'] = cnt[-1] / len(result_list) if len(cnt) == 3 else 0
    d['Clean accuracy with defense'] = clean_corr / len(result_list)
    d['Clean accuracy without defense'] = np.sum(accuracy_list) / len(test_dataset)

    return pred_list, clean_pred_list, result_list, accuracy_list, d


def update_state_dict(path, model):
    # original saved file with DataParallel
    state_dict = torch.load(path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def get_results_transfer_attack(vanilla_model, patchguard_model, x_data, y, thres=0.0,patch_size=100, device='cuda'):
    all_results = {}
    for data_key in x_data.keys():
        vanilla_results = {}
        for vanilla_model_key in vanilla_model:
            model = custom_load_model(path=CUSTOM_MODEL_LOAD_PARAMS[vanilla_model_key]['path'],
                                      agg=CUSTOM_MODEL_LOAD_PARAMS[vanilla_model_key]['agg'],
                                      model_func=CUSTOM_MODEL_LOAD_PARAMS[vanilla_model_key]['model_func'],
                                      need_update_weights=CUSTOM_MODEL_LOAD_PARAMS[vanilla_model_key][
                                          'need_update_weights'],
                                      num_classes=CUSTOM_MODEL_LOAD_PARAMS[vanilla_model_key]['num_classes'],
                                      device=device)
            model.eval()  # возможно надо
            acc = predict_accuracy(model, x_data[data_key], y)
            vanilla_results[vanilla_model_key] = acc

        patchguard_results = {}
        for patchguard_model_key in patchguard_model:
            for key_mask in ['m', 'cbn']:
                model = custom_load_model(path=CUSTOM_MODEL_LOAD_PARAMS[patchguard_model_key]['path'],
                                          agg=CUSTOM_MODEL_LOAD_PARAMS[patchguard_model_key]['agg'],
                                          model_func=CUSTOM_MODEL_LOAD_PARAMS[patchguard_model_key]['model_func'],
                                          need_update_weights=CUSTOM_MODEL_LOAD_PARAMS[patchguard_model_key][
                                              'need_update_weights'],
                                          num_classes=CUSTOM_MODEL_LOAD_PARAMS[patchguard_model_key]['num_classes'],
                                          device=device)
                model.eval()  # возможно надо
                pred_list, clean_pred_list, result_list, accuracy_list, d = predict_patchguard(
                    model=model,
                    x=x_data[data_key],
                    y=y,
                    batch_size=1,
                    kind=key_mask,
                    rf_size=CUSTOM_MODEL_LOAD_PARAMS[patchguard_model_key]['rf_size'],
                    thres=thres,
                    rf_stride=CUSTOM_MODEL_LOAD_PARAMS[patchguard_model_key]['rf_stride'],
                    patch_size=patch_size, device=device)

                patchguard_results[f'{patchguard_model_key}_{key_mask}'] = d
        all_results[data_key] = {'vanilla_results': vanilla_results, 'patchguard_results': patchguard_results}
    return all_results


def convert_dict_to_df(all_results):
    new_dict = {}
    for outerKey, innerDict in all_results.items():
        for innerKey, values in innerDict.items():
            if innerKey == 'patchguard_results':
                for theardKey, theardValues in innerDict[innerKey].items():
                    for fourthKey, fourthValues in theardValues.items():
                        new_dict[(outerKey, innerKey, theardKey, fourthKey)] = fourthValues
            else:
                new_dict[(outerKey, innerKey, 'theard', 'fourth')] = values

    results_df = pd.DataFrame(new_dict)
    return results_df


def get_df_per_data_results(exps):
    big_d = {}
    dats = []
    for e in exps.columns:
        if e[0] not in dats:
            dats.append(e[0])

    for data in dats:
        models = []
        accs = []
        patch_model = []
        vanilla_model = []
        d = {}

        for e in exps[data]['vanilla_results'].index:
            models.append(e)
            vanilla_model.append(e)
        accs += list(exps[data]['vanilla_results'].values.ravel())

        for e in exps[data]['patchguard_results'].columns:
            if e[0] not in models:
                models.append(e[0])
                patch_model.append(e[0])
                accs.append(None)

        patch_col = []
        for e in exps[data]['patchguard_results'][patch_model[0]].columns:
            d[e] = [None] * len(vanilla_model)
            patch_col.append(e)
        patch_col = list(set(patch_col))

        for b in patch_model:
            for e in patch_col:
                d[e].append(exps[data]['patchguard_results'][b][e][0])

        d['model'] = models
        d['accuracy'] = accs
        big_d[data] = d

    return big_d


def get_little_results_transfer_attack(all_results):
    exps = convert_dict_to_df(all_results)
    big_df = get_df_per_data_results(exps)

    keys = list(big_df.keys()) #keys is ['original', 'adversarial_patch_images_resnet_32', 'adversarial_patch_images_cbn_32']
    transfer_attack_df = pd.DataFrame({'model': big_df[keys[0]]['model']})

    for col in ['accuracy', 'Clean accuracy with defense']:
        for key in keys:
            transfer_attack_df[f'{key}_{col}'] = big_df[key][col]
    transfer_attack_df.to_csv('transfer_attack_df_test.csv')
    return transfer_attack_df


def quadric_df(big_df, model, patch_size, model_list=None):
    if model_list is None:
        models = DEFAULT_MODEL_LIST
    else:
        models = model_list
    big_df_cp = big_df.copy()
    big_df_cp.loc[:, 'model'] = np.where(big_df_cp.model == 'patchCBN', 'adv_cbn', big_df_cp.model)

    arr = np.zeros((5, len(models)))
    for i, col in enumerate(models):
        for j, row in enumerate(['clean', 'rotate', 'dark', 'gauss', 'combo']):
            if j == 0:
                name = f'adversarial_patch_images_{col}_{patch_size}_accuracy' # подозрительное место
            else:
                name = f'{col}_{patch_size}_adv_{row}_images_accuracy'
            arr[j][i] = big_df_cp[big_df_cp.model == model][name].values[0]

    new_arr = np.zeros((5, len(models)+1))

    new_arr[0, 0] = big_df_cp[big_df_cp.model == model]['original_accuracy'].values[0]
    for i, col in enumerate(['rotate', 'dark', 'gauss', 'combo']):
        name = f'{model}_{patch_size}_{col}_images_accuracy'
        new_arr[i + 1, 0] = big_df_cp[big_df_cp.model == model][name].values[0]
    new_arr[:, 1:] = arr

    return new_arr


def create_all_quadric_df(df, patch_size, model_list=None, filename='test'):
    values_for_all_models = []
    if model_list is None:
        models = DEFAULT_MODEL_LIST
    else:
        models = model_list

    for model in models:
        values = quadric_df(df, model, patch_size, model_list)
        values_for_all_models.append(values)
    res_arr = np.concatenate(values_for_all_models)

    iterables = [models, ['clean image', 'rotate', 'dark', 'gauss', 'combo']]
    multi = pd.MultiIndex.from_product(iterables, names=["Model", "Transformation"])

    columns = ['original img'] + [f'adv_img_{model}' for model in model_list]
    result_df = pd.DataFrame(res_arr, index=multi, columns=columns)
    result_df.to_csv(f'{filename}.csv')
    return result_df


def for_patch_quadric_df(df, model, patch_size, model_list=None, mapping_model=None):
    if model_list is None:
        models = DEFAULT_MODEL_LIST
    else:
        models = model_list

    df_cp = df.copy()
    df_cp.loc[:, 'model'] = np.where(df_cp.model == 'patchCBN', 'adv_cbn', df_cp.model)

    adv_values_for_all_model = np.zeros((5, len(models)))
    for i, col in enumerate(models):
        for j, row in enumerate(['clean', 'rotate', 'dark', 'gauss', 'combo']):
            if j == 0:
                name = f'adversarial_patch_images_{col}_{patch_size}_Clean accuracy with defense' # подозрительное место
            else:
                name = f'{col}_{patch_size}_adv_{row}_images_Clean accuracy with defense'

            # это костыль по вытаскиванию значений из тензоров. надо изначально записывать числа, а не тензоры.
            val = str(df_cp[df_cp.model == model][name].values[0])
            start = val.find('(')
            end = -1
            adv_values_for_all_model[j][i] = val[start + 1:end]

    all_values_for_all_model = np.zeros((5, len(models)+1))

    val = str(df_cp[df_cp.model == model]['original_Clean accuracy with defense'].values[0])
    start = val.find('(')
    end = -1
    all_values_for_all_model[0, 0] = val[start + 1:end]
    for i, col in enumerate(['rotate', 'dark', 'gauss', 'combo']):
        name = f'{mapping_model[model]}_{patch_size}_{col}_images_Clean accuracy with defense'

        val = str(df_cp[df_cp.model == model][name].values[0])
        start = val.find('(')
        end = -1
        all_values_for_all_model[i + 1, 0] = val[start + 1:end]

    all_values_for_all_model[:, 1:] = adv_values_for_all_model
    return all_values_for_all_model


def for_patch_create_all_quadric_df(df, patch_size, model_list, filename='test'):
    values_for_all_model = []
    if model_list is None:
        model_list = DEFAULT_MODEL_LIST

    models = ['resnet_agg_none_cbn', 'resnet_agg_none_m',
              'clipped_agg_none_cbn', 'clipped_agg_none_m',
              'patch_cbn_agg_none_cbn', 'patch_cbn_agg_none_m']

    mapping_model = {'resnet_agg_none_cbn': 'resnet', 'resnet_agg_none_m': 'resnet',
                     'clipped_agg_none_m': 'cbn', 'clipped_agg_none_cbn': 'cbn',
                     'patch_cbn_agg_none_m': 'adv_cbn', 'patch_cbn_agg_none_cbn': 'adv_cbn'}

    final_model_list = []
    for model in models:
        if mapping_model[model] in model_list:
            final_model_list.append(model)
            values = for_patch_quadric_df(df, model, patch_size, model_list, mapping_model)
            values_for_all_model.append(values)

    values_for_all_model_array = np.concatenate(values_for_all_model)

    iterables = [final_model_list, ['clean image', 'rotate', 'dark', 'gauss', 'combo']]
    multi = pd.MultiIndex.from_product(iterables, names=["Model", "Transformation"])

    columns = ['original img'] + [f'adv_img_{model}' for model in model_list]
    result_df = pd.DataFrame(values_for_all_model_array, index=multi, columns=columns)
    result_df.to_csv(f'{filename}.csv')

    return result_df


def custom_load_model(path, agg, model_func=None, need_update_weights=False, num_classes=10, device='cuda'):
    if model_func == resnet18:
        model = model_func(num_classes=num_classes, pretrained=False)
    else:
        model = model_func(aggregation=agg, num_classes=num_classes)
    if need_update_weights:
        model = update_state_dict(path, model)
    else:
        wei = torch_load(path)
        model.load_state_dict(wei)
    model.eval().to(device)
    return model

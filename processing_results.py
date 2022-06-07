from sklearn.utils import shuffle
import pathlib
import torch

from transfer_utils import *
import argparse

from constant import MODEL_MAPPING, DATA_MAPPING, ADVERSARIAL_IMAGES_MAPPING, \
    OUTPUT_PATH, OUTPUT_PATH_TRANSFORMATION_IMAGE, PREFIX_NAME, \
    VANILLA_MODEL, PATCHGUARD_MODEL, SIMPLE_IMAGE_KEY, ADV_IMAGE_KEY, MODEL_KEY_FOR_PROCESSING_RESULTS, \
    OUTPUT_PATH_TRANSFORMATION_CSV, PATCH_ALL_QUADRIC_DF_OUTPUT, ALL_QUADRIC_DF_OUTPUT, \
    RANDOM_STATE

from main import str2bool

HARD_SIZE_FOR_DEBUGGING = 100


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
        choices=['custom', 'art'],
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
        '--random-shuffle',
        type=str2bool,
        default=False,
        help='if True, then images will be shuffle',
    )
    return parser.parse_args()


def get_data_dictionary(data_key, patch_scale, random_shuffle=False):
    x_data = {} #{'original': X_test[:1000]} #размер надо будет убрать из хардкода
    for key, values in data_key.items():
        if key == 'original':
            name = 'original'
            original_images = np.load(values)
            if random_shuffle:
                print('With random shuffle!')
                original_images = shuffle(original_images, random_state=RANDOM_STATE)
            x_data[name] = original_images[:HARD_SIZE_FOR_DEBUGGING]
        else:
            name = f'adversarial_patch_images_{key}_{patch_scale}'
            x_data[name] = np.load(values)[:HARD_SIZE_FOR_DEBUGGING]  #[:1000]
    return x_data


def get_transformation_file(path, size_key, model_key,
                            simple_image_key=SIMPLE_IMAGE_KEY,
                            adv_image_key=ADV_IMAGE_KEY):
    files = []
    file_key = []
    for model in model_key:
        #for size in size_key:
        for simple_img in simple_image_key:
            filename = path + f'{model}_{size_key}_{simple_img}.npy'
            files.append(filename)
            file_key.append(f'{model}_{size_key}_{simple_img}')
        for adv_img in adv_image_key:
            filename = path + f'{model}_{size_key}_{adv_img}.npy'
            files.append(filename)
            file_key.append(f'{model}_{size_key}_{adv_img}')
    return {'files': files, 'file_key': file_key}


def collect_result_transfer_attaсk_with_transformation(files, file_key, y, vanilla_model, patchguard_model,
                                                       device, patch_scale, output):
    # тут для каждой трансформации собираются отдельные результаты
    for i in range(len(files)):
        arr = np.load(files[i])
        x_data = {file_key[i]: arr[:HARD_SIZE_FOR_DEBUGGING]} #размер надо будет убрать из хардкода

        local_all_results = get_results_transfer_attack(vanilla_model=vanilla_model,
                                                        patchguard_model=patchguard_model,
                                                        x_data=x_data,
                                                        y=y,
                                                        device=device,
                                                        patch_size=patch_scale)

        local_little_big_df = get_little_results_transfer_attack(local_all_results)
        local_little_big_df.to_csv(f'{output}/{file_key[i]}_transfer_test.csv')


def merge_transform_attack_results(transfer_path, file_key):
    df = pd.read_csv(f'{transfer_path}/{file_key[0]}_transfer_test.csv').drop(['Unnamed: 0'], axis=1)

    for key in file_key[1:]:
        second_df = pd.read_csv(f'{transfer_path}/{key}_transfer_test.csv').drop(['Unnamed: 0'], axis=1)
        df = df.merge(second_df, on='model')
    return df


def main():
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Loading data')
    y = np.load(DATA_MAPPING['test_labels'])
    if args.random_shuffle:
        print('With random shuffle!')
        y = shuffle(y, random_state=RANDOM_STATE)
    y = y[:HARD_SIZE_FOR_DEBUGGING] #размер надо будет убрать из хардкода
    x_data = get_data_dictionary(data_key=ADVERSARIAL_IMAGES_MAPPING[args.patch_scale],
                                 patch_scale=args.patch_scale, random_shuffle=args.random_shuffle)

    print('Calculate results for transfer attack')
    all_results = get_results_transfer_attack(vanilla_model=VANILLA_MODEL,
                                              patchguard_model=PATCHGUARD_MODEL,
                                              x_data=x_data,
                                              y=y,
                                              device=device,
                                              patch_size=args.patch_scale)

    transfer_attack_df = get_little_results_transfer_attack(all_results)

    print('Calculate results for transfer attack with transformation')
    file_info = get_transformation_file(path=OUTPUT_PATH_TRANSFORMATION_IMAGE,
                                        size_key=args.patch_scale,
                                        model_key=MODEL_KEY_FOR_PROCESSING_RESULTS,
                                        simple_image_key=SIMPLE_IMAGE_KEY,
                                        adv_image_key=ADV_IMAGE_KEY)

    collect_result_transfer_attaсk_with_transformation(files=file_info['files'],
                                                       file_key=file_info['file_key'],
                                                       y=y,
                                                       vanilla_model=VANILLA_MODEL,
                                                       patchguard_model=PATCHGUARD_MODEL,
                                                       device=device,
                                                       patch_scale=args.patch_scale,
                                                       output=OUTPUT_PATH_TRANSFORMATION_CSV)

    print('Preparing dataframe with results')
    df = merge_transform_attack_results(transfer_path=OUTPUT_PATH_TRANSFORMATION_CSV,
                                        file_key=file_info['file_key'])
    m = df.merge(transfer_attack_df, on='model')

    create_all_quadric_df(df=m,
                          patch_size=args.patch_scale,
                          model_list=MODEL_KEY_FOR_PROCESSING_RESULTS,
                          filename=f'{ALL_QUADRIC_DF_OUTPUT}_{args.patch_scale}')

    for_patch_create_all_quadric_df(df=m,
                                    patch_size=args.patch_scale,
                                    model_list=MODEL_KEY_FOR_PROCESSING_RESULTS,
                                    filename=f'{PATCH_ALL_QUADRIC_DF_OUTPUT}_{args.patch_scale}')


if __name__ == '__main__':
    main()

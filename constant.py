from add_lib.PatchGuard.nets.resnet import resnet18 as resnet18_patch
from add_lib.PatchGuard.nets.bagnet import bagnet33
from torchvision.models import resnet18
from bagnet import ClippedBagNet


MODEL_MAPPING = {'resnet': '/home/kurdenkova/models/resnet18_10_classes_state_dict.pth',
                 'cbn': '/home/kurdenkova/models/cbn_10_classes_state_dict.pth',
                 'adv_cbn': '/home/kurdenkova/models/adv_cbn_10_classes_state_dict.pth',
                 'cbn_patch': '/home/kurdenkova/models/cbn_10_classes_state_dict.pth',
                 'resnet_patch': '/home/kurdenkova/models/resnet18_10_classes_state_dict.pth',
                 'adv_cbn_patch': '/home/kurdenkova/models/adv_cbn_10_classes_state_dict.pth',}


CUSTOM_MODEL_LOAD_PARAMS = {
    'resnet': {
        'model_func': resnet18,
        'path': MODEL_MAPPING['resnet'],
        'num_classes': 10,
        'agg': None,
        'need_update_weights': False,
        'rf_size': 7,
        'rf_stride': 8,
    },
    'resnet_agg_none': {
        'model_func': resnet18_patch,
        'path': MODEL_MAPPING['resnet'],
        'num_classes': 10,
        'agg': None,
        'need_update_weights': False,
        'rf_size': 7,
        'rf_stride': 8,
    },
    'cbn': {
        'model_func': ClippedBagNet,
        'path': MODEL_MAPPING['cbn'],
        'num_classes': 10,
        'agg': 'cbn',
        'need_update_weights': False,
        'rf_size': 33,
        'rf_stride': 8,
    },
    'clipped_agg_none': {
        'model_func': ClippedBagNet,
        'path': MODEL_MAPPING['cbn_patch'],
        'num_classes': 10,
        'agg': None,
        'need_update_weights': False,
        'rf_size': 33,
        'rf_stride': 8,
    },
    'patchCBN': {
        'model_func': bagnet33,
        'path': MODEL_MAPPING['adv_cbn'],
        'num_classes': 10,
        'agg': 'cbn',
        'need_update_weights': False,
        'rf_size': 33,
        'rf_stride': 8,
    },
    'patch_cbn_agg_none': {
        'model_func': bagnet33,
        'path': MODEL_MAPPING['adv_cbn_patch'],
        'num_classes': 10,
        'agg': None,
        'need_update_weights': False,
        'rf_size': 33,
        'rf_stride': 8,
    },
}


DATA_MAPPING = {'images': '~/my_imagenet/images_data_dump_10_class_1000_pic.npy',
                'labels': '~/my_imagenet/res_yys_10_class_1000_pic.npy',
                'test_images': '/home/kurdenkova/my_imagenet/images_test_part_imagenet_10_classes_1000_pic.npy',
                'test_labels': '/home/kurdenkova/my_imagenet/labels_test_part_imagenet_10_classes_1000_pic.npy', }


ADVERSARIAL_IMAGES_MAPPING = {
    100:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet': '/dump/ekurdenkova/article_data/images_without_transform/resnet_100_custom_adv_images_reshape.npy',
            'cbn': '/dump/ekurdenkova/article_data/images_without_transform/cbn_100_custom_adv_images.npy',
            'adv_cbn': '/dump/ekurdenkova/article_data/images_without_transform/adv_cbn_100_custom_adv_images.npy'
        },
    50:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet': '/space/kurdenkova/experiments_dump/exp_for_table_50_resnet/full_50_resnet_images.npy',
            'cbn': '/space/kurdenkova/experiments_dump/exp_for_table_50_cbn/full_50_cbn_images.npy',
        },
    40:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet': '/space/kurdenkova/experiments_dump/exp_for_table_40_resnet/full_40_resnet_images.npy',
            'cbn': '/space/kurdenkova/experiments_dump/exp_for_table_40_cbn/full_40_cbn_images.npy',
        },
    60:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet': '/space/kurdenkova/experiments_dump/exp_for_table_60_resnet/full_60_resnet_images.npy',
            'cbn': '/space/kurdenkova/experiments_dump/exp_for_table_60_cbn/full_60_cbn_images.npy',
        },
    32:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet':'/space/kurdenkova/experiments_dump/resnet_32_custom_adv_images.npy',
            'cbn':'/space/kurdenkova/experiments_dump/cbn_32_custom_adv_images.npy',
            'adv_cbn': '/space/kurdenkova/froms6/dump/article_data/images_without_transform/adv_cbn_32_custom_adv_images.npy' #article_data
        },
    20:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet':'/space/kurdenkova/experiments_dump/exp_for_table_20_resnet/full_20_resnet_images.npy',#'/dump/ekurdenkova/for_exp_with_random/images_without_transform/resnet_20_custom_adv_images.npy',
            'cbn':'/space/kurdenkova/experiments_dump/exp_for_table_20_cbn/full_20_cbn_images.npy',#'/dump/ekurdenkova/for_exp_with_random/images_without_transform/cbn_20_custom_adv_images.npy',
            'adv_cbn': '/space/kurdenkova/froms6/dump/article_data/images_without_transform/adv_cbn_20_custom_adv_images.npy'
        },
    10:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/resnet_10_custom_adv_images.npy',
            'cbn': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/cbn_10_custom_adv_images.npy', #article_data
        },
    15:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/resnet_15_custom_adv_images.npy',
            'cbn': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/cbn_15_custom_adv_images.npy', #for_exp_sizing
        },
    25:
        {
            'original': DATA_MAPPING['test_images'],
            'resnet': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/resnet_25_custom_adv_images.npy',
            'cbn': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/cbn_25_custom_adv_images.npy',
        },
    }

VANILLA_MODEL = ['resnet', 'cbn', 'patchCBN']
PATCHGUARD_MODEL = ['resnet_agg_none', 'clipped_agg_none', 'patch_cbn_agg_none']

SIMPLE_IMAGE_KEY = ['rotate_images', 'dark_images', 'gauss_images', 'combo_images']
ADV_IMAGE_KEY = ['adv_rotate_images', 'adv_dark_images', 'adv_gauss_images', 'adv_combo_images']
MODEL_KEY_FOR_PROCESSING_RESULTS = ['resnet', 'cbn', 'adv_cbn']
DEFAULT_MODEL_LIST = ['resnet', 'adv_cbn', 'cbn']

OUTPUT_PATH = 'output_exp'
OUTPUT_PATH_TRANSFORMATION_IMAGE = '/space/kurdenkova/rotate_change/' #'transformations_for_final_table/' #customPGD/'
OUTPUT_PATH_TRANSFORMATION_CSV = '/space/kurdenkova/rotate_change' #'transformations_for_final_table' #for_exp_sizing' #/article_data'
PREFIX_NAME = 'for_table'
OUTPUT_MODEL_NAME = 'model_t'

ALL_QUADRIC_DF_OUTPUT = 'all_quadric_rotate_change_with_adv'
PATCH_ALL_QUADRIC_DF_OUTPUT = 'patch_all_quadric_rotate_change_eith_adv'

RANDOM_STATE = 42

python processing_results.py \
    --patch-scale 10 \
    --random-shuffle True

# ВАЖНО: поменять адреса на свои папки, чтобы мои данные не перезатерлись.
# предполагается, что в файле constant.py есть следующий параметр с путями до состязательных картинок
#       ADVERSARIAL_IMAGES_MAPPING = {
#         10:
#          {
#            'original': DATA_MAPPING['test_images'],
#            'resnet': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/resnet_10_custom_adv_images.npy',
#            'cbn': '/dump/ekurdenkova/for_exp_with_random/images_without_transform/cbn_10_custom_adv_images.npy',
#           }
#        }
#
# также в этом файле необходимо указать
#       OUTPUT_PATH_TRANSFORMATION_IMAGE = '/dump/ekurdenkova/gitlab/No_fork/guardiann/guardiann/bagnet/transformations/for_exp_with_random/'
#                 (OUTPUT_PATH_TRANSFORMATION_IMAGE - адрес папки, в которой уже лежат трансформированные картинки)
#       OUTPUT_PATH_TRANSFORMATION_CSV = '/dump/ekurdenkova/for_exp_with_random'
#                 (OUTPUT_PATH_TRANSFORMATION_CSV - адрес папки, в которую будут положены промежуточные csv файлы с точностями)
#       ALL_QUADRIC_DF_OUTPUT = 'all_quadric_df_for_table_for_exp_with_random_10seed'
#                 (ALL_QUADRIC_DF_OUTPUT - название файла, в которое сохранится готовый csv для обычных моделей,
#                 обычно в папке запуска находится)
#       PATCH_ALL_QUADRIC_DF_OUTPUT = 'patch_all_quadric_df_for_for_exp_with_random_10seed'
#                 (PATCH_ALL_QUADRIC_DF_OUTPUT - аналогично ALL_QUADRIC_DF_OUTPUT, только для моделей с масками)
#
#       RANDOM_STATE = 42
#
# Данные из ALL_QUADRIC_DF_OUTPUT (вроде для random_seed=42, который настраивается в constants.py)
#Model	Transformation	original img	adv_img_resnet	adv_img_cbn
#0	resnet	clean image	0.80	0.38	0.80
#1	resnet	rotate	0.16	0.69	0.69
#2	resnet	dark	0.17	0.50	0.78
#3	resnet	gauss	0.17	0.38	0.80
#4	resnet	combo	0.16	0.68	0.72
#5	cbn	clean image	0.72	0.72	0.68
#6	cbn	rotate	0.10	0.70	0.68
#7	cbn	dark	0.12	0.73	0.74
#8	cbn	gauss	0.12	0.72	0.68
#9	cbn	combo	0.10	0.71	0.69
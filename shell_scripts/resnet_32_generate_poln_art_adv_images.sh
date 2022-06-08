python runner.py \
        --adversarial-images-arr-path article_data/images_without_transform/resnet_32_custom_adv_images.npy \
        --mode generation \
        --output-trans-image castr_art \
        --prefix-name resnet_32_poln_square_untarg \
        --model resnet \
        --load-from-constant True \
        --patch-scale 32 \
        --type-attack art \
        --target False \
        --patch-type square \
        --poln True \
        --filename resnet_32_poln_square_untarg

# filename поменять на свое название, куда сохранять.
#my_code_alone/bagnet/resnet_32_poln_square_untarg.npy
#for model resnet acc=0.7717690192483959
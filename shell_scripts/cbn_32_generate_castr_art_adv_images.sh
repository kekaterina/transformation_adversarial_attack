python runner.py \
        --adversarial-images-arr-path article_data/images_without_transform/cbn_32_custom_adv_images.npy \
        --mode generation \
        --output-trans-image castr_art \
        --prefix-name cbn_32_castr_square_untarg \
        --model cbn \
        --load-from-constant True \
        --patch-scale 32 \
        --type-attack art \
        --target False \
        --patch-type square \
        --poln False \
        --filename cbn_32_castr_square_untarg

# filename поменять на свое название, куда сохранять.
# my_code_alone/bagnet/cbn_32_castr_square_untarg.npy
# for model cbn acc=0.6614726550565231
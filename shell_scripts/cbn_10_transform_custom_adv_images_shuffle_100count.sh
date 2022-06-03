python runner.py \
    --adversarial-images-arr-path /dump/ekurdenkova/for_exp_with_random/images_without_transform/cbn_15_custom_adv_images.npy \
    --mode transformation \
    --output-trans-image for_exp_with_random \
    --prefix-name cbn_10 \
    --model cbn \
    --load-from-constant True \
    --filename /dump/ekurdenkova/for_exp_with_random/images_without_transform/cbn_10_custom_adv_images.npy \
    --patch-scale 10 \
    --count-images-from-first 100 \
    --max-iters 40 \
    --random-shuffle True

# filename поменять на свой адрес, куда сохранять картинки,
# название имеет вид {путь}/{model}_{patch_scale}_{type_attack}_adv_images.npy
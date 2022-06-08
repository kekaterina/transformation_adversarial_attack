python runner.py \
    --type-attack custom \
    --mode generation \
    --model resnet \
    --load-from-constant True \
    --filename /dump/ekurdenkova/for_exp_with_random/images_without_transform/resnet_10_custom_adv_images.npy \
    --patch-scale 10 \
    --count-images-from-first 100 \
    --max-iters 40 \
    --random-shuffle True

# filename поменять на свое название, куда сохранять. Patch_scale 10 - достаточно долго делается.
# accuracy = 0.38
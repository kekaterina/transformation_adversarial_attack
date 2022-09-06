python runner.py \
        --mode generation \
        --model cbn \
        --load-from-constant True \
        --patch-scale 32 \
        --type-attack art \
        --target False \
        --patch-type square \
        --poln False \
        --filename /space/kurdenkova/experiments_dump/art_full/cbn_32_castr_square_untarg

# filename поменять на свое название, куда сохранять.
# my_code_alone/bagnet/cbn_32_castr_square_untarg.npy
# for model cbn acc=0.6614726550565231
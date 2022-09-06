Предварительно запустить
```shell
$ sh setup.sh
```


Код надо запускать такой командой:

либы:
```shell
$ pip install \
    git+https://github.com/wielandbrendel/bag-of-local-features-models.git
    advertorch
```

запуск:
```shell
$ python3 main.py \
    --images-arr-path /dump/ekurdenkova/lab/images_data_dump_10_class_1000_pic.npy \
    --labels-arr-path /dump/ekurdenkova/lab/res_yys_10_class_1000_pic.npy \
    --attack-iters 40 \
    --sticker-size 100  \
    --train False \
    --num-classes 10 \
    --renumber False \
    --model-path /dump/ekurdenkova/lab/state_dict_best_custom_bagnet_sgd_001_9.pth \
    --model cbn \
    --targeted False \
    --attack spsa \
    --output-path output_exp_00
```

логи:
```shell
$ tensorboard --logdir ./output_exp_00 --port 5001 --host 0.0.0.0
```

Ключи расшифровываются следующим образом:

- **targeted** - (True/False) целевая или нецелевая атака
- **images-arr-path** - это путь до массива с картинками( в виде array размера (3, 224, 224) каждая картинка)
- **labels-arr-path** - это массив с лейблами картинок. Просто массив из верных номеров классов.
- **attack-iters** - число итераций атак
- **attack** - вид атаки: spsa/pgd
- **model-path** - это путь до сохраненной модели
- **model** - это тип модели, резнет или багнет: resnet, cbn
- **sticker-size** - это размер(длина стороны в пикселях) наклейки для атак. Она квадратная и помещается в разные места картинки словно по сетке в процессе атак. Если хоть одна наклейка подберется успешно - супер.
- **renumber** - ключ (True/False) перенумеровки классов из от 1 до 1000 до 1 до /количества_классов/, которое мы будем использовать.
- **train** - (True/False) надо ли обучать модель перед атаками
- **output-model-name** - если перед атаками обучаем модель, то это ключ для указания имени новой модели


Основной исполняемый файл - runner.py. Он может работать в нескольких режимах: генерация состязательных картинок и трансформация состязательных картинок.

Для запуска генерации состязательных картинок, нужно выполнить команду следующего вида:

` python runner.py --type-attack custom --mode generation --output-trans-image customPGD --prefix-name adv_cbn_32 --model adv_cbn --load-from-constant True --filename /dump/ekurdenkova/article_data/images_without_transform/adv_cbn_32_custom_adv_images.npy --patch-scale 32`

В результате выполнения этой команды по адресу, указанному в ключе --**filename** появится массив состязательных картинок.


Для запуска трасформации состязательных картинок, нужно выполнить команду вида:

` python runner.py --adversarial-images-arr-path /dump/ekurdenkova/article_data/images_without_transform/adv_cbn_32_custom_adv_images.npy --mode transformation --output-trans-image customPGD --prefix-name adv_cbn_32 --model adv_cbn --load-from-constant True`

В результате выполнения этой команды, в папке, указанной в ключе --**output-trans-image** появятся файлы с трансформациями картинок в виде numpy массивов.


Расшифровка параметров:

- **type-attack** - тип атаки. Значение _custom_ для кастомной PGD атаки и _art_ для атаки из Adversarial Robustness Toolbox.
- **mode** - режим программы: _generation_ для генерации и _transformation_ для трансформации.
- **prefix-name** - префикс названия файлов. Чтоб все хорошо работало, надо делать префикс по шаблону `<модель>_<размер наклейки>`. Модели обозначаются как 'cbn', 'resnet', 'adv_cbn'.
- **model** - модель из списка: ['cbn', 'resnet', 'adv_cbn']
- **load-from-constant** - логический параметр. _True_, если надо загружать модели из файла constant.py, _False_ в противном случае.
- **model_path** - адрес весов для модели, он будет использоваться при значении параметра **load-from-constant** = _False_.
- **patch-scale** - размер стороны наклейки в пикселях.
- **output-trans-image** - путь до места, куда надо сохранять картинки с трансформациями.
- **images-arr-path** - путь до массива картинок.
- **labels-arr-path** - путь до лейблов картинок.
- **adversarial-images-arr-path** - путь до массива с состязательными картинками.
- **filename** - путь до места, куда надо сохранить сгенерированные состязательные картинки. Для корректной работы название массива должно соответствовать шаблону `<модель>_<размер наклейки>_<тип атаки>_adv_images`, например `adv_cbn_32_custom_adv_images`.



Для получения датафреймов с результатами экспериментов, включая переносы атак, необходимо выполнить команду вида:

`python processing_results.py --patch-scale 32`

**patch-scale** - размер наклейки

Данный модуль использует важный параметр из файла constant.py MODEL_KEY_FOR_PROCESSING_RESULTS - список моделей, для которых необходимо получить результаты. Для этих моделей должны быть уже сгенерированы трансформационные картинки и указаны названия файлов согласно шаблонам. Допустимые модели для этого списка: 'resnet', 'cbn', 'adv_cbn'. После выполнения этой команды будут сохранены промежуточные csv файлы с результатами, а также готовые два датафрейма c названиями, указанными в файле constant.py в параметрах ALL_QUADRIC_DF_OUTPUT,
PATCH_ALL_QUADRIC_DF_OUTPUT.


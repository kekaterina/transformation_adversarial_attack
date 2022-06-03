#!/bin/sh
mkdir ./add_lib
cd ./add_lib

# надо форкнуть репозиторий, иначе не клонируется
git clone git@github.com:inspire-group/PatchGuard.git

# прокатит ли сделать инит сразу в репе или надо во внутренних папках тоже?
cd ./PatchGuard
echo '' > __init__.py

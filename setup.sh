#!/bin/sh
mkdir ./add_lib
cd ./add_lib

git clone git@github.com:inspire-group/PatchGuard.git

cd ./PatchGuard
echo '' > __init__.py

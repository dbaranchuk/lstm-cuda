#! /bin/bash

rm -rf build

baseDir=$(cd `dirname "$0"`;pwd)

# main
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/../src
nvcc  -std=c++11 -L/usr/local/cuda/lib64 -D_FORCE_INLINES -D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED -lcuda -lcudart -dc *.cu *.cpp

mkdir $baseDir/../build
tar cf - .|(cd $baseDir/../build; tar xf -)
cd $baseDir/../build
nvcc *.o -o lstm
echo "path:" $baseDir/../build/lstm
./lstm

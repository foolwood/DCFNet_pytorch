rm -rf build
g++ -c _resize.cpp -o resize.o -std=c++11 -fPIC -O3 -fopenmp
python setup.py build
cp build/lib.linux-x86_64-2.7/_sample.so .
rm -rf build
rm resize.o

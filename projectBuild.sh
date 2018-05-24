cd build
cmake CMakeLists.txt . -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so 
make  
./gpufusion

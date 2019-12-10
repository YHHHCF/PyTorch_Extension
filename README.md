I implemented a 2D high order convolution module which runs 3-5 times faster than PyTorch Python extension on CPU and over 100 times faster than PyTorch Python extension on GPU.  

I'm referring to [this](https://github.com/pytorch/extension-cpp/tree/master/cuda) git repo by [Peter Goldsborough](https://github.com/goldsborough).  


[python/](https://github.com/YHHHCF/PyTorch_Extension/tree/master/python) folder contains the original high order convolution model built on PyTorch extension (nn.Module)  

[cpp/](https://github.com/YHHHCF/PyTorch_Extension/tree/master/cpp) folder contains a python wrapper and uses c++ to implement the critical part of the high order convolution model  

[cuda/](https://github.com/YHHHCF/PyTorch_Extension/tree/master/cuda) folder contains a python wrapper and a c++ wrapper and uses CUDA to implement the critical part of the high order convolution model  

[tools/](https://github.com/YHHHCF/PyTorch_Extension/tree/master/tools) folder are reference data preprocessing tools and dataloaders for users who want to use real data to run the module  

All the operations you might want to know are as follows:  

1. To run Python extension  
python benchmark.py py cpu  
python benchmark.py py gpu  

2. To compile and run C++ extension:  
cd cpp/  
python setup.py install  
cd ..  
python benchmark.py cpp cpu  
python benchmark.py cpp gpu  

3. To compile and run CUDA extension:  
cd cuda/  
python setup.py install  
cd ..  
python benchmark.py cuda cpu  
python benchmark.py cuda gpu  

4. To run all of the experiments  
./run.sh  

5. To check the correctness of C++ and CUDA extension, we use Python extension as ground truth:  
python check.py  

# fast boolean semiring matrix multiplication for CFPQ
## Implementations:
* CPU:
    * matrix multiplication using m4ri lib
    * sparse boolean matrix multiplication using libs on Python
* GPU:
    * naive matrix multiplication with packed into uint32 boolean values
    * Four Russians method for matrix multiplication with packed into uint32 boolean values
## TODO list:
    * implement CFPQ on C++ (yarik)
    * implement CFPQ on Python
    * implement testing and benchmarking system
    * implement naive matrix multiplication on GPU (yarik)
    * implement Four Russians method for matrix multiplication on GPU (Nikita)
    * implement matrix multiplication using m4ri lib (Vladimir Kutuev)
    * implement sparse boolean matrix multiplication using scipy lib on Python
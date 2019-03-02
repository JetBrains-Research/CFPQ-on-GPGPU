# fast boolean semiring matrix multiplication for CFPQ
## Implementations:
* CPU:
    * matrix multiplication using m4ri lib
    * sparse boolean matrix multiplication using libs on Python
* GPU:
    * naive matrix multiplication with packed into uint32 boolean values
    * Four Russians method for matrix multiplication with packed into uint32 boolean values
## TODO list:
   - Implementation:
         - [x] CFPQ on C++ ([#4](https://github.com/SokolovYaroslav/fast-boolean-semiring-matrix-multiplication-for-CFPQ/issues/4))
      - [x] CFPQ on Python ([#9](https://github.com/SokolovYaroslav/fast-boolean-semiring-matrix-multiplication-for-CFPQ/issues/9))
      - [ ] testing and benchmarking system ([#7](https://github.com/SokolovYaroslav/fast-boolean-semiring-matrix-multiplication-for-CFPQ/issues/7))
      - [x] naive matrix multiplication on GPU ([#5](https://github.com/SokolovYaroslav/fast-boolean-semiring-matrix-multiplication-for-CFPQ/issues/5))
      - [x] Four Russians method for matrix multiplication on GPU ([#1](https://github.com/SokolovYaroslav/fast-boolean-semiring-matrix-multiplication-for-CFPQ/issues/1))
      - [ ] matrix multiplication using m4ri lib ()
      - [x] sparse boolean matrix multiplication using scipy lib on Python ()
      - [x] naive matrix multiplication on GPU using numba ([#11](https://github.com/SokolovYaroslav/fast-boolean-semiring-matrix-multiplication-for-CFPQ/issues/11))
   - Evaluation:
      - [ ] graphs
      - [ ] linear input (optopnal)
   - Paper ([TeX sources](https://github.com/YaccConstructor/articles/tree/master/InProgress/CFPQ_on_GPGPU_implementation_comparison))
      - [ ] Abstract, march 11
      - [ ] Full text, march 18

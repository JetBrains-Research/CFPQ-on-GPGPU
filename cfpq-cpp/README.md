# CFPQ C++ implementation

1. To implement cfpq using your own matrix multiplication algorithm you must inherit own matrix class from `Matrix` and call `Grammar::intersection_with_graph` parameterized by this class:
```cpp
grammar.intersection_with_graph<MyMatrixClass>(graph);
```
2. If your implementation requires preprocessing and post-processing of matrices and the environment, you must create your own environment class, inherited from `MatricesEnv` and call `Grammar::intersection_with_graph` parameterized by your matrix class and this class
```cpp
grammar.intersection_with_graph<MyMatrixClass, MyEnvClass>(graph);
```
* Method `MatricesEnv::environment_preprocessing` will be called before algorithm

* Method `MatricesEnv::environment_postprocessing` will be called after algorithm

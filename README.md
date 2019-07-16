# fast boolean semiring matrix multiplication for CFPQ

## Implementations

* CPU:
  * matrix multiplication using m4ri lib
  * sparse boolean matrix multiplication using libs on Python
* GPU:
  * naive matrix multiplication with packed into uint32 boolean values
  * Four Russians method for matrix multiplication with packed into uint32 boolean values

## Documentation

Prerequirements.
1. [Docker](https://docs.docker.com/) is installed and started.
2. [Git LFS](https://git-lfs.github.com/) is installed.
3. Some of the tests require `Cuda`, you cannot run these tests without a `Nvidia` video card. CPU-based implementations can be run anyway.

###Quick start

1. Clone this repo

2. Build docker image.
All tests should run inside docker, so first of all you should build image via `Dockerfile` in root folder.  
We use `ubuntu 18.04` with `CUDA` compability as main image and install `anaconda`, `mono`, `m4ri library`, [`GTgraph`](http://www.cse.psu.edu/~kxm85/software/GTgraph/) and some usefull utilites. Builded image has entrypoint for run testing, so generate tests description file and run docker with mounting project root folder.  

3. Run 'init.py' script to initialize environment.

4. Generate tests description.
For testing system we use `tests.csv` file which describes all tests, we need it because differents tests may use same files, you can't run testing **without** this file. For creating this file you can use `test_utils/build_testset.py` script, it has only parameter ‒ path to folder with data.  
For example, if your data stores in `data` folder, run this command for generate tests description:

```(bash)
python test_utils/build_testset.py data
```

5. Run tests  

Use the following command to run tests.
```(bash)
docker run -v /<path to project>:/work/ <image name>
```

After running tests, you can find `result.csv` file in root folder with time measure for each test and each solution in table format.  



### Data representation  
All tests are divided to groups and placed in different folders. Each folder must contain `Grammars` folder with describing all grammars and `Matrices` for describing all graphs of this group. Tests for each group is cross product of all grammars and graphs in it.  
**grammar** file is a file with rules definition in format: `nt nt1 nt2`, which means `nt1 -> nt1 nt2`, or `a T`, which means `a -> T`. We use capital letters for terminal symbols and other letters/words for non-terminal.  
**graph** file contain description of graph with lines likes `0 T 1`, where `0` and `1` is vertices numbers and `T` is a terminal symbol for this edge.  
So, if you want to add your data, just put it in data folder in described format and rebuild `tests.csv`.

## CEOs - A novel dimensionality reduction method for maximum inner product search

CEOs is a novel dimensionality reduction method that leverages the behavior of concomintants of extreme order statistics.
Different from the forklore random projection, CEOs uses a significantly larger number of random vectors `n_proj`.
The projection values on a few closest/furtherst vectors regarding the query are enough to estimate inner product between data points and the query.
We also propose a practical variant called coCEOs that has smaller indexing size and supports streaming indexing update.

We generate `n_repeats * n_proj` number of random vectors.
Each group of `n_proj` random vector is simulated by the [FFHT](https://github.com/FALCONN-LIB/FFHT) to speed up the projection time.
CEOs and coCEOs also support multi-threading for both indexing and querying by adding only ```#pragma omp parallel for```.

We call [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) that supports SIMD dot product computation.

## Prerequisites

* A compiler with C++17 support
* CMake >= 3.27 (test on Ubuntu 20.04 and Python3)
* Ninja >= 1.10 
* Eigen >= 3.3
* Boost >= 1.71
* Pybinding11 (https://pypi.org/project/pybind11/) 

## Installation

Just clone this repository and run

```bash
python3 setup.py install
```

or 

```bash
mkdir build && cd build && cmake .. && make
```


## Test call

Data and query must be d x n matrices.

```
import CEOs

# CEOs
index = CEOs.CEOs(n_points, n_features)
index.setIndexParam(n_proj, repeats, numThreads, -1)
index.build(dataset_t)  # size d x N

# query param
index.top_proj = 10
index.n_cand = 100
kNN, dist = index.search(query_t, k, True) # size d x Q

# coCEOs
index = CEOs.coCEOs(n_features)
index.setIndexParam(n_proj, repeats, numThreads, -1)
index.build(dataset_t)  # size d x N

# query param
index.top_proj = 40
index.n_cand = 100

kNN, dist = index.search(query_t, k, True) # size d x Q

index.add_remove(new_data, 1000) # remove the first 1000 points, and add new_data
kNN, dist = index.search(query_t, k, True) # size d x Q
```

See test/ceos.py for Python example and src/main.cpp for C++ example.

## Authors

It is developed by Ninh Pham. .
If you want to cite CEOs in a publication, please use

```
@inproceedings{DBLP:conf/kdd/Pham21,
author       = {Ninh Pham},
title        = {Simple Yet Efficient Algorithms for Maximum Inner Product Search via
Extreme Order Statistics},
booktitle    = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
pages        = {1339--1347},
publisher    = {{ACM}},
year         = {2021},
url          = {https://doi.org/10.1145/3447548.3467345},
doi          = {10.1145/3447548.3467345},
}
```






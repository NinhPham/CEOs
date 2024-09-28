## Falconn++ - A Locality-sensitive Filtering Approach for ANNS with Inner Product

Falconn++ is a locality-sensitive filtering (LSF) approach, built on top of cross-polytope LSH ([FalconnLib](https://github.com/FALCONN-LIB/FALCONN)) to answer approximate nearest neighbor search with inner product. 
The filtering mechanism of Falconn++  bases on the asymptotic property of the concomitant of extreme order statistics where the projections of $x$ onto closest or furthest vector to $q$ preserves the dot product $x^T q$.
Similar to FalconnLib, Falconn++ utilizes many random projection vectors and uses the [FFHT](https://github.com/FALCONN-LIB/FFHT) to speed up the hashing evaluation.
Apart from many hashing-based approaches, Falconn++ has multi-probes on both indexing and querying to improve the quality of candidates.
Falconn++ also supports multi-threading for both indexing and querying by adding only ```#pragma omp parallel for```.

We call [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) that supports SIMD dot product computation.
We have not engineered Falconn++ much with other techniques, e.g. prefetching.

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
index = CEOs.CEOs(n_points, n_features)
index.setIndexParam(n_proj, repeats, numThreads, -1)
index.build(dataset_t)  # size d x N
kNN, dist = index.search(query_t, k, True) # size d x Q

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

> [CEOs](https://dl.acm.org/doi/10.1145/3447548.3467345)
> Ninh Pham
> KDD 2021




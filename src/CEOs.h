//
// Created by npha145 on 15/03/24.
//

/**
 * CEOs class
 * Mainly used for estimating inner product
 * There are two algorithms: CEOs and coCEOs
 * - CEOs aggregates n projections on top-s closest random vector to estimate inner product for n points
 * - coCEOs reduces the whole projection matrix by m x (n_proj * n_repeats) where m is the number of top-points closest/furthest to the random vector
 */

#ifndef CEOS_H
#define CEOS_H

#include "Header.h"

class CEOs{

protected:

    int n_points;
    int n_features;

    int n_proj = 256;
    int n_rotate = 3;
    int top_m = 100; // query might use a subset of closest/furthest points to the vector
    
    int n_repeats = 1;
    int seed = -1;

    RowMajorMatrixXf matrix_X; // n x d

    // Use for both CEOs-Est and coCEOs
    // - CEOs Est: n x (n_proj * repeat) where the first (n_proj x n) is for the first set of random rotation
    // - coCEOs: (4 * top-points) x (n_proj * repeat)
    // the first/second is index/projection value of close points, the third/forth is index/projection value of far points
    MatrixXf matrix_P;

    // coCEOs-hash: (2 * indexBucketSize) x (n_proj * repeat)
    MatrixXi matrix_H;

    int fhtDim;

    boost::dynamic_bitset<> bitHD;

public:

    int n_threads = -1;

    // Query param
    int n_probed_vectors = 10;
    int n_probed_points = 10; // query might use a subset of closest/furthest points to the vector
    int n_cand = 10;
    bool centering = false;

    // function to initialize private variables
    CEOs(int n, int d){
        n_points = n;
        n_features = d;
    }

    void set_CEOsParam(int numProj, int numRepeats, int m, int t, int s) {

        n_proj = numProj;
        n_repeats = numRepeats;

        top_m = m;
        n_probed_points = top_m;

        set_threads(t);
        seed = s;

        // setting fht dimension. Note n_proj must be 2^a, and > n_features
        // Ensure fhtDim > n_proj
        if (n_proj < n_features)
            fhtDim = 1 << int(ceil(log2(n_features)));
        else
            fhtDim = 1 << int(ceil(log2(n_proj)));

    }

    void clear() {
        matrix_X.resize(0, 0);
        matrix_P.resize(0, 0);
        matrix_H.resize(0, 0);

        bitHD.clear();
    }

    // Will use for setting candidate for re-rank
//    void set_qProbes(int p){
//        qProbes = p;
//    }

    void set_threads(int t)
    {
        if (t <= 0)
            n_threads = omp_get_max_threads();
        else
            n_threads = t;
    }

    void build_CEOs(const Ref<const RowMajorMatrixXf> &);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> search_CEOs(const Ref<const RowMajorMatrixXf> &, int, bool=false);

    void build_coCEOs_Est(const Ref<const RowMajorMatrixXf> &);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> search_coCEOs_Est(const Ref<const RowMajorMatrixXf> &, int, bool=false);

    void build_CEOs_Hash(const Ref<const RowMajorMatrixXf> &);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> search_CEOs_Hash(const Ref<const RowMajorMatrixXf> &, int, bool=false);

    ~CEOs() { clear(); }
};

#endif //CEOS_H

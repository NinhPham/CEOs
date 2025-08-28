//
// Created by npha145 on 22/09/24.
//

#ifndef STREAMCEOS_H
#define STREAMCEOS_H


#include "Header.h"
#include <tuple>

class streamCEOs{

protected:

    int n_points;
    int n_features;

    int n_proj = 256;
    int n_rotate = 3;
    int top_m = 100; // we make it public as query might retrieve a subset of points in the bucket

    int n_repeats = 1;
    int seed = -1;

    deque<VectorXf> deque_X; // It is fast for remove and add at the end of queue
    VectorXf vec_center; // When initialize the data structure with large n_points, we can apply centering trick

    // coCEOs-Est and CEOs-Hash: (4 * top-points) x (n_proj * repeat)
    // the first/second is index/projection value of close points, the third/forth is index/projection value of far points
    MatrixXf matrix_P; // (n_proj * repeat) x n where the first D x n is for the first set of random rotation

    int fhtDim;
    boost::dynamic_bitset<> bitHD;

public:

    int n_threads = -1;

    // Query param
    int n_probed_vectors = 10;
    int n_probed_points = 10;
    int n_cand = 10;
    bool centering = false;

    // we need n_features to design fhtDim
    streamCEOs(int d){
//        n_points = n; // we do not need n_points as we support add_remove
        n_features = d;
        vec_center = VectorXf::Zero(d);
    }

    void set_streamCEOsParam(int numProj, int repeats, int m, int threads, int s) {

        n_proj = numProj;
        n_repeats = repeats;
        top_m = m;
        n_probed_vectors = top_m;

        set_threads(threads);
        seed = s;

        // setting fht dimension. Note n_proj must be 2^a, and > n_features
        // Ensure fhtDim > n_proj
        if (n_proj < n_features)
            fhtDim = 1 << int(ceil(log2(n_features)));
        else
            fhtDim = 1 << int(ceil(log2(n_proj)));

    }

    void clear() {

        deque_X.clear();

        matrix_P.resize(0, 0);
        vec_center.resize(0);

        bitHD.clear();
    }

    void set_threads(int t)
    {
        if (t <= 0)
            n_threads = omp_get_max_threads();
        else
            n_threads = t;
    }

    void build(const Ref<const RowMajorMatrixXf> &);
    void update(const Ref<const RowMajorMatrixXf> &, int = 0);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> estimate_search(const Ref<const RowMajorMatrixXf> &, int, bool=false);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> hash_search(const Ref<const RowMajorMatrixXf> &, int, bool=false);

    ~streamCEOs() { clear(); }

// TODO: Support other distances
//  Add support L2 via transformation: x --> {2x, -|x|^2}, q --> {q, 1}
//  If called via Python on million points, call this transformation externally
//  If called via loading file on billion points, then it must an internal transformation

// TODO: Support billion points
//  Add sketch to estimate distance
//  coCEOs might be useful to estimate distance if increasing top-r. Then we need small n_cand.
//  Since n_cand is small, then disk-based (SSD) index should work very well on coCEOs

};


#endif //STREAMCEOS_H

//
// Created by npha145 on 22/09/24.
//

#ifndef COCEOS_H
#define COCEOS_H


#include "Header.h"
#include <tuple>

class coCEOs{

protected:

    int n_points = 0;
    int n_features;

    int n_proj = 256;
    int n_rotate = 3;
    int iTopPoints = 100; // we make it public as query might retrieve a subset of points in the bucket

    bool centering = true;

    int n_repeats = 1;
    int seed = -1;

    deque<VectorXf> deque_X; // It is fast for remove and add at the end of queue
    VectorXf vec_centerX; // When initialize the data structure with large n_points, we can apply centering trick

    // coCEOs: estimating
    // (n_proj * repeat) x  (2 * indexBucketSize)
    MatrixXf matrix_P; // (n_proj * repeat) x n where the first D x n is for the first set of random rotation

    int fhtDim;

    // Each vector corresponds to each rotation
    // Each boost::bitset corresponds to each tensor call
    vector<boost::dynamic_bitset<>> vecHD1;
    vector<boost::dynamic_bitset<>> vecHD2;
    vector<boost::dynamic_bitset<>> vecHD3;


protected:

    /**
     * Generate 3 vectors of random sign, each for one random rotation
     * One vector contain n_repeats boost::bitset, each corresponds to each tensor call D^{n_exp}
     *
     * @param p_numBits = fhtDim
     * @param p_numRepeats = n_exp
     */
    void bitGenerator(int p_numBits, int p_numRepeats)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        if (coCEOs::seed > -1) // then use the assigned seed
            seed = coCEOs::seed;

        default_random_engine generator(seed);
        uniform_int_distribution<uint32_t> unifDist(0, 1);

        vecHD1 = vector<boost::dynamic_bitset<>>(p_numRepeats);
        vecHD2 = vector<boost::dynamic_bitset<>>(p_numRepeats);
        vecHD3 = vector<boost::dynamic_bitset<>>(p_numRepeats);

        for (int e = 0; e < p_numRepeats; ++e)
        {
            vecHD1[e] = boost::dynamic_bitset<> (p_numBits);
            vecHD2[e] = boost::dynamic_bitset<> (p_numBits);
            vecHD3[e] = boost::dynamic_bitset<> (p_numBits);

            for (int d = 0; d < p_numBits; ++d)
            {
                vecHD1[e][d] = unifDist(generator) & 1;
                vecHD2[e][d] = unifDist(generator) & 1;
                vecHD3[e][d] = unifDist(generator) & 1;
            }
        }

        // Print to test
//        for (int e = 0; e < p_numRepeats; ++e)
//        {
//            for (int d = 0; d < p_numBits; ++d)
//            {
//                cout << vecHD1[e][d] << endl;
//                cout << vecHD2[e][d] << endl;
//                cout << vecHD3[e][d] << endl;
//            }
//        }
    }

public:

    int n_threads = -1;


    // Query param
    int n_probedVectors = 10;
    int n_probedPoints = 10;
    int n_cand = 10;


    // we need n_features to design fhtDim
    coCEOs(int d){
//        n_points = n; // we do not need n_points as we support add_remove
        n_features = d;
        vec_centerX = VectorXf::Zero(d);
    }

    void setIndexParam(int numProj, int repeats, int top_m, int threads, int s, bool center) {

        n_proj = numProj;
        n_repeats = repeats;
        iTopPoints = top_m;
        n_probedVectors = top_m;

        set_threads(threads);
        seed = s;
        centering = center;

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
        vec_centerX.resize(0);

        vecHD1.clear();
        vecHD2.clear();
        vecHD3.clear();


    }

    void set_threads(int t)
    {
        if (t <= 0)
            n_threads = omp_get_max_threads();
        else
            n_threads = t;
    }

    void build(const Ref<const Eigen::MatrixXf> &);
    void update(const Ref<const Eigen::MatrixXf> &, int = 0);
    tuple<MatrixXi, MatrixXf> estimate_search(const Ref<const MatrixXf> &, int, bool=false);
    tuple<MatrixXi, MatrixXf> hash_search(const Ref<const MatrixXf> &, int, bool=false);

    ~coCEOs() { clear(); }

// TODO: Support other distances
//  Add support L2 via transformation: x --> {2x, -|x|^2}, q --> {q, 1}
//  If called via Python on million points, call this transformation externally
//  If called via loading file on billion points, then it must an internal transformation

// TODO: add read binary/hdfs file for fast reading IO

// TODO: Support billion points
//  Add sketch to estimate distance
//  coCEOs might be useful to estimate distance if increasing top-r. Then we need small n_cand.
//  Since n_cand is small, then disk-based (SSD) index should work very well on coCEOs

};


#endif //COCEOS_H

//
// Created by npha145 on 15/03/24.
//

#ifndef CEOS_H
#define CEOS_H

#include "Header.h"

class CEOs{

protected:

    int n_points;
    int n_features;

    int n_proj = 256;
    int n_rotate = 3;
    int iTopPoints = 100; // query might use a subset of closest/furthest points to the vector
    
    int n_repeats = 1;
    int seed = -1;

    MatrixXf matrix_X; // d x n

    // Use for both CEOs-Est and coCEos
    // - CEOs Est: n x (n_proj * repeat) where the first (n_proj x n) is for the first set of random rotation
    // - coCEOs: (4 * top-points) x (n_proj * repeat)
    // the first/second is index/projection value of close points, the third/forth is index/projection value of far points
    MatrixXf matrix_P;

    // coCEOs-hash: (2 * indexBucketSize) x (n_proj * repeat)
    MatrixXi matrix_H;

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
    inline void bitGenerator(int p_numBits, int p_numRepeats)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        if (CEOs::seed > -1) // then use the assigned seed
            seed = CEOs::seed;

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
    int n_probedPoints = 10; // query might use a subset of closest/furthest points to the vector
    int n_cand = 10;


    // function to initialize private variables
    CEOs(int n, int d){
        n_points = n;
        n_features = d;
    }

    void set_CEOsParam(int D, int repeats, int t, int s) {

        n_proj = D;
        n_repeats = repeats;
        set_threads(t);
        seed = s;

        // setting fht dimension. Note n_proj must be 2^a, and > n_features
        // Ensure fhtDim > n_proj
        if (n_proj < n_features)
            fhtDim = 1 << int(ceil(log2(n_features)));
        else
            fhtDim = 1 << int(ceil(log2(n_proj)));

    }

    void set_coCEOsParam(int numProj, int numRepeats, int top_m, int t, int s) {

        n_proj = numProj;
        n_repeats = numRepeats;

        iTopPoints = top_m;
        n_probedPoints = top_m;

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

        vecHD1.clear();
        vecHD2.clear();
        vecHD3.clear();
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

    void build_CEOs(const Ref<const Eigen::MatrixXf> &);
    tuple<MatrixXi, MatrixXf> search_CEOs(const Ref<const MatrixXf> &, int, bool=false);

    void build_coCEOs_Est(const Ref<const Eigen::MatrixXf> &);
    tuple<MatrixXi, MatrixXf> search_coCEOs_Est(const Ref<const MatrixXf> &, int, bool=false);

    void build_coCEOs_Hash(const Ref<const Eigen::MatrixXf> &);
    tuple<MatrixXi, MatrixXf> search_coCEOs_Hash(const Ref<const MatrixXf> &, int, bool=false);

// TODO: add support L2 via transformation (similar to sDbscan)
// TODO: add read binary/hdfs file for fast reading IO


    ~CEOs() { clear(); }
};

#endif //CEOS_H

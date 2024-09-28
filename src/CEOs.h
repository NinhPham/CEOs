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

    int n_repeats = 1;
    int seed = -1;

    MatrixXf matrix_X; // d x n

    // CEOs Est: (n_proj * repeat) x n where the first D x n is for the first set of random rotation
    // coCEOs: (n_proj * repeat) x  (2 * top_points)
    MatrixXf matrix_P; // (n_proj * repeat) x n where the first D x n is for the first set of random rotation

    // vector of minQueue for coCEOs
    vector< priority_queue< IFPair, vector<IFPair>, greater<> > >  vec_minQueClose, vec_minQueFar;

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
     * @param p_nNumBit = fhtDim
     * @param p_nExponent = n_exp
     */
    void bitGenerator(int p_nNumBit, int p_nExponent)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        if (CEOs::seed > -1) // then use the assigned seed
            seed = CEOs::seed;

        default_random_engine generator(seed);
        uniform_int_distribution<uint32_t> unifDist(0, 1);

        vecHD1 = vector<boost::dynamic_bitset<>>(p_nExponent);
        vecHD2 = vector<boost::dynamic_bitset<>>(p_nExponent);
        vecHD3 = vector<boost::dynamic_bitset<>>(p_nExponent);

        for (int e = 0; e < p_nExponent; ++e)
        {
            vecHD1[e] = boost::dynamic_bitset<> (p_nNumBit);
            vecHD2[e] = boost::dynamic_bitset<> (p_nNumBit);
            vecHD3[e] = boost::dynamic_bitset<> (p_nNumBit);

            for (int d = 0; d < p_nNumBit; ++d)
            {
                vecHD1[e][d] = unifDist(generator) & 1;
                vecHD2[e][d] = unifDist(generator) & 1;
                vecHD3[e][d] = unifDist(generator) & 1;
            }
        }

        // Print to test
//        for (int e = 0; e < p_nExponent; ++e)
//        {
//            for (int d = 0; d < p_nNumBit; ++d)
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
    int top_proj = 10;
    int top_points = 10;
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

    void set_coCEOsParam(int D, int repeats, int m, int t, int s) {

        n_proj = D;
        n_repeats = repeats;
        top_points = m;
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


    void build_coCEOs(const Ref<const Eigen::MatrixXf> &);
    void add_coCEOs(const Ref<const Eigen::MatrixXf> &);
    tuple<MatrixXi, MatrixXf> search_coCEOs(const Ref<const MatrixXf> &, int, bool=false);

// TODO: add support L2 via transformation (similar to sDbscan)

// TODO: add read binary/hdfs file for fast reading IO


    ~CEOs() { clear(); }
};

#endif //CEOS_H

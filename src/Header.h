#pragma once

#include "fht.h"

#include <Eigen/Dense>

//#include <unordered_map>
//#include <unordered_set>
#include "tsl/robin_map.h" // https://github.com/Tessil/robin-map
#include "tsl/robin_set.h" // https://github.com/Tessil/robin-map


#include <vector>
#include <queue>
#include <list>
#include <algorithm>
#include <random>

#include <chrono>
#include <iostream> // cin, cout

//#include <boost/multi_array.hpp>
#include <boost/dynamic_bitset.hpp> // use in vector HD

using namespace Eigen;
using namespace std;

typedef vector<float> FVector;
typedef vector<int> IVector;

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//typedef vector<uint32_t> I32Vector;
//typedef vector<uint64_t> I64Vector;

//#define CACHE_LINE  32
//#define CACHE_ALIGN __declspec(align(CACHE_LINE))

//typedef boost::multi_array<int, 3> IVector3D;

//struct myComp {
//
//    constexpr bool operator()(
//        pair<double, int> const& a,
//        pair<double, int> const& b)
//        const noexcept
//    {
//        return a.first > b.first;
//    }
//};

struct IFPair
{
    int m_iIndex;
    float	m_fValue;

    IFPair()
    {
        m_iIndex = 0;
        m_fValue = 0.0;
    }

    IFPair(int p_iIndex, double p_fValue)
    {
        m_iIndex = p_iIndex;
        m_fValue = p_fValue;
    }

    // Overwrite operation < to get top K largest entries
    bool operator<(const IFPair& p) const
    {
        return m_fValue < p.m_fValue;
    }

    bool operator>(const IFPair& p) const
    {
        return m_fValue > p.m_fValue;
    }
};


struct IndexParam
{
    int n_points;
    int n_features;
    int n_proj;
    int top_m;
    int n_repeats;
    int n_threads;
    int seed;
    bool centering;
};

struct QueryParam{

    int n_queries;
    int n_neighbors;
    int n_probed_vectors;
    int n_probed_points;
    int n_cand;
    bool verbose;
};

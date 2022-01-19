#ifndef HEADER_H_INCLUDED
#define HEADER_H_INCLUDED

#pragma once
#include "fht.h"

#include <cstdlib> // for abs, atoi
#include <chrono> // for wall clock time
#include <iostream> // cin, cout
#include <cstdio> // printf()

//#include <math.h> // ceil()
//#include <cmath> // cos(), pow()
#include <algorithm>


#include <vector>
#include <utility>
#include <unordered_map>
#include <queue>
#include <unordered_set>

#include <string>
#include <time.h> // for time(0) to generate different random number
#include <random>

using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include <boost/dynamic_bitset.hpp>

//#include <tsl/robin_set.h>
//using namespace tsl;

//#include <sparsehash/dense_hash_set>
//using namespace google;

//#include <sparsepp/spp.h>
//using spp::sparse_hash_set;

#define PI				3.141592653589793238460
#define PRIME           2147483647 // 2^19 - 1 or 2^31 - 1 = 2147483647
#define EPSILON         0.000001

typedef vector<int> IVector;
typedef vector<float> FVector;

/*
Structure of pair
- point index
- dot product
*/
struct IFPair
{
    int m_iIndex;
    double	m_fValue;

	IFPair()
	{
        m_iIndex	= 0;
        m_fValue	= 0.0;
	}

	IFPair(int p_iIndex, float p_fValue)
	{
		m_iIndex	= p_iIndex;
		m_fValue	= p_fValue;
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

/*
Structure of pair
- point index
- counter
*/

struct IIPair
{
    int m_iIndex;
    int	m_iValue;

	IIPair()
	{
	    m_iIndex	= 0;
	    m_iValue	= 0;
	}

	IIPair(int p_iIndex, int p_iCounter)
	{
	    m_iIndex	= p_iIndex;
	    m_iValue	= p_iCounter;
	}

	// Overwrite operation < to get top K largest entries
    bool operator<(const IIPair& p) const
    {
        return m_iValue < p.m_iValue;
    }

    bool operator>(const IIPair& p) const
    {
        return m_iValue > p.m_iValue;
    }
};

/*
- PARAM OF ALGORITHMS
*/

// Input
extern int PARAM_DATA_N; // Number of points (rows) of X
extern int PARAM_QUERY_Q; // Number of rows (queries) of Q
extern int PARAM_DATA_D; // Number of dimensions

// Internal params
extern bool PARAM_INTERNAL_SAVE_OUTPUT; // save the results

// MIPS
extern int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
extern int PARAM_MIPS_TOP_B; // number of points to compute dot products
extern int PARAM_MIPS_NUM_SAMPLES; // Budgeted number of points to compute dot products

// dWedge & Greedy
extern int PARAM_INTERNAL_dWEDGE_N; // only keep top-dWEDGE_N largest value from each dimension

// LSH table & code
extern int PARAM_LSH_NUM_PARTITIONS; // number of partitions in normRange LSH
extern int PARAM_LSH_NUM_HASH; // Number of concanation hash functions
extern int PARAM_LSH_NUM_TABLES; // Number of hash tables

// CEOs
extern int PARAM_CEOs_D_UP;
extern int PARAM_CEOs_S0;
extern int PARAM_CEOs_N_DOWN;
extern int PARAM_CEOs_NUM_ROTATIONS; // Use Gauss or HD3
extern int PARAM_INTERNAL_LOG2_CEOs_D_UP;
extern bool PARAM_INTERNAL_NOT_STORE_MATRIX_X; // used on CEOs-TA if we do not have space

// Conventional random projection
extern int PARAM_RP_D_DOWN;

/*
- DATA STRUCTURE
*/

// Input
//extern Matrix<float, Dynamic, Dynamic, ColMajor> MATRIX_X; // matrix X
extern MatrixXf MATRIX_X;
//extern Matrix<float, Dynamic, Dynamic, ColMajor> MATRIX_Q; // matrix Q
extern MatrixXf MATRIX_Q;

// Sampling
extern VectorXf WEDGE_COL_NORM_1; // matrix S for positive case
extern vector<IFPair> WEDGE_SORTED_COL; // Sampling & Greedy

// LSH
extern vector<unordered_map<uint64_t, IVector>> VECTOR_LSH_TABLES;
extern vector<boost::dynamic_bitset<uint64_t>> VECTOR_LSH_CODES;
extern VectorXf VECTOR_LSH_PARTITION_NORM;

// CEOs
extern MatrixXi MATRIX_1CEOs;
extern MatrixXi MATRIX_2CEOs;

extern MatrixXi CEOs_TA_SORTED_IDX;
extern vector<IFPair> coCEOs_MAX_IDX; // co-CEOs
extern vector<IFPair> coCEOs_MIN_IDX; // co-CEOs

extern MatrixXf PROJECTED_X;
extern MatrixXi HD3;
extern MatrixXf MATRIX_G;

/**
Struct of comparing matrix elements
**/
struct compare_PROJECTED_X
{
    int dim;
    compare_PROJECTED_X(int d) : dim(d) { }
    bool operator() (int i, int j) { return (PROJECTED_X(i, dim) > PROJECTED_X(j, dim)); }
};

#endif // HEADER_H_INCLUDED

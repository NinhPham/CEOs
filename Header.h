#ifndef HEADER_H_INCLUDED
#define HEADER_H_INCLUDED

#include <time.h> // clock(), time(NULL)
#include <chrono> // for measuring time

#include <cstdio> // printf()
//#include <math.h> // ceil()
#include <cmath> // cos()
#include <algorithm>

#include <vector>
#include <utility>
#include <unordered_map>
#include <string>

using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

//#include <tsl/robin_set.h>
//using namespace tsl;

//#include <sparsehash/dense_hash_set>
//using namespace google;

//#include <sparsepp/spp.h>
//using spp::sparse_hash_set;

#define PI				3.141592653589793238460
#define PRIME           2147483647 // 2^19 - 1 or 2^31 - 1 = 2147483647
#define EPSILON         0.000001

typedef vector<double> DVector;
typedef vector<int> IVector;
typedef vector<uint32_t> I32Vector;
typedef vector<uint64_t> I64Vector;

/*
Structure of pair
- point index
- dot product
*/
struct IDPair
{
    int m_iIndex;
    double	m_dValue;

	IDPair()
	{
        m_iIndex	= 0;
        m_dValue	= 0.0;
	}

	IDPair(int p_iIndex, double p_dValue)
	{
		m_iIndex	= p_iIndex;
		m_dValue	    = p_dValue;
	}

	// Overwrite operation < to get top K largest entries
    bool operator<(const IDPair& p) const
    {
        return m_dValue < p.m_dValue;
    }

    bool operator>(const IDPair& p) const
    {
        return m_dValue > p.m_dValue;
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
Structure of triple
- point index
- Xij
- Xij/alpha (sort criteria)
*/
struct IDDTriple
{
    int     m_iIndex;
    double	m_dValue;
    double	m_dSort;

	IDDTriple()
	{
        m_iIndex	= 0;
        m_dValue	= 0.0;
        m_dSort     = 0.0;
	}

	IDDTriple(int p_iIndex, double p_dValue, double p_dSort)
	{
		m_iIndex	= p_iIndex;
		m_dValue	= p_dValue;
		m_dSort	    = p_dSort;
	}

	// Overwrite operation < to get top K largest entries
    bool operator<(const IDDTriple& p) const
    {
        return m_dSort < p.m_dSort;
    }

    bool operator>(const IDDTriple& p) const
    {
        return m_dSort > p.m_dSort;
    }
};


extern int PARAM_DATA_N; // Number of points (rows) of X
extern int PARAM_QUERY_Q; // Number of rows (queries) of Q
extern int PARAM_DATA_D; // Number of dimensions

extern int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
extern int PARAM_MIPS_SAMPLES; // Budgeted number of points to compute dot products
extern int PARAM_MIPS_DOT_PRODUCTS; // number of points to compute dot products
extern int PARAM_MIPS_PRESAMPLES_MAX;

extern int PARAM_LSH_PARTITIONS; // number of partitions in normRange LSH
extern int PARAM_LSH_HASH_FUNCTIONS; // Number of concanation hash functions
extern int PARAM_LSH_HASH_TABLES; // Number of hash tables
extern int PARAM_MIPS_DOT_PRODUCTS; // Number of dot products

//extern Matrix<double, Dynamic, Dynamic, RowMajor> MATRIX_X; // matrix X
extern MatrixXd MATRIX_X;
//extern Matrix<double, Dynamic, Dynamic, ColMajor> MATRIX_Q; // matrix Q
extern MatrixXd MATRIX_Q;

extern IVector POS_PRESAMPLES; // matrix S for positive case
extern IVector NEG_PRESAMPLES; // matrix S for negative case

extern DVector POS_COL_NORM_1; // matrix S for positive case
extern DVector NEG_COL_NORM_1; // matrix S for negative case

extern VectorXd VECTOR_COL_MIN;
extern VectorXd VECTOR_COL_MAX;

// extern MatrixXi GREEDY_STRUCT;

extern vector<IDPair> COL_SORT_DATA_IDPAIR; // for PPS and priority sampling
extern vector<IDPair> SORTED_DATA; // for heuristic wedge
extern vector<IDDTriple> COL_SORT_DATA_IDDTRIPLE; // for distributed priority sampling

extern MatrixXd MATRIX_LSH_SIM_HASH; // matrix of L * D * K normal random variables
extern I32Vector VECTOR_LSH_UNIVERSAL_HASH; // vector of L * K random integers
extern vector<unordered_map<uint64_t, IVector>> VECTOR_LSH_TABLES;
extern I64Vector VECTOR_LSH_CODES;
extern DVector VECTOR_LSH_PARTITION_NORM;
extern int PARAM_LSH_NUM_DECIMAL;;

extern bool PARAM_TEST_SAVE_OUTPUT; // save the results
extern int PARAM_TEST_SAMPLING_RANDOM_GENERATOR;

/**
Struct of comparing matrix elements
**/
struct compareMatrixStruct
{
    int dim;
    compareMatrixStruct(int d) : dim(d) { }
    bool operator() (int i, int j) { return (MATRIX_X(i, dim) > MATRIX_X(j, dim)); }
};

#endif // HEADER_H_INCLUDED

#include "Header.h"

/**
We need to declare all global variable here so that we do not declare anywhere else.
Compiler will find their declaration here
**/


int PARAM_DATA_N; // Number of points (rows) of X
int PARAM_QUERY_Q; // Number of rows (queries) of Q
int PARAM_DATA_D; // Number of dimensions

int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
int PARAM_MIPS_SAMPLES; // Budgeted number of points to compute dot products
int PARAM_MIPS_DOT_PRODUCTS; // number of points to compute dot products
int PARAM_MIPS_PRESAMPLES_MAX; // the maximum  number of samples for each dimension

int PARAM_LSH_PARTITIONS; // number of partitions in normRangeLSH
int PARAM_LSH_HASH_FUNCTIONS; // Number of concanation hash functions
int PARAM_LSH_HASH_TABLES; // Number of hash tables

MatrixXd MATRIX_X;

//Matrix<double, Dynamic, Dynamic, RowMajor> MATRIX_X; // matrix X
Matrix<double, Dynamic, Dynamic, ColMajor> MATRIX_Q; // matrix Q

//VectorXd VECTOR_C_POS_SUM; // sum of each column of matrix X for positive query scalar
//VectorXd VECTOR_C_NEG_SUM; // sum of each column of matrix X for negative query scalar
//
//MatrixXi MATRIX_X_POS_PRESAMPLES; // matrix S for positive case
//MatrixXi MATRIX_X_NEG_PRESAMPLES; // matrix S for negative case

IVector POS_PRESAMPLES; // vector of D * MAX_PRESAMPLES for positive case
IVector NEG_PRESAMPLES; // vector of D * MAX_PRESAMPLES for negative case

DVector POS_COL_NORM_1; // vector of D for positive case
DVector NEG_COL_NORM_1; // vector of D for negative case

VectorXd VECTOR_COL_MIN; // vector of D for minimum value of each dimension
VectorXd VECTOR_COL_MAX; // vector of D for maximum value of each dimension

// MatrixXi GREEDY_STRUCT; // data structure for greedy

vector<IDPair> COL_SORT_DATA_IDPAIR; // PPS and Prirority sampling
vector<IDPair> SORTED_DATA; // Wedge heuristics
vector<IDDTriple> COL_SORT_DATA_IDDTRIPLE; // for distributed priority sampling

MatrixXd MATRIX_LSH_SIM_HASH; // vector of L * D * K normal random variables
I32Vector VECTOR_LSH_UNIVERSAL_HASH; // vector of L * K random integers
vector<unordered_map<uint64_t, IVector>> VECTOR_LSH_TABLES;

I64Vector VECTOR_LSH_CODES;
int PARAM_LSH_NUM_DECIMAL;

DVector VECTOR_LSH_PARTITION_NORM;

bool PARAM_TEST_SAVE_OUTPUT;
int PARAM_TEST_SAMPLING_RANDOM_GENERATOR;


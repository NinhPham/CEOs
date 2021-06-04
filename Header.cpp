#include "Header.h"

/**
We need to declare all global variable here so that we do not declare anywhere else.
Compiler will find their declaration here
**/

/*
- PARAM OF ALGORITHMS
*/

// Input
int PARAM_DATA_N; // Number of points (rows) of X
int PARAM_QUERY_Q; // Number of rows (queries) of Q
int PARAM_DATA_D; // Number of dimensions

// Internal params
bool PARAM_INTERNAL_SAVE_OUTPUT; // save the results

// MIPS
int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
int PARAM_MIPS_TOP_B; // number of points to compute dot products

// Sampling
int PARAM_MIPS_SAMPLES; // Budgeted number of points to compute dot products

// LSH
int PARAM_LSH_PARTITIONS; // number of partitions in normRange LSH
int PARAM_LSH_HASH_FUNCTIONS; // Number of concanation hash functions
int PARAM_LSH_HASH_TABLES; // Number of hash tables
int PARAM_INTERNAL_LSH_NUM_DECIMAL;

// CEOs
int PARAM_CEOs_D_UP;
int PARAM_CEOs_S0;
int PARAM_CEOs_N_DOWN;
bool PARAM_INTERNAL_GAUSS_HD3; // Use Gauss or HD3
bool PARAM_INTERNAL_NOT_STORE_MATRIX_X; // data structure does not fit into RAM

/*
- DATA STRUCTURE
*/

// Input
MatrixXd MATRIX_X;
//Matrix<double, Dynamic, Dynamic, RowMajor> MATRIX_X; // matrix X
Matrix<double, Dynamic, Dynamic, ColMajor> MATRIX_Q; // matrix Q

DVector POS_COL_NORM_1; // matrix S for positive case
vector<IDPair> COL_SORT_DATA_IDPAIR; // Sampling & Greedy

// LSH
MatrixXd MATRIX_LSH_SIM_HASH; // matrix of (L*K) x (D+1) normal random variables
I32Vector VECTOR_LSH_UNIVERSAL_HASH; // vector of (L*K) random integers

vector<unordered_map<uint64_t, IVector>> VECTOR_LSH_TABLES;
I64Vector VECTOR_LSH_CODES;
DVector VECTOR_LSH_PARTITION_NORM;

// CEOs
IVector MAX_COL_SORT_IDX;
MatrixXi MATRIX_CEOs_PRECOMPUTED_MIPS;

vector<IDPair> MAX_COL_SORT_DATA_IDPAIR;
vector<IDPair> MIN_COL_SORT_DATA_IDPAIR;

MatrixXd MATRIX_NORMAL_DISTRIBUTION;
MatrixXd PROJECTED_X;
MatrixXi PROJECT_X_SORTED_IDX;

IVector HD1;
IVector HD2;
IVector HD3;

// Random Projection
int PARAM_RP_D_DOWN;

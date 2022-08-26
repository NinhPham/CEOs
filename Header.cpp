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
int PARAM_MIPS_TOP_B_RANGE;

// CEOs
int PARAM_CEOs_D_UP;
int PARAM_INTERNAL_FWHT_PROJECTION; // in case D_up < D
int PARAM_INTERNAL_LOG2_FWHT_PROJECTION; // in case D_up < D
int PARAM_CEOs_S0;
int PARAM_CEOs_N_DOWN;
int PARAM_CEOs_NUM_ROTATIONS; // Use Gauss or HD3
int PARAM_INTERNAL_LOG2_CEOs_D_UP;
//bool PARAM_INTERNAL_NOT_STORE_MATRIX_X;


// Random Projection
int PARAM_RP_D_DOWN;

/*
- DATA STRUCTURE
*/

// Input
MatrixXf MATRIX_X;
MatrixXf MATRIX_Q;
//Matrix<double, Dynamic, Dynamic, RowMajor> MATRIX_X; // matrix X
//Matrix<double, Dynamic, Dynamic, ColMajor> MATRIX_Q; // matrix Q

// CEOs
MatrixXi MATRIX_1CEOs;
MatrixXi MATRIX_2CEOs;
MatrixXi CEOs_TA_SORTED_IDX;
vector<IFPair> coCEOs_MAX_IDX;
vector<IFPair> coCEOs_MIN_IDX;

MatrixXf MATRIX_G;
MatrixXf PROJECTED_X;
MatrixXi HD3;

boost::dynamic_bitset<> bitHD3; // all 0's by default

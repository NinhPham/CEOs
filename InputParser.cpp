#include "InputParser.h"
#include "Header.h"

#include <stdlib.h>     /* atoi */
#include <iostream> // cin, cout
#include <fstream> // fscanf, fopen, ofstream

#include <vector>
#include <string.h> // strcmp

int loadInput(int nargs, char** args)
{
    if (nargs < 6)
        exit(1);

    // Parse arguments: Note that don't know what args[0] represents for !!!
    PARAM_DATA_N = atoi(args[1]);
    cout << "Number of rows of X: " << PARAM_DATA_N << endl;

    PARAM_QUERY_Q = atoi(args[2]);
    cout << "Number of rows of Q: " << PARAM_QUERY_Q << endl;

    PARAM_DATA_D = atoi(args[3]);
    cout << "Number of dimensions: " << PARAM_DATA_D << endl;

    PARAM_MIPS_TOP_K = atoi(args[4]);
    cout << "Top K: " << PARAM_MIPS_TOP_K << endl;
    cout << endl;

    // Read the row-wise matrix X, and convert to col-major Eigen matrix
    int n, q, d;
    cout << "Read row-wise X, it will be converted to col-major Eigen matrix of size N x D..." << endl;
    if (args[5])
    {
        FILE *f = fopen(args[5], "r");
        if (!f)
        {
            printf("Data file does not exist");
            exit(1);
        }

        DVector vecTempX(PARAM_DATA_D * PARAM_DATA_N, 0.0);

        // Each line is a vector of D dimensions
        for (n = 0; n < PARAM_DATA_N; ++n)
        {
            for (d = 0; d < PARAM_DATA_D; ++d)
            {
                fscanf(f, "%lf", &vecTempX[n + d * PARAM_DATA_N]);
                // cout << vecTempX[n + d * PARAM_DATA_N] << " ";
            }
            // cout << endl;
        }

        // Matrix_X is col-major
        MATRIX_X = Map<MatrixXd>(vecTempX.data(), PARAM_DATA_N, PARAM_DATA_D);
        cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

        /**
        Print the first col (1 x N)
        Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
        **/
//        cout << MATRIX_X.col(0) << endl << endl;
//        cout << "In memory (col-major):" << endl;
//        for (n = 0; n < 10; n++)
//            cout << *(MATRIX_X.data() + n) << "  ";
//        cout << endl << endl;

    }

    // Read the row-wise matrix X, and convert to col-major Eigen matrix
    cout << "Read row-wise Q, it will be converted to col-major Eigen matrix of size D x Q..." << endl;
    if (args[6])
    {
        FILE *f = fopen(args[6], "r+");
        if (!f)
        {
            printf("Data file does not exist");
            exit(1);
        }

        DVector vecTempQ(PARAM_DATA_D * PARAM_QUERY_Q, 0.0);
        for (q = 0; q < PARAM_QUERY_Q; ++q)
        {
            for (d = 0; d < PARAM_DATA_D; ++d)
            {
                fscanf(f, "%lf", &vecTempQ[q * PARAM_DATA_D + d]);
                //cout << vecTempQ[q * D + d] << " ";
            }
            //cout << endl;
        }

        MATRIX_Q = Map<MatrixXd>(vecTempQ.data(), PARAM_DATA_D, PARAM_QUERY_Q);
        cout << "Q has rows " << MATRIX_Q.rows() << " and cols " << MATRIX_Q.cols() << endl;

        /**
        Print the first row (1 x Q)
        Print some of the first elements of the MATRIX_Q to see that these elements are on consecutive memory cell.
        **/

//        cout << MATRIX_Q.col(0) << endl << endl;
//        cout << "In memory (col-major):" << endl;
//        for (n = 0; n < 10; n++)
//            cout << *(MATRIX_Q.data() + n) << "  ";
//        cout << endl << endl;

    }

    cout << endl;

    // Algorithm
    int iType = 0;

    // Exact solution
    if (strcmp(args[7], "BF") == 0)
    {
        iType = 11;
        cout << "Bruteforce topK... " << endl;
        cout << endl;
    }

    // Sampling soluions
    else if (strcmp(args[7], "wedge") == 0)
    {
        iType = 21;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "Wedge: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "Wedge: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        // For pre-sampling
        PARAM_MIPS_PRESAMPLES_MAX = PARAM_DATA_N; //ceil(50 * PARAM_WEDGE_S / PARAM_DATA_D);

        cout << endl;
    }

    else if (strcmp(args[7], "dWedge") == 0)
    {
        iType = 22;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "dWedge: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "dWedge: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        cout << endl;
    }

    else if (strcmp(args[7], "shift_Wedge") == 0)
    {
        iType = 23;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "Shift Wedge: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "Shift Wedge: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        // For pre-sampling
        PARAM_MIPS_PRESAMPLES_MAX = PARAM_DATA_N; //ceil(50 * PARAM_WEDGE_S / PARAM_DATA_D);

        cout << endl;
    }

    else if (strcmp(args[7], "shift_dWedge") == 0)
    {
        iType = 24;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "Shift dWedge: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "Shift dWedge: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        cout << endl;
    }

    else if (strcmp(args[7], "pos_dWedge") == 0)
    {
        iType = 25;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "Positive dWedge: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "Positive dWedge: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        cout << endl;
    }

    else if (strcmp(args[7], "diamond") == 0)
    {
        iType = 31;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "Diamond: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "Diamond: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        // For pre-sampling
        PARAM_MIPS_PRESAMPLES_MAX = PARAM_DATA_N; //ceil(50 * PARAM_WEDGE_S / PARAM_DATA_D);

        cout << endl;
    }

    else if (strcmp(args[7], "dDiamond") == 0)
    {
        iType = 32;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "dDiamond: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "dDiamond: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        cout << endl;
    }

    else if (strcmp(args[7], "shift_Diamond") == 0)
    {
        iType = 33;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "Shift Diamond: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "Shift Diamond: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        // For pre-sampling
        PARAM_MIPS_PRESAMPLES_MAX = PARAM_DATA_N; //ceil(50 * PARAM_WEDGE_S / PARAM_DATA_D);

        cout << endl;
    }

    // Greedy
    else if (strcmp(args[7], "greedy") == 0)
    {
        iType = 41;

        PARAM_MIPS_SAMPLES = atoi(args[8]);
        cout << "Greedy: Total samples: S = " << PARAM_MIPS_SAMPLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "Greedy: Total dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        cout << endl;
    }

    // Simple LSH
    else if (strcmp(args[7], "simpleLSH_Table") == 0)
    {
        iType = 51;

        PARAM_LSH_HASH_FUNCTIONS = atoi(args[8]);
        cout << "SimpleLSH Table: Number of concatenated hash functions: K = " << PARAM_LSH_HASH_FUNCTIONS << "... " << endl;

        PARAM_LSH_HASH_TABLES = atoi(args[9]); //ceil(PARAM_MIPS_BUDGET / (2 * PARAM_DATA_D));
        cout << "SimpleLSH Table: Number of hash tables: L = " << PARAM_LSH_HASH_TABLES << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[10]);
        cout << "SimpleLSH Table: Number of dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        cout << endl;
    }

    // Simple LSH
    else if (strcmp(args[7], "simpleLSH_Code") == 0)
    {
        iType = 52;

        PARAM_LSH_HASH_FUNCTIONS = atoi(args[8]);
        cout << "SimpleLSH Code: Number of concatenated hash functions: K = " << PARAM_LSH_HASH_FUNCTIONS << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[9]);
        cout << "SimpleLSH Code: Number of dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        PARAM_LSH_NUM_DECIMAL = ceil((double)PARAM_LSH_HASH_FUNCTIONS / 64);
        cout << "SimpleLSH Code: Number of decimal values = " << PARAM_LSH_NUM_DECIMAL << "... " << endl;

        // Must set this param = 1 due to the SimHashGenerator()
        PARAM_LSH_HASH_TABLES = 1;

        cout << endl;
    }

    // Simple LSH
    else if (strcmp(args[7], "rangeLSH_Table") == 0)
    {
        iType = 53;

        PARAM_LSH_HASH_FUNCTIONS = atoi(args[8]);
        cout << "rangeLSH Table: Number of concatenated hash functions: K = " << PARAM_LSH_HASH_FUNCTIONS << "... " << endl;

        PARAM_LSH_HASH_TABLES = atoi(args[9]); //ceil(PARAM_MIPS_BUDGET / (2 * PARAM_DATA_D));
        cout << "rangeLSH Table: Number of hash tables: L = " << PARAM_LSH_HASH_TABLES << "... " << endl;

        PARAM_LSH_PARTITIONS = atoi(args[10]); //ceil(PARAM_MIPS_BUDGET / (2 * PARAM_DATA_D));
        cout << "rangeLSH Table: Number of partitions: P = " << PARAM_LSH_PARTITIONS << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[11]);
        cout << "rangeLSH Table: Number of dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        cout << endl;
    }

    // Simple LSH
    else if (strcmp(args[7], "rangeLSH_Code") == 0)
    {
        iType = 54;

        PARAM_LSH_HASH_FUNCTIONS = atoi(args[8]);
        cout << "rangeLSH Code: Number of concatenated hash functions: K = " << PARAM_LSH_HASH_FUNCTIONS << "... " << endl;

        PARAM_LSH_PARTITIONS = atoi(args[9]); //ceil(PARAM_MIPS_BUDGET / (2 * PARAM_DATA_D));
        cout << "rangeLSH Code: Number of partitions: P = " << PARAM_LSH_PARTITIONS << "... " << endl;

        PARAM_MIPS_DOT_PRODUCTS = atoi(args[10]);
        cout << "rangeLSH Code: Number of dot products: M = " << PARAM_MIPS_DOT_PRODUCTS << "... " << endl;

        PARAM_LSH_NUM_DECIMAL = ceil((double)PARAM_LSH_HASH_FUNCTIONS / 64);
        cout << "rangeLSH Code: Number of decimal values = " << PARAM_LSH_NUM_DECIMAL << "... " << endl;

        // Must set this param = 1 due to the SimHashGenerator()
        PARAM_LSH_HASH_TABLES = 1;

        cout << endl;
    }

    return iType;
}


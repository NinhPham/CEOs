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

    //cout << "Read row-wise X, it will be converted to col-major Eigen matrix of size D x N..." << endl;
    if (args[5])
    {
        FILE *f = fopen(args[5], "r");
        if (!f)
        {
            printf("Data file does not exist");
            exit(1);
        }

        FVector vecTempX(PARAM_DATA_D * PARAM_DATA_N, 0.0);

        // Each line is a vector of D dimensions
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            for (int d = 0; d < PARAM_DATA_D; ++d)
            {
                fscanf(f, "%f", &vecTempX[n * PARAM_DATA_D + d]);
                // cout << vecTempX[n * PARAM_DATA_D + d] << " ";
            }
            // cout << endl;
        }

        // Matrix_X is col-major
        MATRIX_X = Map<MatrixXf>(vecTempX.data(), PARAM_DATA_D, PARAM_DATA_N);
        //cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

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
    //cout << "Read row-wise Q, it will be converted to col-major Eigen matrix of size D x Q..." << endl;
    if (args[6])
    {
        FILE *f = fopen(args[6], "r+");
        if (!f)
        {
            printf("Data file does not exist");
            exit(1);
        }

        FVector vecTempQ(PARAM_DATA_D * PARAM_QUERY_Q, 0.0);
        for (int q = 0; q < PARAM_QUERY_Q; ++q)
        {
            for (int d = 0; d < PARAM_DATA_D; ++d)
            {
                fscanf(f, "%f", &vecTempQ[q * PARAM_DATA_D + d]);
                //cout << vecTempQ[q * D + d] << " ";
            }
            //cout << endl;
        }

        MATRIX_Q = Map<MatrixXf>(vecTempQ.data(), PARAM_DATA_D, PARAM_QUERY_Q);
        //cout << "Q has rows " << MATRIX_Q.rows() << " and cols " << MATRIX_Q.cols() << endl;

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

    // Concomitant order statistic (COS)
    else if (strcmp(args[7], "CEOs_Est") == 0)
    {
        iType = 20;

        PARAM_CEOs_D_UP = atoi(args[8]);
        cout << "CEOs-Est: Number of up dimensions: D_UP = " << PARAM_CEOs_D_UP << "... " << endl;

        if (PARAM_CEOs_D_UP < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_CEOs_D_UP;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_CEOs_S0 = atoi(args[9]);
        cout << "CEOs-Est: Number of down dimensions: S0 = " << PARAM_CEOs_S0 << "... " << endl;

        PARAM_MIPS_TOP_B = atoi(args[10]);
        cout << "CEOs-Est: Total dot products: B =  " << PARAM_MIPS_TOP_B << "... " << endl;
    }

    // CEOs with Threhold algorithm
    else if (strcmp(args[7], "CEOs_TA") == 0)
    {
        iType = 21;

        PARAM_CEOs_D_UP = atoi(args[8]);
        cout << "CEOs-TA: Number of up dimensions: D_UP = " << PARAM_CEOs_D_UP << "... " << endl;

        if (PARAM_CEOs_D_UP < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_CEOs_D_UP;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_CEOs_S0 = atoi(args[9]);
        cout << "CEOs-TA: Number of down dimensions: S0 = D_DOWN = " << PARAM_CEOs_S0 << "... " << endl;

        PARAM_MIPS_TOP_B = atoi(args[10]);
        cout << "CEOs-TA: Total dot products: B =  " << PARAM_MIPS_TOP_B << "... " << endl;
    }

    // Concomitant order statistic (COS)
    else if (strcmp(args[7], "coCEOs") == 0)
    {
        iType = 22;

        PARAM_CEOs_D_UP = atoi(args[8]);
        cout << "coCEOs: Number of up dimensions: D_UP = " << PARAM_CEOs_D_UP << "... " << endl;

        if (PARAM_CEOs_D_UP < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_CEOs_D_UP;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_CEOs_S0 = atoi(args[9]);
        cout << "coCEOs: Number of down dimensions: S0 = D_DOWN = " << PARAM_CEOs_S0 << "... " << endl;

        PARAM_CEOs_N_DOWN = atoi(args[10]);
        cout << "coCEOs: Number of down points: N0 = N_DOWN = " << PARAM_CEOs_N_DOWN << "... " << endl;

        PARAM_MIPS_TOP_B = atoi(args[11]);
        cout << "coCEOs: Total dot products: B =  " << PARAM_MIPS_TOP_B << "... " << endl;

    }

    else if (strcmp(args[7], "1CEOs") == 0)
    {
        iType = 23;

        PARAM_CEOs_D_UP = atoi(args[8]);
        cout << "1CEOs: Number of up dimensions: D_UP = " << PARAM_CEOs_D_UP << "... " << endl;

        if (PARAM_CEOs_D_UP < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_CEOs_D_UP;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_MIPS_TOP_B = atoi(args[9]);
        cout << "1CEOs Search: Total dot products: B =  " << PARAM_MIPS_TOP_B << "... " << endl;
    }

    else if (strcmp(args[7], "2CEOs") == 0)
    {
        iType = 24;

        PARAM_CEOs_D_UP = atoi(args[8]);
        cout << "2CEOs: Number of up dimensions: D_UP = " << PARAM_CEOs_D_UP << "... " << endl;

        if (PARAM_CEOs_D_UP < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_CEOs_D_UP;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_MIPS_TOP_B = atoi(args[9]);
        cout << "2CEOs Search: Total dot products: B =  " << PARAM_MIPS_TOP_B << "... " << endl;
    }

    else if (strcmp(args[7], "test_coCEOs_TopB") == 0)
    {
        iType = 221;

        PARAM_CEOs_D_UP = atoi(args[8]);
        cout << "coCEOs: Number of up dimensions: D_UP = " << PARAM_CEOs_D_UP << "... " << endl;

        if (PARAM_CEOs_D_UP < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_CEOs_D_UP;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_CEOs_S0 = atoi(args[9]);
        cout << "coCEOs: Number of down dimensions: S0 = D_DOWN = " << PARAM_CEOs_S0 << "... " << endl;

        PARAM_CEOs_N_DOWN = atoi(args[10]);
        cout << "coCEOs: Number of down points: N0 = N_DOWN = " << PARAM_CEOs_N_DOWN << "... " << endl;

        PARAM_MIPS_TOP_B_RANGE = atoi(args[11]);
        cout << "coCEOs: Range of # dot products: B =  " << PARAM_MIPS_TOP_B_RANGE << "... " << endl;

    }

    // Precompute some internal paramemters
    PARAM_INTERNAL_LOG2_FWHT_PROJECTION = log2(PARAM_INTERNAL_FWHT_PROJECTION);

    return iType;
}


// #include <omp.h>

#include "Header.h"
#include "InputParser.h"
#include "Utilities.h"

#include "BF.h"

#include "WedgeSampling.h"

#include "Greedy.h"

#include "SimpleLSH.h"
#include "NormRangeLSH.h"

#include "Concomitant.h"

#include <limits> // for max and min of integer
#include <cstdlib>
#include <chrono> // for wall clock time

/**
Parameters:
- N : number of rows of matrix X
- Q : number of rows of matrix Q
- D : number of dimensions
- file_name_X : filename contains the matrix X of format N x D
- file_name_Q : filename contains the matrix Y of format Q x D
- method and its paramenters : name of method

Notes:
- The format of dataset should be the matrix format N x D
**/


int main(int nargs, char** args)
{

    srand(time(NULL)); // should only be called once for random generator

    /************************************************************************/
	/* Load input file                                                      */
	/************************************************************************/
	int iType = loadInput(nargs, args);

    /*************************************************************************
    Parameter for testing
    1) PARAM_INTERNAL_SAVE_OUTPUT (bool): save output file (each line = pointIdx, dotProductValue)
    2) PARAM_INTERNAL_GAUSS_HD3: Use random rotation
    3) PARAM_INTERNAL_NOT_STORE_MATRIX_X: Use PROJECT_X to compute MIPS, only work with PARAM_INTERNAL_GAUSS_HD3
    *************************************************************************/

    PARAM_INTERNAL_SAVE_OUTPUT = false; // saving results
    PARAM_INTERNAL_NOT_STORE_MATRIX_X = false;
	PARAM_INTERNAL_GAUSS_HD3 = true;

	double dStart;
    chrono::steady_clock::time_point begin, end;

	/************************************************************************/
	/* Algorithms                                             */
	/************************************************************************/
	switch (iType)
	{

    case 11: // Bruteforce topK

        dStart = clock();
        begin = chrono::steady_clock::now();

        BF_TopK_MIPS();

        end = chrono::steady_clock::now();
        cout << "BF Wall Clock = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;
        printf("Bruteforce topK time is %f \n", getCPUTime(clock() - dStart));

        break;

    case 21: // dWedge

        /**
        Pre-processing data
        **/
        dStart = clock();
        dimensionSort();
        printf("dWedge CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        if (PARAM_MIPS_SAMPLES >= PARAM_DATA_N / 2)
            dWedge_Vector_TopK();
        else
            dWedge_Map_TopK();

        printf("dWedge CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "dWedge Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 22: // Greedy-MIPS

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        dimensionSort(false);
        printf("Greedy CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        greedy_TopK();

        printf("Greedy CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Greedy Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

	case 31: // Simple LSH

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_SimpleLSH_Table();

        printf("SimpleLSH Table CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        simpleLSH_Table_TopK();

        printf("SimpleLSH Table CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "SimpleLSH Table Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 32: // Norm Range LSH

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_RangeLSH_Table();

        printf("rangeLSH Table CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        rangeLSH_Table_TopK();

        printf("rangeLSH Table CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "rangeLSH Table Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 40: // SimHash Code

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_SimHash_Code();

        printf("SimHash Code CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        SimHash_Code_TopK();

        printf("SimHash Code CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "SimHash Code Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

	case 41: // Simple LSH Code

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_SimpleLSH_Code();

        printf("SimpleLSH Code CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        simpleLSH_Code_TopK();

        printf("SimpleLSH Code CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "SimpleLSH Code Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;


    case 42: // Norm Range LSH Code

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_RangeLSH_Code();

        printf("rangeLSH Code CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        rangeLSH_Code_TopK();

        printf("rangeLSH Code CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "rangeLSH Code Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 50: // sCEOs Estimation (specific dimensionality reduction)

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();

        GaussianRP_Data();

        cout << "Finish projection !!!" << endl;
        //system("PAUSE");

        if (PARAM_INTERNAL_GAUSS_HD3 && PARAM_INTERNAL_NOT_STORE_MATRIX_X)
        {
            cout << "We clean our memory for MATRIX_X " << endl;
            MATRIX_X.resize(0, 0); // HD3 preserve inner products
        }

        printf("sCEOs Est CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        sCEOs_Est_TopK();

        printf("sCEOs Est Top-K CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "sCEOs Est Top-K Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 51: // sCEOs Threshold algorithm

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_sCEOs_TA();

        cout << "Finish projection !!!" << endl;
        //system("PAUSE");

        if (PARAM_INTERNAL_GAUSS_HD3 && PARAM_INTERNAL_NOT_STORE_MATRIX_X)
        {
            cout << "We clean our memory for MATRIX_X " << endl;
            MATRIX_X.resize(0, 0); // HD3 preserve inner products
        }

        printf("sCEOs TA CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        sCEOs_TA_TopK();

        printf("sCEOs TA Top-K CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "sCEOs TA Top-K Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 52: // coCEOs - Search

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_coCEOs_Search();

        printf("coCEOs Search CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        coCEOs_Map_Search();

        printf("coCEOs Map Search CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "coCEOs Map Search Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl << endl;

        // Vector
        dStart = clock();
        begin = chrono::steady_clock::now();

        coCEOs_Vector_Search();

        printf("\n coCEOs Vector Search CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "coCEOs Vector Search Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;


        break;

    case 53: // 1CEOs - Hash

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_1CEOs_Hash();
        printf("CEOs Hash CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        maxCEOs_Hash();

        printf("CEOs Hash CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "CEOs Hash Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 54: // 1CEOs - Search

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_1CEOs_Search();
        printf("CEOs Search CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        maxCEOs_Search();

        printf("CEOs Search CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "CEOs Search Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 55: // 2CEOs - Search

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_2CEOs_Search();
        printf("2CEOs Search CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        minmaxCEOs_Search();

        printf("2CEOs Search CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "2CEOs Search Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;


    case 59: // Random Projection

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        GaussianRP();

        printf("Random Projection CPU: preprocessing time in second is %f \n\n", getCPUTime(clock() - dStart));
        //system("PAUSE");

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        RP_Est_TopK();

        printf("Random Projection Top-K CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Random Projection Top-K Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

	}

	//system("PAUSE");

	return 0;
}

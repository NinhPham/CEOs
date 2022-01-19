<<<<<<< HEAD
// #include <omp.h>

#include "Header.h"
#include "InputParser.h"
#include "Utilities.h"

#include "BF.h"
#include "WedgeSampling.h"
#include "NormRangeLSH.h"
#include "Concomitant.h"

//#include <limits> // for max and min of integer




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

    getRAM();

    /************************************************************************/
	/* Load input file                                                      */
	/************************************************************************/
	int iType = loadInput(nargs, args);

    getRAM();

    /*************************************************************************
    Parameter for testing
    1) PARAM_INTERNAL_SAVE_OUTPUT (bool): save output file (each line = pointIdx, dotProductValue)
    2) PARAM_INTERNAL_GAUSS_HD3: Use random rotation
    *************************************************************************/

    PARAM_INTERNAL_SAVE_OUTPUT = true; // saving results
	PARAM_CEOs_NUM_ROTATIONS = 3;
    PARAM_INTERNAL_NOT_STORE_MATRIX_X = false;

	/************************************************************************/
	/* Algorithms                                             */
	/************************************************************************/

	switch (iType)
	{

        // Bruteforce topK
        case 11:
        {
            auto startTime = chrono::high_resolution_clock::now();

            BF_TopK_MIPS();

            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
            printf("Bruteforce topK time in second is %f \n\n", (float)durTime.count() / 1000000);

            getRAM();

            break;
        }

        // sCEOs Estimation (specific dimensionality reduction)
        case 20:
        {
            // Building index
            auto start = chrono::high_resolution_clock::now();
            rotateData();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building sCEOs-Est index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();

//            for (int i = 1; i <= 5; ++i)
//            {
//                PARAM_MIPS_TOP_B = 10 * i;
//
//                cout << "Top B = " << PARAM_MIPS_TOP_B << endl;
//                sCEOs_Est_TopK();
//            }

            sCEOs_Est_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("sCEOs-Est Top-K time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 21: // sCEOs Threshold algorithm
        {
            // Building index
            auto start = chrono::high_resolution_clock::now();
            build_sCEOs_TA_Index();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building sCEOs-TA index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
//            for (int i = 1; i <= 5; ++i)
//            {
//                PARAM_MIPS_TOP_B = 10 * i;
//
//                cout << "Top B = " << PARAM_MIPS_TOP_B << endl;
//                sCEOs_TA_TopK();
//            }

            sCEOs_TA_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("sCEOs-TA query time in second is %f \n\n", (float)durTime.count() / 1000000);
            break;
        }

        case 22: // coCEOs - Search
        {

            // Building index
            auto start = chrono::high_resolution_clock::now();
            build_coCEOs_Index();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building coCEOs index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
//            start = chrono::high_resolution_clock::now();
//            for (int i = 1; i <= 5; ++i)
//            {
//                PARAM_MIPS_TOP_B = 10 * i;
//
//                cout << "Top B = " << PARAM_MIPS_TOP_B << endl;
//                coCEOs_Map_TopK();
////                coCEOs_Vector_TopK();
//            }

            // Heuristic to decide using map or vector
            if (PARAM_MIPS_NUM_SAMPLES <= PARAM_DATA_N / 2)
                coCEOs_Map_TopK();
            else
                coCEOs_Vector_TopK();

            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("coCEOs query time in second is %f \n", (float)durTime.count() / 1000000);

            break;
        }

        case 23: // 1CEOs
        {
            // Building index
            auto start = chrono::high_resolution_clock::now();
            build_1CEOs_Index();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building 1CEOs index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
            maxCEOs_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("1CEOs query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 24: // 2CEOs - Search
        {
            auto start = chrono::high_resolution_clock::now();
            build_2CEOs_Index();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building 2CEOs index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
            minmaxCEOs_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("2CEOs query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 29: // Random Projection
        {

            auto start = chrono::high_resolution_clock::now();
            GaussianRP();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Conventional RP index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
            RP_Est_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Conventional RP query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 31: // dWedge
        {
            auto start = chrono::high_resolution_clock::now();
            dimensionSort();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building dWedge index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
//            start = chrono::high_resolution_clock::now();
//            for (int i = 1; i <= 5; ++i)
//            {
//                PARAM_MIPS_TOP_B = 10 * i;
//
//                cout << "Top B = " << PARAM_MIPS_TOP_B << endl;
//                dWedge_Map_TopK();
//                dWedge_Vector_TopK();
//            }

            if (PARAM_MIPS_NUM_SAMPLES <= PARAM_DATA_N / 2)
                dWedge_Map_TopK();
            else
                dWedge_Vector_TopK();

            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("dWedge query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 32: // Greedy-MIPS
        {
            auto start = chrono::high_resolution_clock::now();
            dimensionSort(false);
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building Greedy index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
            for (int i = 1; i <= 5; ++i)
            {
                PARAM_MIPS_TOP_B = 10 * i;

                cout << "Top B = " << PARAM_MIPS_TOP_B << endl;
                greedy_TopK();
            }

//            greedy_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Greedy query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 41: // Norm Range LSH
        {
            auto start = chrono::high_resolution_clock::now();
            build_RangeLSH_Table();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building RangeLSH Table index time in second is %f \n", (float)durTime.count() / 1000000);

            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
            rangeLSH_Table_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("RangeLSH Table query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 50: // SimHash Code
        {
            auto start = chrono::high_resolution_clock::now();
            build_SimHash_Code();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building SimHash Code index time in second is %f \n", (float)durTime.count() / 1000000);

            // Querying
            start = chrono::high_resolution_clock::now();
            simHash_Code_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("SimHash Code query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }

        case 51: // Norm Range LSH Code
        {
            auto start = chrono::high_resolution_clock::now();
            build_RangeLSH_Code();
            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building RangeLSH Code index time in second is %f \n", (float)durTime.count() / 1000000);

            // Querying
            start = chrono::high_resolution_clock::now();
            rangeLSH_Code_TopK();
            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
            printf("RangeLSH Code query time in second is %f \n\n", (float)durTime.count() / 1000000);

            break;
        }
	}

	//system("PAUSE");

	return 0;
}
=======
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
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

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

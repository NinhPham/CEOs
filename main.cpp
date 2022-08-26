#include "Header.h"
#include "InputParser.h"
#include "Utilities.h"

#include "BF.h"
#include "Test.h"
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

// TODO: Only maintaining coCEOs since it tends to be better than CEOs-TA

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
	PARAM_CEOs_NUM_ROTATIONS = 3; // = 0 then use Gaussian matrix
//    PARAM_INTERNAL_NOT_STORE_MATRIX_X = false;

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

            auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
            printf("Bruteforce topK time in ms is %f \n\n", (float)durTime.count());

//            getRAM();

            break;
        }

        // CEOs-Est (specific dimensionality reduction)
        case 20:
        {
            // Building index
            auto start = chrono::high_resolution_clock::now();
            rotateData();
            auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building sCEOs-Est index time in ms is %f \n", (float)durTime.count());

//            getRAM();

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
            durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("sCEOs-Est Top-K time in ms is %f \n\n", (float)durTime.count());

            break;
        }

        case 21: // CEOs-TA algorithm
        {
            // Building index
            auto start = chrono::high_resolution_clock::now();
            build_sCEOs_TA_Index();
            auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building sCEOs-TA index time in ms is %f \n", (float)durTime.count());

//            getRAM();

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
            durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("sCEOs-TA query time in ms is %f \n\n", (float)durTime.count());
            break;
        }

        case 22: // coCEOs - Search
        {

            // Building index
            auto start = chrono::high_resolution_clock::now();
            build_coCEOs_Index();
            auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building coCEOs index time in ms is %f \n", (float)durTime.count());

//            getRAM();

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
            if (2 * PARAM_CEOs_S0 * PARAM_CEOs_N_DOWN <= PARAM_DATA_N / 2)
                coCEOs_Map_TopK();
            else
                coCEOs_Vector_TopK();

            durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("coCEOs query time in ms is %f \n", (float)durTime.count());

            break;
        }

        case 23: // 1CEOs
        {
            // Building index
            auto start = chrono::high_resolution_clock::now();
            build_1CEOs_Index();
            auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building 1CEOs index time in ms is %f \n", (float)durTime.count());

//            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
            maxCEOs_TopK();
            durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("1CEOs query time in ms is %f \n\n", (float)durTime.count());

            break;
        }

        case 24: // 2CEOs - Search
        {
            auto start = chrono::high_resolution_clock::now();
            build_2CEOs_Index();
            auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("Building 2CEOs index time in ms is %f \n", (float)durTime.count());

//            getRAM();

            // Querying
            start = chrono::high_resolution_clock::now();
            minmaxCEOs_TopK();
            durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
            printf("2CEOs query time in ms is %f \n\n", (float)durTime.count());

            break;
        }

        case 221:
        {
            test_coCEOs_TopB();
            break;
        }
	}

	//system("PAUSE");

	return 0;
}

// This C++ source code is written by Ninh Pham (ninh.pham@auckland.ac.nz) and Stephan Lorenzen (lorenzen@di.ku.dk) as a part of the DABAI project
// Feel free to re-use and re-distribute this source code for any purpose,
// And cite our work when you re-use this code.

// #include <omp.h>

#include "Header.h"
#include "InputParser.h"
#include "Utilities.h"

#include "Test.h"
#include "BF.h"

#include "WedgeSampling.h"
#include "DiamondSampling.h"

#include "Greedy.h"

#include "SimpleLSH.h"
#include "NormRangeLSH.h"

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

//    #pragma omp parallel
//    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());

    srand(time(NULL)); // should only be called once for random generator

    /************************************************************************/
	/* Load input file                                                      */
	/************************************************************************/
	int iType = loadInput(nargs, args);

    /*************************************************************************
    Internal parameters for testing
    1) PARAM_TEST_SAVE_OUTPUT (bool): save output file (each line = pointIdx, dotProductValue)
    2) PARAM_TEST_SAMPLING_RANDOM_GENERATOR (int):
    - 1: C++ (uniform generator)
    - 2: Diamond sampling suggestion
    - 3: GreedySam (see arXiv version: https://arxiv.org/abs/1908.08656)
    *************************************************************************/


	PARAM_TEST_SAVE_OUTPUT = true; // saving results
	PARAM_TEST_SAMPLING_RANDOM_GENERATOR = 3;

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

    case 21: // standard wedgeSampling

        /**
        Pre-processing data
        **/
        dStart = clock();
        wedge_PreSampling();
        printf("Wedge CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        if (PARAM_MIPS_SAMPLES >= PARAM_DATA_N / 2)
            wedge_Vector_TopK();
        else
            wedge_Map_TopK();

        printf("Wedge CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Wedge Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;


    case 22: // dWedge

        /**
        Pre-processing data
        **/
        dStart = clock();
        dimensionSort();
        printf("dWedge CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

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

    case 23: // shift Wedge Sampling

        /**
        Pre-processing data
        **/
        dStart = clock();
        shift_Wedge_PreSampling();
        printf("shiftWedge CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        if (PARAM_MIPS_SAMPLES >= PARAM_DATA_N / 2)
            shift_Wedge_Vector_TopK();
        else
            shift_Wedge_Map_TopK();

        printf("Shift Wedge CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Shift Wedge Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 24: // Shift dWedge

        /**
        Pre-processing data
        **/
        dStart = clock();
        dimensionShiftSort();
        printf("Shift dWedge CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        if (PARAM_MIPS_SAMPLES >= PARAM_DATA_N / 2)
            shift_dWedge_Vector_TopK();
        else
            shift_dWedge_Map_TopK();

        printf("Shift dWedge CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Shift dWedge Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 25: // Pos dWedge

        /**
        Pre-processing data
        **/
        dStart = clock();
        dimensionPosSort();
        printf("Positive dWedge CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        if (PARAM_MIPS_SAMPLES >= PARAM_DATA_N / 2)
            pos_dWedge_Vector_TopK();
        else
            pos_dWedge_Map_TopK();

        printf("Positive dWedge CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Positive dWedge Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 31: // diamond

        /**
        Pre-processing data
        **/
        dStart = clock();
        wedge_PreSampling();
        printf("Diamond CPU: Preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        diamond_Vector_TopK();

        printf("Diamond CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Diamond Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 32: // heuristic diamond

        /**
        Pre-processing data
        **/
        dStart = clock();
        dimensionSort();
        printf("dDiamond CPU: Preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        dDiamond_Vector_TopK();

        printf("dDiamond CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "dDiamond Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 33: // shiftDiamond

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        shift_Wedge_PreSampling();
        printf("shiftDiamond CPU: Preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        shift_Diamond_Vector_TopK();

        printf("Shift Diamond CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Shift Diamond Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

    case 41: // greedy

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        // greedyPreProcessing();
        dimensionSort(false);
        printf("Greedy CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

        /**
        Query
        **/
        dStart = clock();
        begin = chrono::steady_clock::now();

        // greedyTopK_Post_NIPS17();
        greedy_TopK();

        printf("Greedy CPU: Time is %f \n", getCPUTime(clock() - dStart));
        end = chrono::steady_clock::now();
        cout << "Greedy Wall: Time is " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;

        break;

	case 51: // Simple LSH

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_SimpleLSH_Table();
        printf("SimpleLSH Table CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

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

	case 52: // Simple LSH Code

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_SimpleLSH_Code();
        printf("SimpleLSH Code CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

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

    case 53: // Norm Range LSH

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_RangeLSH_Table();
        printf("rangeLSH Table CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

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

    case 54: // Norm Range LSH Code

        //---------------------------------
        // Pre-processing data
        //---------------------------------
        dStart = clock();
        build_RangeLSH_Code();
        printf("rangeLSH Code CPU: preprocessing time in ms is %f \n", getCPUTime(clock() - dStart));

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
	}

	//system("PAUSE");

	return 0;
}

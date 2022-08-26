#include "Test.h"
#include "Concomitant.h"
#include "Utilities.h"
#include "Header.h"

/**
Test coCEO by varying topB

**/
void test_coCEOs_TopB()
{
    // Building index
    auto start = chrono::high_resolution_clock::now();
    build_coCEOs_Index();
    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    printf("Building coCEOs index time in ms is %f \n", (float)durTime.count());

    getRAM();

    // Querying
    start = chrono::high_resolution_clock::now();
    for (int i = 1; i <= 20; ++i)
    {
        PARAM_MIPS_TOP_B = PARAM_MIPS_TOP_B_RANGE * i;

        cout << "Top B = " << PARAM_MIPS_TOP_B << endl;

        // Heuristic to decide using map or vector
        if (2 * PARAM_CEOs_S0 * PARAM_CEOs_N_DOWN <= PARAM_DATA_N / 2)
            coCEOs_Map_TopK();
        else
            coCEOs_Vector_TopK();
    }
}

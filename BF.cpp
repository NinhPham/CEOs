#include "BF.h"
#include "Utilities.h"
#include "Header.h"

/**
Retrieve topK MIPS entries using brute force computation

Input:
- MatrixXd: MATRIX_X (col-major) of size N x D
- MatrixXd: MATRIX_Q (col-major) of size D x Q

**/
void BF_TopK_MIPS()
{
    double dStart1 = clock();

    int n, q;
    double dValue = 0;
    double dStart = 0, dTopKTime = 0, dDotTime = 0;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> queTopK;
    VectorXd vecRes(PARAM_DATA_N);
    VectorXd vecQuery(PARAM_DATA_D);

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Reset
        queTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>();

        vecQuery = MATRIX_Q.col(q); // D x 1

        // Compute vector-matrix multiplication
        vecRes = MATRIX_X * vecQuery;

        // cout << vecRes.maxCoeff() << endl;

        dDotTime += clock() - dStart;

        // Insert into priority queue to get TopK
        dStart = clock();
        for (n = 0; n < PARAM_DATA_N; ++n)
        {
            dValue = vecRes(n);

            //cout << dValue << endl;

            // If we do not have enough top K point, insert
            if (int(queTopK.size()) < PARAM_MIPS_TOP_K)
                queTopK.push(IDPair(n, dValue));
            else // we have enough,
            {
                if (dValue > queTopK.top().m_dValue)
                {
                    queTopK.pop();
                    queTopK.push(IDPair(n, dValue));
                }
            }
        }

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveQueue(queTopK, "BF_TopK_" + int2str(q) + ".txt");
    }

    printf("Computing dot products time is %f \n", getCPUTime(dDotTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("BF time is %f \n", getCPUTime(clock() - dStart1));
}

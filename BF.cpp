#include "BF.h"
#include "Utilities.h"
#include "Header.h"

/**
Retrieve topK MIPS entries using brute force computation

Input:
- MatrixXd: MATRIX_X (col-major) of size D x N
- MatrixXd: MATRIX_Q (col-major) of size D x Q

**/
void BF_TopK_MIPS()
{
    auto start = chrono::high_resolution_clock::now();
    float distTime = 0, topKTime = 0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q); // K x Q

    // For each query
    #pragma omp parallel for
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        priority_queue<IFPair, vector<IFPair>, greater<IFPair>> queTopK;

        VectorXf vecQuery = MATRIX_Q.col(q); // D x 1
        VectorXf vecRes = vecQuery.transpose() * MATRIX_X;

        // cout << vecRes.maxCoeff() << endl;
        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        distTime += (float)durTime.count() / 1000;

        // Insert into priority queue to get TopK
        startTime = chrono::high_resolution_clock::now();
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            float fValue = vecRes(n);

            //cout << fValue << endl;

            // If we do not have enough top K point, insert
            if (int(queTopK.size()) < PARAM_MIPS_TOP_K)
                queTopK.push(IFPair(n, fValue));
            else // we have enough,
            {
                if (fValue > queTopK.top().m_fValue)
                {
                    queTopK.pop();
                    queTopK.push(IFPair(n, fValue));
                }
            }
        }

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000;

        // Save result into mat_topK
        // Note that priorityQueue pop smallest element firest

        IVector vecTopK(PARAM_MIPS_TOP_K, -1);
        for (int k = PARAM_MIPS_TOP_K - 1; k >= 0; --k)
        {
            vecTopK[k] = queTopK.top().m_iIndex;
            queTopK.pop();
        }

        matTopK.col(q) = Map<VectorXi>(vecTopK.data(), PARAM_MIPS_TOP_K);

    }

    printf("Distance time in second is %f \n", distTime);
    printf("Extract TopK time in second is %f \n", topKTime);

    auto duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
    printf("BF Querying Time in second is %f \n", (float)duration.count() / 1000000);

    //cout << matTopK << endl;

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "BF_TopK_" + int2str(PARAM_MIPS_TOP_K) + ".txt";

        outputFile(matTopK, sFileName);
    }
}

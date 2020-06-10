#include "DiamondSampling.h"
#include "Utilities.h"
#include "Header.h"

/** \brief Return approximate TopK of MAD for each query with Wedge (use vector to store samples)
 - Using priority queue to find top K occurrences
 *
 * \param
 *
 - POS_PRESAMPLES: pre-sampled set for each column
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1: vector of norm-1 of each dimension
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void diamond_Vector_TopK()
{
    double dStart0 = clock();

    int q, d, s, pointIdx, iNumSamples, iDim;
    double dValue = 0, Qj = 0.0;
    int iSignQj, iSignXij;
    double dStart = 0, dSamTime = 0, dTopBTime = 0, dTopKTime = 0, dBasicSamTime = 0, dStart1 = 0;

    DVector vecWeight(PARAM_DATA_D, 0.0); // number of samples for each dimension
    DVector vecCounter(PARAM_DATA_N, 0.0); // counting histogram of N points
//    IVector vecSampleCounter(PARAM_DATA_N, 0); // counting the number of samples used for each points in order to measure the correlation

    VectorXd vecQuery(PARAM_DATA_D); // vector of query
    DVector vecCDF;

    vector<double>::iterator low;

    // Presamples
    vector<int> vecWedgeSamples, vecQuerySamples, vecTopB;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1
        vecCDF = CDF(vecQuery);

        //---------------------------
        // Compute the sum of N dot products and its dimension weight
        //---------------------------
        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeWeight(vecQuery, vecWeight);

        // Get samples from each dimension, store in a vector for faster sequential access
        fill(vecCounter.begin(), vecCounter.end(), 0.0);
//        fill(vecSampleCounter.begin(), vecSampleCounter.end(), 0.0);

        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            Qj = vecQuery(d);

            if (Qj == 0)
                continue;

            iSignQj = sgn(Qj); //sign of Qj

            // Get number of samples for each dimension
            iNumSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);

            getDeterSamples(vecWedgeSamples, d, iNumSamples, 1); // always get POS info

            // pre-sort point indexes so that we do not have many random accesses
            // Only work for diamond sampling since we have to access the DATA for executing the second basic sampling step
            sort(vecWedgeSamples.begin(), vecWedgeSamples.end());

            // For each samples
            for (s = 0; s < iNumSamples; ++s)
            {
                pointIdx = vecWedgeSamples[s]; // get Xi and its sign

                dStart1 = clock();

                // Get sign of Xij and the index
                iSignXij = sgn(pointIdx);
                pointIdx = abs(pointIdx);

                // Sample from CDF of the query - this is the bootleneck of diamond sampling: O(logD)
                low = lower_bound(vecCDF.begin(), vecCDF.end(), (double)(rand() / RAND_MAX));
                //printVector(vecCDF);

                if (low == vecCDF.end())
                    iDim = PARAM_DATA_D - 1;
                else
                    iDim = low - vecCDF.begin(); // get j'

                dValue = MATRIX_X(pointIdx, iDim); //get Xij' - this is the bootleneck of diamond sampling
                dBasicSamTime += clock() - dStart1;

                vecCounter[pointIdx] += iSignQj * iSignXij * sgn(vecQuery(iDim)) * dValue;

//                vecSampleCounter[pointIdx]++;
            }

        }

        dSamTime += clock() - dStart;

        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "diamondCounter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "diamond_Vector_TopK_NoPost_" + int2str(q) + ".txt");

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_TEST_SAVE_OUTPUT)
            saveQueue(minQueTopK, "diamond_Vector_TopK_Post_" + int2str(q) + ".txt");
    }


    // Print time complexity of each step
    printf("Basic Sampling time is %f \n", getCPUTime(dBasicSamTime));
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Vector-Diamond: Time is %f \n", getCPUTime(clock() - dStart0));
}

/** \brief Return approximate TopK of MAD for each query with dWedge (use vector to store samples)
 - Using priority queue to find top K occurrences
 *
 * \param
 *
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1: vector of norm-1 of each dimension
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void dDiamond_Vector_TopK()
{
    double dStart0 = clock();

    int q, d, s, iPointIdx, iColSamples, iDim;
    double dValue = 0, Qj = 0.0;
    int iSignQj, iSignXij, iCount, iSamples;
    double dStart = 0, dSamTime = 0, dTopBTime = 0, dTopKTime = 0;

    DVector vecWeight(PARAM_DATA_D, 0.0); // number of samples for each dimension
    DVector vecCounter(PARAM_DATA_N, 0.0); // counting histogram of N points
//    IVector vecSampleCounter(PARAM_DATA_N, 0); // counting the number of samples used for each points in order to measure the correlation

    VectorXd vecQuery(PARAM_DATA_D); // vector of query
    DVector vecCDF;

    vector<double>::iterator low;
    vector<IDPair>::iterator iter;

    // Presamples
    IVector vecTopB;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1
        vecCDF = CDF(vecQuery);

        //---------------------------
        // Compute the sum of N dot products and its dimension weight
        //---------------------------
        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeWeight(vecQuery, vecWeight);

        // Get samples from each dimension, store in a vector for faster sequential access
        fill(vecCounter.begin(), vecCounter.end(), 0.0);
//        fill(vecSampleCounter.begin(), vecSampleCounter.end(), 0.0);

        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            Qj = vecQuery(d);

            if (Qj == 0)
                continue;

            iSignQj = sgn(Qj); //sign of Qj

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);

            iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N;

            // Reset counting samples
            iCount = 0;
            while (iCount < iColSamples)
            {
                // number of samples
                iSamples = ceil((*iter).m_dValue * iColSamples / POS_COL_NORM_1[d]);

                iCount += iSamples;

                iPointIdx = abs((*iter).m_iIndex);
                iSignXij = sgn((*iter).m_iIndex);

                for (s = 0; s < iSamples; s++)
                {
                    // Sample from CDF of the query - this is the bootleneck of diamond sampling: O(logD)
                    low = lower_bound(vecCDF.begin(), vecCDF.end(), (double)(rand() / RAND_MAX));
                    //printVector(vecCDF);

                    if (low == vecCDF.end())
                        iDim = PARAM_DATA_D - 1;
                    else
                        iDim = low - vecCDF.begin(); // get j'

                    dValue = MATRIX_X(iPointIdx, iDim); //get Xij' - this is the bootleneck of diamond sampling
                    vecCounter[iPointIdx] += iSignQj * iSignXij * sgn(vecQuery(iDim)) * dValue;
                }

                ++iter;
            }
        }

        dSamTime += clock() - dStart;

        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "dDiamondCounter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "dDiamond_Vector_TopK_NoPost_" + int2str(q) + ".txt");

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_TEST_SAVE_OUTPUT)
            saveQueue(minQueTopK, "dDiamond_Vector_TopK_Post_" + int2str(q) + ".txt");
    }


    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Vector-dDiamond: Time is %f \n", getCPUTime(clock() - dStart0));
}


/** \brief Return approximate TopK of MIPS using shiftDiamond (use vector to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - POS_PRESAMPLES, NEG_PRESAMPLES: pre-sampled set for each column
 - MATRIX_X: point set (D x N)
 - MATRIX_Q: query set (Q x D)
 - POS_COL_NORM_1, NEG_COL_NORM_1: vector of sum of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void shift_Diamond_Vector_TopK()
{
    double dStart0 = clock();

    int q, d, s, pointIdx, iNumSamples, iDim;
    double dValue = 0, Qj = 0.0;
    int iSignQj;
    double dStart = 0, dSamTime = 0, dTopBTime = 0, dTopKTime = 0, dBasicSamTime = 0, dStart1;

    DVector vecWeight(PARAM_DATA_D, 0.0); // number of samples for each dimension
    DVector vecCounter(PARAM_DATA_N, 0.0); // counting histogram of N points
    //IVector vecSampleCounter(PARAM_DATA_N, 0); // counting the number of samples used for each points in order to measure the correlation


    VectorXd vecQuery(PARAM_DATA_D); // vector of query
    DVector vecCDF;

    vector<double>::iterator low;

    // Presamples
    IVector vecWedgeSamples, vecTopB;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1
        vecCDF = CDF(vecQuery);

        //---------------------------
        // Compute the sum of N dot products and its dimension weight
        //---------------------------
        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeShiftWeight(vecQuery, vecWeight);

        // Get samples from each dimension, store in a vector for faster sequential access
        fill(vecCounter.begin(), vecCounter.end(), 0.0);
//        fill(vecSampleCounter.begin(), vecSampleCounter.end(), 0.0);

        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            Qj = vecQuery(d);

            if (Qj == 0)
                continue;

            iSignQj = sgn(Qj);

            // Get number of samples for each dimension
            iNumSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);

            getDeterSamples(vecWedgeSamples, d, iNumSamples, iSignQj);
            sort(vecWedgeSamples.begin(), vecWedgeSamples.end());

            // For each samples
            for (s = 0; s < iNumSamples; ++s)
            {
                pointIdx = vecWedgeSamples[s];

                dStart1 = clock();

                // Sample from CDF of the query
                low = lower_bound(vecCDF.begin(), vecCDF.end(), (double)(rand() / RAND_MAX));

                if (low == vecCDF.end())
                    iDim = PARAM_DATA_D - 1;
                else
                    iDim = low - vecCDF.begin(); // get j'


                // Generate two sequential samples then sort them together :-)

                dValue = MATRIX_X(pointIdx, iDim);
                dBasicSamTime += clock() - dStart1;

                if (iSignQj >= 0)
                    vecCounter[pointIdx] += dValue - VECTOR_COL_MIN(d);
                else
                    vecCounter[pointIdx] += VECTOR_COL_MAX(d) - dValue;

//                vecSampleCounter[pointIdx]++;
            }
        }

        dSamTime += clock() - dStart;

        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "shiftDiamondCounter_" + int2str(q) + ".txt");


        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();
        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "shift_Diamond_Vector_TopK_NoPost_" + int2str(q) + ".txt");

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_TEST_SAVE_OUTPUT)
            saveQueue(minQueTopK, "shift_Diamond_Vector_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Basic sampling time is %f \n", getCPUTime(dBasicSamTime));
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Shift Vector-Diamond: Time is %f \n", getCPUTime(clock() - dStart0));
}






#include "WedgeSampling.h"
#include "Utilities.h"
#include "Header.h"

/** \brief Compute the vector weight contains fraction of samples for each dimension
 *
 * \param
 *
 - vecQuery: vector query of size 1 x D
 - C_POS_SUM: norm 1 of each column (1 x D)
 *
 * \return
 *
 - DVector: vector of normalized weight
 - p_dSum: norm 1 of vector weight
 *
 */
void wedge_ColWeight(const VectorXd &vecQuery, DVector &vecWeight)
{
    double dSum = 0;
    int d;

    for (d = 0; d < PARAM_DATA_D; ++d)
    {
        vecWeight[d] = POS_COL_NORM_1[d] * abs(vecQuery(d));
        dSum += vecWeight[d];
    }

    // Normalize weight
    for (d = 0; d < PARAM_DATA_D; ++d)
        vecWeight[d] = vecWeight[d] / dSum;
}

/**
Presorting data for each dimension

Input:
- MATRIX_X: col-wise point set (N x D)
- p_bSign = 1/0: sort based on the absolute value (in dWedge) or exact value (in MIPS-Greedy)

Output:
- vector<IDPair> COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N x D (col-maj)
- DVector POS_COL_NORM_1: column 1-norm for dWedge

**/
void dimensionSort(bool p_bSign)
{
    int d, n;

    double dXij = 0.0;

    VectorXd vecCol(PARAM_DATA_N);

    COL_SORT_DATA_IDPAIR = vector<IDPair>(PARAM_DATA_D * PARAM_DATA_N);

    // Init for precomputed vector
    POS_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);

    // Sort every column based on the value dXij
    vector<IDPair> priVec(PARAM_DATA_N);

    for (d = 0; d < PARAM_DATA_D; ++d)
    {
        vecCol = MATRIX_X.col(d); // N x 1

        // Create an array of Xi/ui
        for (n = 0; n < PARAM_DATA_N; ++n)
        {
            dXij = vecCol(n);

            POS_COL_NORM_1[d] += abs(dXij);

            // True: for dWedge since it uses the |dXij| for sampling
            if (p_bSign)
                priVec[n] = IDPair(sgn(dXij) * n, abs(dXij));
            else // False for Greedy
                priVec[n] = IDPair(n, dXij);
        }

        // Sort X1 > X2 > ... > Xn
        sort(priVec.begin(), priVec.end(), greater<IDPair>());
        // printVector(priVec);

        // Store
        copy(priVec.begin(), priVec.end(), COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N);
    }
}

/** \brief Return approximate TopK using dWedge (using vector to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - vector<IDPair> SORTED_DATA: col-wise matrix with sorted columns of size N x D (col-maj)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1: vector of norm-1 of each dimension
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void dWedge_Vector_TopK()
{
    double dStart0 = clock();
    int iColSamples;

    vector<IDPair>::iterator iter;

    int q, d;
    // double dThreshold = 0.0;

    int iCount, iSamples, iSignQj;

    double dQj = 0.0;
    double dStart = 0.0, dSamTime = 0.0, dTopBTime = 0.0, dTopKTime = 0.0;

    VectorXd vecQuery(PARAM_DATA_D); // vector of query

    IVector vecTopB;
    DVector vecCounter(PARAM_DATA_N, 0.0);
    DVector vecWeight(PARAM_DATA_D, 0.0);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        // reset everything
        fill(vecWeight.begin(), vecWeight.end(), 0.0);

        wedge_ColWeight(vecQuery, vecWeight);

        //---------------------------
        // Access samples and update counter
        //---------------------------

        fill(vecCounter.begin(), vecCounter.end(), 0.0);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0)
                continue;

            iSignQj = sgn(dQj);

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

                // update counter: we need abs(m_iIndex) since we keep the originial sign in the index
                vecCounter[abs((*iter).m_iIndex)] += iSamples * sgn((*iter).m_iIndex) * iSignQj;

                ++iter;
            }
        }

        dSamTime += clock() - dStart;


        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "dWedgeCounter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_TOP_B);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "dWedge_Vector_TopK_NoPost_" + int2str(q) + ".txt");
        */

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            saveQueue(minQueTopK, "dWedge_Vector_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Vector-dWedge: Time is %f \n", getCPUTime(clock() - dStart0));
}

/** \brief Return approximate TopK using dWedge (use unordered_map to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - vector<IDPair> SORTED_DATA: col-wise matrix with sorted columns of size N x D (col-maj)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1: vector of norm-1 of each dimension
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void dWedge_Map_TopK()
{
    double dStart0 = clock();
    int iColSamples, iPointIdx;

    vector<IDPair>::iterator iter;

    int q, d;
    // double dThreshold = 0.0;

    int iCount, iSamples, iSignQj, iSignXij;

    double dQj = 0.0;
    double dStart = 0.0, dSamTime = 0.0, dTopBTime = 0.0, dTopKTime = 0.0;

    VectorXd vecQuery(PARAM_DATA_D); // vector of query

    IVector vecTopB;
    DVector vecWeight(PARAM_DATA_D, 0.0);

    // Will be used when # Samples << # Data points
    unordered_map<int, int> mapCounter;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        // reset everything
        fill(vecWeight.begin(), vecWeight.end(), 0.0);

        wedge_ColWeight(vecQuery, vecWeight);

        //---------------------------
        // Access samples and update counter
        //---------------------------

        mapCounter.clear();
        mapCounter.reserve(PARAM_MIPS_SAMPLES); // For faster

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            iSignQj = sgn(dQj);

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);

            iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N;

            // Reset counting samples
            iCount = 0;
            while (iCount < iColSamples)
            {
                // Get point index with its sign
                iPointIdx = abs((*iter).m_iIndex);
                iSignXij = sgn((*iter).m_iIndex);

                // number of samples
                iSamples = ceil((*iter).m_dValue * iColSamples / POS_COL_NORM_1[d]);

                // Update current # samples
                iCount += iSamples;

                auto mapIter = mapCounter.find(iPointIdx);

                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second += iSamples * iSignXij * iSignQj;
                else // not exist
                    mapCounter.insert(make_pair(iPointIdx, iSamples * iSignXij * iSignQj));


                // Check next iteration
                ++iter;
            }
        }

        dSamTime += clock() - dStart;


        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "dWedgeCounter_" + int2str(q) + ".txt");


        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_TOP_B);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "dWedge_Map_TopK_NoPost_" + int2str(q) + ".txt");
        */

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            saveQueue(minQueTopK, "dWedge_Map_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Map-dWedge: Time is %f \n", getCPUTime(clock() - dStart0));
}

#include "Greedy.h"
#include "Utilities.h"
#include "Header.h"

#include <numeric>      // std::iota

/**
Preprocessing Greedy - Code by Stephan Lorenzen

Input:
- MATRIX_X: point set (N x D)

Output:
- MatrixXi::GREEDY_STRUCT: matrix with sorted columns (data structure used for greedy) of size N x D (col-maj)

**/
//void greedyPreProcessing()
//{
//    int d, n;
//
//    // Create greedy struct: GREEDY_STRUCT of size N x D
//    // For each dimension, we keep track the index of points sorted by its values.
//    GREEDY_STRUCT = MatrixXi(PARAM_DATA_N, PARAM_DATA_D);
//    vector<int> idx(PARAM_DATA_N);
//
//    // Sort every column - store index in GREEDY_STRUCT
//    for(d = 0; d < PARAM_DATA_D; ++d)
//    {
//        // 0..n
//        iota(idx.begin(), idx.end(), 0);
//
//        // sort indexes based on MATRIX_X.row(d), smallest at n - 1, largest at 0
//        sort(idx.begin(), idx.end(), compareMatrixStruct(d));
//
//        // Write to GREEDY_STRUCT
//        for (n = 0; n < PARAM_DATA_N; ++n)
//            GREEDY_STRUCT(n, d) = idx[n]; // Only contain the indexes of point
//    }
//}

/** \brief Return approximate TopK of MIPS for each query. Implements the basic Greedy from their paper
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 - Code by Stephan Lorenzen
 *
 * \param
 *
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (Q x D)
 - GREEDY_STRUCT: matrix with sorted columns (data structure used for greedy) of size N x D (col-maj)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
//void greedyTopK_NoPost()
//{
//    int q, d, s, iSignQj, pIdx;
//    double dValue = 0.0;
//
//    double dStart = 0.0, dSetupTime = 0.0, dTopKTime = 0.0;
//
//    VectorXd vecQuery(PARAM_DATA_D); // vector of query
//    vector<int> vecNextPointIdx(PARAM_DATA_D, 0);
//
//    priority_queue<IDPair, vector<IDPair>, less<IDPair>> candQueue; // Queue used to store candidates.
//
//    unordered_set<int> maxSet; // Set with candidates already added.
//
//    for (q = 0; q < PARAM_QUERY_Q; ++q)
//    {
//        dStart = clock();
//
//        vecQuery = MATRIX_Q.col(q); // size D x 1
//
//        // Clear everything
//        maxSet.clear();
//        fill(vecNextPointIdx.begin(), vecNextPointIdx.end(), 0);
//        candQueue = priority_queue<IDPair, vector<IDPair>, less<IDPair>>();
//
//        // Get pointIdx with largest value for each dimension
//        for (d = 0; d < PARAM_DATA_D; ++d)
//        {
//            // First, set up vecCounter (0 if signQ < 0 else n-1)
//            iSignQj = sgn(vecQuery[d]);
//            if (iSignQj < 0)
//                vecNextPointIdx[d] = PARAM_DATA_N - 1;
//
//            pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Index of point
//            dValue = MATRIX_X(pIdx, d) * vecQuery(d); // Value of point
//
//            candQueue.push(IDPair(d, dValue));      // Add to queue
//        }
//
//        dSetupTime += clock() - dStart;
//
//        // Extract candidates
//        dStart = clock();
//        for (s = 0; s < PARAM_MIPS_SAMPLES && (int)maxSet.size() < PARAM_MIPS_TOP_K; ++s)
//        {
//            // Extract max product
//            d = candQueue.top().m_iIndex;
//
//            // cout << candQueue.top().m_dValue << endl;
//
//            candQueue.pop();
//
//            // Add to result list
//            pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Point index we are interested in
//            maxSet.insert(pIdx);                    // Add to set
//
//            // In-/decrement counter
//            iSignQj = sgn(vecQuery[d]);
//
//            if (iSignQj == 0) // Fix bug when Qd = 0
//                iSignQj = 1;
//
//            vecNextPointIdx[d] += iSignQj;
//
//            // Add next element for this dimension to candQueue if any more left
//            if (vecNextPointIdx[d] >= 0 && vecNextPointIdx[d] < PARAM_DATA_N)
//            {
//                pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Point index of next
//
//                dValue = MATRIX_X(pIdx, d) * vecQuery(d); // Value of next
//
//                candQueue.push(IDPair(d, dValue));      // Add to queue
//            }
//        }
//
//        dTopKTime += clock() - dStart;
//
//        if (PARAM_TEST_SAVE_OUTPUT)
//            saveSet(maxSet, "greedyTopK_NoPost_" + int2str(q) + ".txt");
//    }
//
//    // Print time complexity of each step
//    printf("Setup time is %f \n", getCPUTime(dSetupTime));
//    printf("TopK time is %f \n", getCPUTime(dTopKTime));
//}

/** \brief Return approximate TopK of MIPS for each query. Implements the basic Greedy from their paper
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 - Code by Stephan Lorenzen
 *
 * \param
 *
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (Q x D)
 - GREEDY_STRUCT: matrix with sorted columns (data structure used for greedy) of size N x D (col-maj)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
//void greedyTopK_Post()
//{
//    int q, d, s, signQ, pIdx;
//    double dValue = 0.0;
//    double dStart = 0, dCandTime = 0, dTopKTime = 0;
//
//    VectorXd vecQuery(PARAM_DATA_D); // vector of query
//    vector<int> vecNextPointIdx(PARAM_DATA_D, 0);
//
//    priority_queue<IDPair, vector<IDPair>, less<IDPair>> candQueue; // Queue used to store candidates.
//    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;
//
//    unordered_set<int> candSet; // Set with candidates already added.
//
//    for (q = 0; q < PARAM_QUERY_Q; ++q)
//    {
//        dStart = clock();
//
//        candSet.clear();
//        vecQuery = MATRIX_Q.col(q); // size D x 1
//
//        fill(vecNextPointIdx.begin(), vecNextPointIdx.end(), 0);
//        candQueue = priority_queue<IDPair, vector<IDPair>, less<IDPair>>();
//
//        // Get pointIdx with largest value Xd*Qd for each dimension
//        for (d = 0; d < PARAM_DATA_D; ++d)
//        {
//            // First, set up vecCounter (0 if signQ > 0 else n-1)
//            signQ = sgn(vecQuery(d));
//            if (signQ < 0)
//                vecNextPointIdx[d] = PARAM_DATA_N - 1;
//
//            pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Index of point
//            dValue = MATRIX_X(pIdx, d) * vecQuery(d); // Value of point
//
//            candQueue.push(IDPair(d, dValue));      // Add to queue
//        }
//
//        // Extract candidates
//        for (s = 0; (int)candSet.size() < PARAM_MIPS_DOT_PRODUCTS; ++s) // Will do at most Bk rounds
//        {
//            // Extract max product
//            d = candQueue.top().m_iIndex;
//
//            // cout << candQueue.top().m_dValue << endl;
//
//            candQueue.pop();
//
//            // Add to candidate set
//            pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Point index we are interested in
//            candSet.insert(pIdx);                   // Add to set
//
//            // In-/decrement counter
//            signQ = sgn(vecQuery(d));
//
//            if (signQ == 0) // Fix bug when Qd = 0
//                signQ = 1;
//
//            vecNextPointIdx[d] += signQ;
//
//            // Add next element for this dimension to candQueue if any more left
//            if (vecNextPointIdx[d] >= 0 && vecNextPointIdx[d] < PARAM_DATA_N)
//            {
//                pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Point index of next
//                dValue = MATRIX_X(pIdx, d) * vecQuery(d); // Value of next
//                candQueue.push(IDPair(d, dValue));      // Add to queue
//            }
//        }
//
//        dCandTime += clock() - dStart;
//
//        // Post computation
//        dStart = clock();
//
//        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
//        extract_TopK_MIPS(vecQuery, candSet, PARAM_MIPS_TOP_K, minQueTopK);
//
//        dTopKTime += clock() - dStart;
//
//        if (PARAM_TEST_SAVE_OUTPUT)
//            saveQueue(minQueTopK, "greedyTopK_Post_" + int2str(q) + ".txt");
//    }
//
//    // Print time complexity of each step
//    printf("Time for generating candidate set %f \n", getCPUTime(dCandTime));
//    printf("TopK time is %f \n", getCPUTime(dTopKTime));
//}

/** \brief Return approximate TopK of MIPS for each query. Implements the basic Greedy from the paper NIPS 17
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 - Code by Ninh Pham (strictly follow the paper implementation)
 *
 * \param
 *
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (Q x D)
 - GREEDY_STRUCT: matrix with sorted columns (data structure used for greedy) of size N x D (col-maj)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
//void greedyTopK_Post_NIPS17()
//{
//    double dStart0 = clock();
//
//    int q, d, signQ, pIdx;
//    double dValue = 0.0;
//    double dStart = 0, dCandTime = 0, dTopKTime = 0;
//
//    VectorXd vecQuery(PARAM_DATA_D); // vector of query
//
//    vector<int> vecNextPointIdx(PARAM_DATA_D, 0); // contain the index of the point for the next verification
//    vector<int> candSet; // Set with candidates already added.
//    vector<bool> vecVisited(PARAM_DATA_N); // Set with candidates already added.
//
//    priority_queue<IDPair, vector<IDPair>, less<IDPair>> candQueue; // Queue used to store candidates.
//    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;
//
//    for (q = 0; q < PARAM_QUERY_Q; ++q)
//    {
//        dStart = clock();
//
//        vecQuery = MATRIX_Q.col(q); // size D x 1
//
//        candSet.clear();
//
//        fill(vecNextPointIdx.begin(), vecNextPointIdx.end(), 0);
//        fill(vecVisited.begin(), vecVisited.end(), 0);
//
//        candQueue = priority_queue<IDPair, vector<IDPair>, less<IDPair>>();
//
//        // Get the pointIdx with max value for each dimension
//        for (d = 0; d < PARAM_DATA_D; ++d)
//        {
//            // First, set up vecNextPointIdx (0 if signQ < 0 else n-1)
//            signQ = sgn(vecQuery(d));
//            if (signQ < 0)
//                vecNextPointIdx[d] = PARAM_DATA_N - 1;
//
//            pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Index of point whose value is largest
//            dValue = MATRIX_X(pIdx, d) * vecQuery(d); // Value of point
//
//            candQueue.push(IDPair(d, dValue)); // Add to queue
//        }
//
//        // Extract candidates
//        while ((int)candSet.size() < PARAM_MIPS_DOT_PRODUCTS) // Will do at most Bk rounds
//        {
//            // Extract the dimension d with the max product
//            d = candQueue.top().m_iIndex;
//            candQueue.pop();
//
//            // Get pointIdx and add to candidate set if not visited
//            pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Point index whose value is largest
//
//            // If not visited
//            if (!vecVisited[pIdx])
//            {
//                candSet.push_back(pIdx); // Add to set
//                vecVisited[pIdx] = 1;
//            }
//
//            // In-/decrement counter
//            signQ = sgn(vecQuery(d));
//
//            if (signQ == 0) // Fix bug when Qd = 0
//                signQ = 1;
//
//            while (true)
//            {
//                vecNextPointIdx[d] += signQ; // next index
//
//                // Add next element for this dimension to candQueue if any more left
//                if (vecNextPointIdx[d] >= 0 && vecNextPointIdx[d] < PARAM_DATA_N)
//                {
//                    pIdx = GREEDY_STRUCT(vecNextPointIdx[d], d); // Point index of next
//
//                    if (!vecVisited[pIdx]) // if not exist
//                    {
//                        dValue = MATRIX_X(pIdx, d) * vecQuery(d); // Value of next
//                        candQueue.push(IDPair(d, dValue));      // Add to queue
//
//                        break;
//                    }
//                }
//                else
//                    break;
//            }
//
//        }
//
//        dCandTime += clock() - dStart;
//
//        // Post computation
//        dStart = clock();
//
//        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
//        extract_TopK_MIPS(vecQuery, candSet, PARAM_MIPS_TOP_K, minQueTopK);
//
//        dTopKTime += clock() - dStart;
//
//        if (PARAM_TEST_SAVE_OUTPUT)
//            saveQueue(minQueTopK, "greedyTopK_Post_" + int2str(q) + ".txt");
//    }
//
//    // Print time complexity of each step
//    printf("Time for generating candidate set %f \n", getCPUTime(dCandTime));
//    printf("TopK time is %f \n", getCPUTime(dTopKTime));
//
//    printf("Greedy time is %f \n", getCPUTime(clock() - dStart0));
//}


/** \brief Return approximate TopK of MIPS for each query. Implements the basic Greedy from the paper NIPS 17
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (Q x D)
 - GREEDY_STRUCT: matrix with sorted columns (data structure used for greedy) of size N x D (col-maj)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
void greedy_TopK()
{
    double dStart0 = clock();

    int q, d, iSignQj, pIdx;
    double dValue = 0.0;
    double dStart = 0, dCandTime = 0, dTopKTime = 0;
    IDPair idPair;

    VectorXd vecQuery(PARAM_DATA_D); // vector of query

    vector<int> vecNextPointIdx(PARAM_DATA_D, 0); // contain the index of the point for the next verification
    vector<int> candSet; // Set with candidates already added.
    vector<bool> vecVisited(PARAM_DATA_N); // Set with candidates already added.

    priority_queue<IDPair, vector<IDPair>, less<IDPair>> candQueue; // Queue used to store candidates.
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    vector<IDPair>::iterator iter;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        candSet.clear();

        fill(vecNextPointIdx.begin(), vecNextPointIdx.end(), 0);
        fill(vecVisited.begin(), vecVisited.end(), 0);

        candQueue = priority_queue<IDPair, vector<IDPair>, less<IDPair>>();

        // Get the pointIdx with max value for each dimension
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            // First, set up vecNextPointIdx (0 if signQ < 0 else n-1)
            iSignQj = sgn(vecQuery(d));
            if (iSignQj < 0)
                vecNextPointIdx[d] = PARAM_DATA_N - 1;

            // Get the point index whose value is largest
            iter = COL_SORT_DATA_IDPAIR.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

            dValue = (*iter).m_dValue * vecQuery(d); // Value of point
            candQueue.push(IDPair(d, dValue)); // Add to queue
        }

        // Extract candidates
        while ((int)candSet.size() < PARAM_MIPS_DOT_PRODUCTS) // Will do at most Bd rounds
        {
            // Extract the dimension d with the max product
            d = candQueue.top().m_iIndex;
            candQueue.pop();

            // Get pointIdx and add to candidate set if not visited
            iter = COL_SORT_DATA_IDPAIR.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

            pIdx = (*iter).m_iIndex; // get index

            // If not visited
            if (!vecVisited[pIdx])
            {
                candSet.push_back(pIdx); // Add to set
                vecVisited[pIdx] = 1;
            }

            // In-/decrement counter
            iSignQj = sgn(vecQuery(d));

            //if (iSignQj == 0) // Fix bug when Qd = 0
                //iSignQj = 1;

            while (true)
            {
                vecNextPointIdx[d] += iSignQj; // next index

                // Add next element for this dimension to candQueue if any more left
                if (vecNextPointIdx[d] >= 0 && vecNextPointIdx[d] < PARAM_DATA_N)
                {
                    iter = COL_SORT_DATA_IDPAIR.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

                    pIdx = (*iter).m_iIndex; // Point index of next

                    if (!vecVisited[pIdx]) // if not exist
                    {
                        dValue = (*iter).m_dValue * vecQuery(d); // Value of next
                        candQueue.push(IDPair(d, dValue));      // Add to queue

                        break;
                    }
                }
                else
                    break;
            }

        }

        dCandTime += clock() - dStart;

        // Post computation
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, candSet, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveQueue(minQueTopK, "greedy_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Time for generating candidate set %f \n", getCPUTime(dCandTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Greedy: Time is %f \n", getCPUTime(clock() - dStart0));
}

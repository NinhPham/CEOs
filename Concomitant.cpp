#include "Concomitant.h"
#include "Utilities.h"
#include "Header.h"

/**
Compute matrix-maxtrix multiplication with Gaussian

Input:
- MATRIX_X: col-wise point set (N x D)
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x D_UP)

Output:
- MatrixXd::ROTATED_X: col-wise matrix of size N x D_UP (col-maj)

**/
void rotateData()
{
    // size upD x n at the beginning, after that it would be N x upD
    // PROJECTED_X = MatrixXf::Zero(PARAM_CEOs_D_UP, PARAM_DATA_N);
    PROJECTED_X = MatrixXf::Zero(PARAM_DATA_N, PARAM_CEOs_D_UP); // col-wise N x D_UP

    // Init HD3
    bitHD3Generator(PARAM_INTERNAL_FWHT_PROJECTION * PARAM_CEOs_NUM_ROTATIONS);

    // Fast Hadamard transform
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get data
        VectorXf vecPoint = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION); // PARAM_INTERNAL_FWHT_PROJECTION > D_UP
        vecPoint.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);

        for (int r = 0; r < PARAM_CEOs_NUM_ROTATIONS; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
            {
                vecPoint(d) *= (2 * (int)bitHD3[r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecPoint.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
        }

        // We only get the position from 0 to PARAM_CEOs_D_UP, and ignore the rest
        //PROJECTED_X.col(n) = vecPoint.segment(0, PARAM_CEOs_D_UP);

        // Need to scale D1 * sqrt{2 * log{D2}} to have an unbiased estimate
        // D1 = PARAM_INTERNAL_FWHT_PROJECTION, since all rotations use PARAM_INTERNAL_FWHT_PROJECTION
        // D2 = D_UP since we only consider D_UP position, hence max = 2 * log{D_UP}
        // add into each row. so no need transposeInPlace()
        PROJECTED_X.row(n) = vecPoint.segment(0, PARAM_CEOs_D_UP); // / (PARAM_INTERNAL_FWHT_PROJECTION * sqrt( 2 * log2(PARAM_CEOs_D_UP)));
    }

    // In case TopB = TopK, no need to store MATRIX_X
    if (PARAM_MIPS_TOP_B == PARAM_MIPS_TOP_K)
        MATRIX_X.resize(0, 0);

    // In case we do not have enough RAM for MATRIX_X
//    if (PARAM_INTERNAL_NOT_STORE_MATRIX_X)
//        MATRIX_X.resize(0, 0);


    // Project X must be N x Dup since we will sort and access each col N x 1 many times
    //PROJECTED_X.transposeInPlace();

    // Testing col-wise order
//    cout << PROJECTED_X.col(0).transpose() << endl;
//    cout << "In memory (col-major):" << endl;
//    for (int n = 0; n < 20; n++)
//        cout << *(PROJECTED_X.data() + n) << "  ";
//    cout << endl << endl;
//
//    float fSum = 0.0;
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//        fSum += (*(PROJECTED_X.data() + n) - PROJECTED_X.col(0)(n));
//    cout << fSum << endl;
}

/** \brief Return approximate TopK for each query (using vector to store samples)
 - We compute all N inner products using D_DOWN dimensions based on order statistics
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - PROJECTED_X: Concomitants of Q after Gaussian transformation
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
void sCEOs_Est_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float estTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

//    #pragma omp parallel for reduction(+:estTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecMIPS = VectorXf::Zero(PARAM_DATA_N); // n x 1
        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1

        IVector vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        rotateAndGet_S0_MinMax(vecQuery, vecMinIdx, vecMaxIdx);

        // Compute estimate MIPS for N points
        // Note: PROJECT_X is col-wise and has size N x D_UP

        // TODO: prefetching 2s0 columns of PROJECTED_X would be faster
        for (int d = 0; d < PARAM_CEOs_S0; d++)
        {
//            #ifdef DO_PREFETCH
//            // max part
//            _mm_prefetch(reinterpret_cast<const char*>(PROJECTED_X.data() + PARAM_DATA_N * vecMaxIdx[d]), _MM_HINT_T0);
//            __mm_prefetch (PROJECTED_X.data() + PARAM_DATA_N * vecMaxIdx[d] + 512, 0, 1);
//            // min part
//            _mm_prefetch(reinterpret_cast<const char*>(PROJECTED_X.data() + PARAM_DATA_N * vecMinIdx[d]), _MM_HINT_T0);
//            __mm_prefetch (PROJECTED_X.data() + PARAM_DATA_N * vecMinIdx[d] + 512, 0, 1);
//            #endif

            vecMIPS += PROJECTED_X.col(vecMaxIdx[d]) - PROJECTED_X.col(vecMinIdx[d]);
        }


            // q must be normalized, and not that it does not provide unbiased estimate
//            vecMIPS = vecMIPS + vecProjectedQuery(vecMaxIdx(d)) * PROJECTED_X.col(vecMaxIdx(d))
//                              + vecProjectedQuery(vecMinIdx(d)) * PROJECTED_X.col(vecMinIdx(d));

        // Output estimation to check the estimate
//        vecMIPS = vecMIPS * vecQuery.norm() / (2 * PARAM_CEOs_S0);

//        cout << vecMIPS << endl;

//        if (PARAM_INTERNAL_SAVE_OUTPUT)
//            outputFile(vecMIPS, "CEOs_Est_upD_" + int2str(PARAM_CEOs_D_UP) +
//                      "_s0_" + int2str(PARAM_CEOs_S0) + "_q_" + int2str(q+1) + ".txt");

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        estTime += (float)durTime.count() * 1e-3;

        //----------------------------------------------
        // Find topB & topK together
        startTime = chrono::high_resolution_clock::now();

        // If TopB = TopK, then no need to compute distance since MATRIX_X was deleted
        extract_TopB_TopK_Histogram(vecMIPS, vecQuery, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() * 1e-3;

    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in ms is %f \n", estTime);
    printf("TopK time in ms is %f \n", topKTime);

    printf("sCEOs Est TopK Time in ms is %f \n", (float)durTime.count());

    if (PARAM_INTERNAL_SAVE_OUTPUT)
        outputFile(matTopK, "CEOs_Est_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_s0_" + int2str(PARAM_CEOs_S0) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt");
}

/**

Input:
- MATRIX_X: point set (N x D)

Output:
- MatrixXi::PROJECT_X_SORTED_IDX: matrix with sorted columns (data structure used for greedy) of size upD x N (col-maj)

**/
void build_sCEOs_TA_Index()
{
    rotateData();

    // Create struct: PROJECT_X_SORTED_IDX of size Dup x N
    // For each dimension, we keep track the index of points sorted by its values from large to small
    CEOs_TA_SORTED_IDX = MatrixXi(PARAM_CEOs_D_UP, PARAM_DATA_N);

    #pragma omp parallel for
    for (int d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        vector<IFPair> priVec(PARAM_DATA_N);
        for (int n = 0; n < PARAM_DATA_N; ++n)
            priVec[n] = IFPair(n, PROJECTED_X(n, d));

        // Sort X1 > X2 > ... > Xn
        sort(priVec.begin(), priVec.end(), greater<IFPair>());

        vector<int> idx(PARAM_DATA_N);
    	for (int n = 0; n < PARAM_DATA_N; ++n)
            idx[n] = priVec[n].m_iIndex;

        //printVector(PROJECTED_X.col(d));
        //printVector(idx);

        // Write to GREEDY_STRUCT
        //for (n = 0; n < PARAM_DATA_N; ++n)
            //PROJECT_X_SORTED_IDX(d, n) = idx[n]; // Only contain the indexes of point
        CEOs_TA_SORTED_IDX.row(d) = Map<VectorXi>(idx.data(), PARAM_DATA_N);  // Add into each row, so no need to call transposeInPlace()
    }

    // We have to access each level
    // n = 0, check all maxIdx and minIdx.
    // Hence, D_UP x N is cache-efficient since we iterate each level (e.g. column)
//    CEOs_TA_SORTED_IDX.transposeInPlace();
}

/** \brief Return approximate TopK for each query (using TA algorithm)
 - We check each col of PROJECT_X_SORTED_IDX for the row in TA algorithm
 - Sometime, TA is worse than Est since there are some negative values occured in PROJECT_X
 *
 * \param
 *
 - PROJECTED_X: Concomitants of Q after Gaussian transformation
 - PROJECT_X_SORTED_IDX: Sorted index of Concomitants of Q after Gaussian transformation
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
void sCEOs_TA_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float projectTime = 0.0, topBTime = 0.0, topKTime = 0.0;

    uint64_t iNumAccessedRows = 0.0, iNumProducts = 0.0;
    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    //TODO: If using PROJECT_X with D-UP x N, we can compute estimate faster
    // by prefetching (:, Xi) into cache
//    #pragma omp parallel for reduction(+:projectTime, topBTime, topKTime, iNumAccessedRows, iNumProducts)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1

        IVector vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        rotateAndGet_S0_MinMax(vecQuery, vecMinIdx, vecMaxIdx);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() * 1e-3;

        // Start TA algorithm to find top-B
        startTime = chrono::high_resolution_clock::now();

        // The space overhead of unordered_set is 32 bytes for each item
        // If we do not compute inner product of many points, then using std::unordered_set should be fine
        //unordered_set<int> bitsetHist; // point already estimate
        boost::dynamic_bitset<> bitsetHist(PARAM_DATA_N); // default at 0

        priority_queue<IFPair, vector<IFPair>, greater<IFPair>> minQueTopB;

        // For each row of CEOs_TA_SORTED_IDX of TA algorithm
        for (int n = 0; n < PARAM_DATA_N; ++n) // now n is the row idx
        {
            float fThreshold = 0.0;

            // Compute inner product for all point in max & min dimensions
            for (int s = 0; s < PARAM_CEOs_S0; ++s)
            {
                // Extract s0 max elements
                int iDim = vecMaxIdx[s];
                int iPointIdx = CEOs_TA_SORTED_IDX(iDim, n); // get the index Xi

                // Project X is N x Dup
                fThreshold += PROJECTED_X(iPointIdx, iDim); // Project X is N x Dup

                // Compute inner product estimate if we have not computed it
                //if (bitsetHist.find(iPointIdx) == bitsetHist.end())
                if (~bitsetHist[iPointIdx])
                {
                    bitsetHist[iPointIdx] = 1;
                    //bitsetHist.insert(iPointIdx);

                    // TODO: prefetching a col-wise D-UP x N might make this step faster
                    float fEstProduct = 0.0;
                    for (int d = 0; d < PARAM_CEOs_S0; ++d)
                        fEstProduct += (PROJECTED_X(iPointIdx, vecMaxIdx[d]) - PROJECTED_X(iPointIdx, vecMinIdx[d]));

                    // Insert into minQueue
                    if ((int)minQueTopB.size() < PARAM_MIPS_TOP_B)
                        minQueTopB.push(IFPair(iPointIdx, fEstProduct));
                    else
                    {
                        if (fEstProduct > minQueTopB.top().m_fValue)
                        {
                            minQueTopB.pop();
                            minQueTopB.push(IFPair(iPointIdx, fEstProduct));
                        }
                    }
                }

                // Extract s0 min elements
                iDim = vecMinIdx[s];
                iPointIdx = CEOs_TA_SORTED_IDX(iDim, PARAM_DATA_N - n - 1); // get the index Xi at the end
                fThreshold -= PROJECTED_X(iPointIdx, iDim); // Project X is N x Dup

                //if (bitsetHist.find(iPointIdx) == bitsetHist.end())
                if (~bitsetHist[iPointIdx])
                {
                    // Compute inner product estimate
                    bitsetHist[iPointIdx] = 1;
                    //bitsetHist.insert(iPointIdx);

                    // TODO: prefetching a col-wise D-UP x N might make this step faster
                    float fEstProduct = 0.0;
                    for (int d = 0; d < PARAM_CEOs_S0; ++d)
                        fEstProduct += (PROJECTED_X(iPointIdx, vecMaxIdx[d]) - PROJECTED_X(iPointIdx, vecMinIdx[d]));

                    // Insert into minQueue
                    if ((int)minQueTopB.size() < PARAM_MIPS_TOP_B)
                        minQueTopB.push(IFPair(iPointIdx, fEstProduct));
                    else
                    {
                        if (fEstProduct > minQueTopB.top().m_fValue)
                        {
                            minQueTopB.pop();
                            minQueTopB.push(IFPair(iPointIdx, fEstProduct));
                        }
                    }
                }
            }

            // Finishing a level, then check condition to stop
            if (((int)minQueTopB.size() == PARAM_MIPS_TOP_B) && (minQueTopB.top().m_fValue >= fThreshold))
            {
                iNumAccessedRows += (n + 1);
                iNumProducts += bitsetHist.count(); // bitsetHist.size(); // number of inner product computation for each query
                break;
            }
        }

        VectorXi vecTopB(PARAM_MIPS_TOP_B);
        for (int b = PARAM_MIPS_TOP_B - 1; b >= 0; --b)
        {
            vecTopB(b) = minQueTopB.top().m_iIndex;
            minQueTopB.pop();
        }

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topBTime += (float)durTime.count() * 1e-3;

        // Top-k
        startTime = chrono::high_resolution_clock::now();

        extract_TopK_MIPS(vecQuery, vecTopB, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() * 1e-3;

    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Projection time in ms is %f \n", projectTime);
    printf("TopB time in ms is %f \n", topBTime);
    printf("TopK time in ms is %f \n", topKTime);

    printf("Number of accessing rows per query is %f \n", (float)iNumAccessedRows / PARAM_QUERY_Q);
    printf("Number of inner products per query is %f \n", (float)iNumProducts / PARAM_QUERY_Q);

    printf("sCEOs TA TopK Time in ms is %f \n", (float)durTime.count());

    if (PARAM_INTERNAL_SAVE_OUTPUT)
        outputFile(matTopK, "CEOs_TA_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_s0_" + int2str(PARAM_CEOs_S0) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt");
}


/**
- Compute matrix-maxtrix multiplication with Gaussian
- Truncate PARAM_N_DOWN at the beginning and the end of the order statistics

Output:
- vector<IDPair> coCEOs_MAX_IDX: col-wise matrix with sorted columns of size N_DOWN x D_UP (col-maj)
- vector<IDPair> coCEOs_MIN_IDX: col-wise matrix with sorted columns of size N_DOWN x D_UP (col-maj)

**/

void build_coCEOs_Index()
{
    rotateData();

    // Getting top m (i.e. PARAM_CEOs_N_DOWN) largest and smallest position (including pointIdx and its projection value)
    coCEOs_MAX_IDX = vector<IFPair>(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP);
    coCEOs_MIN_IDX = vector<IFPair>(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP);

    #pragma omp parallel for
    for (int d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        // For each dimension, we need to sort and keep top m
        vector<IFPair> priVec(PARAM_DATA_N);

        for (int n = 0; n < PARAM_DATA_N; ++n)
            priVec[n] = IFPair(n, PROJECTED_X(n, d));

        // Sort X1 > X2 > ... > Xn
        sort(priVec.begin(), priVec.end(), greater<IFPair>());
        // printVector(priVec);

        // Store the beginning to N_DOWN
        copy(priVec.begin(), priVec.begin() + PARAM_CEOs_N_DOWN, coCEOs_MAX_IDX.begin() + d * PARAM_CEOs_N_DOWN);

        // Store the ending to N_DOWN
        copy(priVec.end() - PARAM_CEOs_N_DOWN, priVec.end(), coCEOs_MIN_IDX.begin() + d * PARAM_CEOs_N_DOWN);
    }

    //cout << "Finish projection !!!" << endl;
    //system("PAUSE");

    // vector might generate more space to store data (perhaps not need for a pre-defined length)
    coCEOs_MAX_IDX.shrink_to_fit();
    coCEOs_MIN_IDX.shrink_to_fit();

    // Clear projection
    PROJECTED_X.resize(0, 0);

    double dSize = 1.0 * sizeof(coCEOs_MAX_IDX)  / (1 << 30) +
                   1.0 * sizeof(coCEOs_MAX_IDX[0]) * coCEOs_MAX_IDX.size() / (1 << 30) ; // capacity() ?

    cout << "Size of coCEOs index in GB: " << 2 * dSize << endl;

}


/** \brief Return approximate TopK for each query (using map to store samples)
 - Estimating the inner product in the histogram of N_DOWN x D_DOWN based on order statistics (using value)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - ROTATED_X: Concomitants of Q after Gaussian transformation
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

 // TODO: Replace by Misra-Gries or SpaceSaving for more cache-efficient
void coCEOs_Map_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float projectTime = 0.0, topKTime = 0.0;

    float fMapSize = 0.0;
    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

//    #pragma omp parallel for  reduction(+:projectTime, topKTime, fMapSize)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1

        IVector vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        rotateAndGet_S0_MinMax(vecQuery, vecMinIdx, vecMaxIdx);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() * 1e-3;

        // Partially estimating inner product
        startTime = chrono::high_resolution_clock::now();

        unordered_map<int, float> mapCounter; // counting histogram of N points
        mapCounter.reserve(10 * PARAM_CEOs_S0 * PARAM_CEOs_N_DOWN);

        // TODO: prefetching coCEOs_MIN_IDX and coCEOs_MAX_IDX
        // Estimtate MIPS using the minIdx, take negative
        for (const auto& minIdx: vecMinIdx)
        {
        // TODO:
//            //    #pragma omp parallel for  reduction(+:projectTime, topKTime, fMapSize)
//__builtin_prefetch (coCEOs_MIN_IDX.data() + minIdx * PARAM_CEOs_N_DOWN, 0, 1);
//            _mm_prefetch(reinterpret_cast<const char*>(coCEOs_MIN_IDX.data() + minIdx * PARAM_CEOs_N_DOWN), _MM_HINT_T0);

            vector<IFPair>::iterator iterBegin = coCEOs_MIN_IDX.begin() + minIdx * PARAM_CEOs_N_DOWN;

            for (vector<IFPair>::iterator iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
            {
                // cout << *iter << endl;
                //int iPointIdx = (*iter).m_iIndex;
                //float fValue = (*iter).m_fValue;

                auto mapIter = mapCounter.find((*iter).m_iIndex);

                // We might need to multiply to projected query since we use larger s0
                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second -= (*iter).m_fValue; // update partial inner product
                    // Since both (*iter).m_fValue and vecProjectedQuery(minIdx) are negative, you can use <q, r_1> * <x, r_1> as an estimate
                    // mapIter->second += (*iter).m_fValue * vecProjectedQuery(minIdx);
                else // not exist
                    mapCounter.insert(make_pair((*iter).m_iIndex, -(*iter).m_fValue)); // (iPointIdx, -fValue)
//                    mapCounter.insert(make_pair((*iter).m_iIndex,
//                                                (*iter).m_fValue * vecProjectedQuery(minIdx)));
            }
        }

        // Estimate MIPS using the maxIdx, take positive
        for (const auto& maxIdx: vecMaxIdx)
        {
        // TODO:
//            __builtin_prefetch (coCEOs_MAX_IDX.data() + maxIdx * PARAM_CEOs_N_DOWN, 0, 1);
//            _mm_prefetch(reinterpret_cast<const char*>(coCEOs_MAX_IDX.data() + maxIdx * PARAM_CEOs_N_DOWN), _MM_HINT_T0);

            vector<IFPair>::iterator iterBegin = coCEOs_MAX_IDX.begin() + maxIdx * PARAM_CEOs_N_DOWN;

            for (vector<IFPair>::iterator iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
            {
                // cout << *iter << endl;
                //int iPointIdx = (*iter).m_iIndex;
                //float fValue = (*iter).m_fValue;

                auto mapIter = mapCounter.find((*iter).m_iIndex);

                // We might need to multiply to projected query if we use larger s0
                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second += (*iter).m_fValue;
//                    mapIter->second += (*iter).m_fValue * vecProjectedQuery(maxIdx);
                else // not exist
                    mapCounter.insert(make_pair((*iter).m_iIndex, (*iter).m_fValue));
//                    mapCounter.insert(make_pair((*iter).m_iIndex,
//                                                (*iter).m_fValue * vecProjectedQuery(maxIdx)));
            }
        }

        fMapSize += 1.0 * mapCounter.size() / PARAM_QUERY_Q;

        // If TopB = TopK, no need to compute distance, we also deleted MATRIX_X after building the index
        extract_TopB_TopK_Histogram(mapCounter, vecQuery, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() *1e-3;

    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Project time in ms is %f \n", projectTime);
    printf("TopK time in ms is %f \n", topKTime);
    printf("Avg mapCounter Size is %f \n", fMapSize);

    printf("coCEOs-Map TopK Time in ms is %f \n", (float)durTime.count());

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "coCEOs_Map_downN_" + int2str(PARAM_CEOs_N_DOWN) + "_upD_" + int2str(PARAM_CEOs_D_UP)
                    + "_s0_" + int2str(PARAM_CEOs_S0) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";

        outputFile(matTopK, sFileName);
    }
}

/** \brief Return approximate TopK for each query (using map to store samples)
 - Estimating the inner product in the histogram of N_DOWN x D_DOWN based on order statistics (using value)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - ROTATED_X: Concomitants of Q after Gaussian transformation
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void coCEOs_Vector_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float projectTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

//    #pragma omp parallel for reduction(+:projectTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1

        IVector vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        rotateAndGet_S0_MinMax(vecQuery, vecMinIdx, vecMaxIdx);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() * 1e-3;

        // Partially estimating inner product
        startTime = chrono::high_resolution_clock::now();

        VectorXf vecCounter = VectorXf::Zero(PARAM_DATA_N); // counting histogram of N points

        //TODO: prefetching coCEOs_MIN_IDX and coCEOs_MAX_IDX

        // Estimtate MIPS using the minIdx, take negative
        for (const auto& minIdx : vecMinIdx)
        {
            vector<IFPair>::iterator iterBegin = coCEOs_MIN_IDX.begin() + minIdx * PARAM_CEOs_N_DOWN;

            for (vector<IFPair>::iterator iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
                vecCounter((*iter).m_iIndex) -= (*iter).m_fValue;
        }

        // Estimate MIPS using the maxIdx, take positive
        for (const auto& maxIdx : vecMaxIdx)
        {
            vector<IFPair>::iterator iterBegin = coCEOs_MAX_IDX.begin() + maxIdx * PARAM_CEOs_N_DOWN;

            for (vector<IFPair>::iterator iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
                vecCounter((*iter).m_iIndex) += (*iter).m_fValue;
        }

        extract_TopB_TopK_Histogram(vecCounter, vecQuery, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() * 1e-3;

    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in ms is %f \n", projectTime);
    printf("TopK time in ms is %f \n", topKTime);

    printf("coCEOs-Vector TopK Time in ms is %f \n", (float)durTime.count());

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "coCEOs_Vector_downN_" + int2str(PARAM_CEOs_N_DOWN) + "_upD_" + int2str(PARAM_CEOs_D_UP)
                    + "_s0_" + int2str(PARAM_CEOs_S0) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";

        outputFile(matTopK, sFileName);
    }
}

/**
- For each dimension, store only top-B largest values

Output:
- MATRIX_1CEOs of size TopB x D_UP containing index of top B for each dimension

**/
void build_1CEOs_Index()
{
    rotateData();

    MATRIX_1CEOs = MatrixXi::Zero(PARAM_MIPS_TOP_B, PARAM_CEOs_D_UP);

    // Only need max
    // TODO: get max absolute
    #pragma omp parallel for
    for (int d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        extract_max_TopB(PROJECTED_X.col(d), MATRIX_1CEOs.col(d));
    }

    // Clear projection
    PROJECTED_X.resize(0, 0);

}

/** \brief Return approximate TopK for each query
 - Only use the maximum Q(1)
 *
 * \param
 *
 - MATRIX_1CEOs of size Top-B x Dup: Top-B indexes of each dimension
 *
 * \return
 - Top K MIPS
 *
 */

void maxCEOs_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float projectTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

//    #pragma omp parallel for reduction(+:projectTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1

        int iMaxDim;

        // Note that if you want to keep abs() value, then the indexing must be change since we currently only keep topB max points
        rotateAndGetMax(vecQuery, iMaxDim);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() * 1e-3;

        // Find top-K
        startTime = chrono::high_resolution_clock::now();

        extract_TopK_MIPS(vecQuery, MATRIX_1CEOs.col(iMaxDim), matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() * 1e-3;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in ms is %f \n", projectTime);
    printf("TopK time in ms is %f \n", topKTime);

    printf("1CEOs TopK Time in ms is %f \n", (float)durTime.count());

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "1CEOs_upD_" + int2str(PARAM_CEOs_D_UP) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";
        outputFile(matTopK, sFileName);
    }
}

/**
- Compute matrix-maxtrix multiplication with Gaussian
- Store only B largest values of each UP_D dimensions

Output:
- vector<int> MAX_COL_SORT_IDX: index of top B for each dimension

**/
void build_2CEOs_Index()
{
    rotateData();

    // We do not use all Dup x Dup since there is no case maxIdx = minIdx
    // We just keep the case d1 = d2 as 0 for simplicity of implementation
    MATRIX_2CEOs = MatrixXi::Zero(PARAM_MIPS_TOP_B, PARAM_CEOs_D_UP * PARAM_CEOs_D_UP);

    #pragma omp parallel for
    for (int d1 = 0; d1 < PARAM_CEOs_D_UP; ++d1)
    {
        // Consider it as max
        VectorXf vecMaxCol = PROJECTED_X.col(d1);

        // Consider the rest as min
        for (int d2 = 0; d2 < PARAM_CEOs_D_UP; ++d2)
        {
            if (d2 == d1)
                continue;

            VectorXf vecMinCol = PROJECTED_X.col(d2); // N x 1

            VectorXf vecDif = vecMaxCol - vecMinCol;

            extract_max_TopB(vecDif, MATRIX_2CEOs.col(d1 * PARAM_CEOs_D_UP + d2));
        }
    }

    // Clear projection
    PROJECTED_X.resize(0, 0);

}

/** \brief Return approximate TopK for each query
 - Only use the maximum Q(1) and minimum Q(D)
 *
 * \param
 *
 - MATRIX_2CEOs: Top-B indexes of each dimension
 *
 * \return
 - Top K MIPS
 *
 */

void minmaxCEOs_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float projectTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

//    #pragma omp parallel for reduction(+:projectTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1

        int iMinDim, iMaxDim;
        rotateAndGetMinMax(vecQuery, iMinDim, iMaxDim);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() * 1e-3;

        // Find topK
        startTime = chrono::high_resolution_clock::now();

        extract_TopK_MIPS(vecQuery, MATRIX_2CEOs.col(iMaxDim * PARAM_CEOs_D_UP + iMinDim), matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() * 1e-3;

    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in ms is %f \n", projectTime);
    printf("TopK time in ms is %f \n", topKTime);

    printf("2CEOs TopK Time in ms is %f \n", (float)durTime.count());

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "2CEOs_upD_" + int2str(PARAM_CEOs_D_UP) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";
        outputFile(matTopK, sFileName);
    }
}


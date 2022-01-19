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
    // TODO: write a function to set internal param
    PARAM_INTERNAL_LOG2_CEOs_D_UP = log2(PARAM_CEOs_D_UP);

    // size upD x n at the beginning, after that it would be N x upD
    PROJECTED_X = MatrixXf::Zero(PARAM_CEOs_D_UP, PARAM_DATA_N);

    if (PARAM_CEOs_NUM_ROTATIONS)
    {
        // Init HD3
        HD3Generator(PARAM_CEOs_D_UP);

        // Fast Hadamard transform
        #pragma omp parallel for
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            // Get data
            VectorXf vecPoint = VectorXf::Zero(PARAM_CEOs_D_UP);
            vecPoint.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);

            // Rotate 1 time for dense data, 2 or 3 times for sparse data
            for (int i = 0; i < PARAM_CEOs_NUM_ROTATIONS; ++i)
            {
                vecPoint = vecPoint.cwiseProduct(HD3.col(i).cast<float>());
                fht_float(vecPoint.data(), PARAM_INTERNAL_LOG2_CEOs_D_UP);
                //FWHT(vecPoint);
            }

            PROJECTED_X.col(n) = vecPoint;
        }
    }
    else
    {
        // Generate normal distribution
        gaussGenerator(PARAM_CEOs_D_UP, PARAM_DATA_D);
        PROJECTED_X = MATRIX_G * MATRIX_X; // (Dup x D) * (D x N)
    }

    // In case we do not have enough RAM for MATRIX_X
    if (PARAM_INTERNAL_NOT_STORE_MATRIX_X && PARAM_CEOs_NUM_ROTATIONS)
        MATRIX_X.resize(0, 0);

    // Project X must be N x Dup since we will sort and access each col N x 1 many times
    PROJECTED_X.transposeInPlace();

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

    //#pragma omp parallel for reduction(+:estTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecMIPS = VectorXf::Zero(PARAM_DATA_N); // n x 1

        // Rotate query or Gaussian projection
        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1
        VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_CEOs_D_UP);

        if (PARAM_CEOs_NUM_ROTATIONS > 0)
        {
            vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;
            for (int i = 0; i < PARAM_CEOs_NUM_ROTATIONS; ++i)
            {
                vecProjectedQuery = vecProjectedQuery.cwiseProduct(HD3.col(i).cast<float>());
                fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_CEOs_D_UP);
                //FWHT(vecProjectedQuery);
            }
        }
        else
            vecProjectedQuery = MATRIX_G * vecQuery;

        // Get the minimum and maximum indexes
        VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        extract_TopK_MinMax_Idx(vecProjectedQuery, PARAM_CEOs_S0, vecMinIdx, vecMaxIdx);

        // Compute estimate MIPS for N points
        // Note: PROJECT_X is col-wise and has size N x D_UP
//        cout << "Num col: " << PROJECTED_X.cols() << ", num rows: " << PROJECTED_X.rows() << endl;

        for (int d = 0; d < PARAM_CEOs_S0; d++)
            vecMIPS = vecMIPS + PROJECTED_X.col(vecMaxIdx(d)) - PROJECTED_X.col(vecMinIdx(d));
//            vecMIPS = vecMIPS + vecProjectedQuery(vecMaxIdx(d)) * PROJECTED_X.col(vecMaxIdx(d))
//                              + vecProjectedQuery(vecMinIdx(d)) * PROJECTED_X.col(vecMinIdx(d));

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        estTime += (float)durTime.count() / 1000000;

        //----------------------------------------------
        // Find topB & topK together
        startTime = chrono::high_resolution_clock::now();

        extract_TopB_TopK_Histogram(vecMIPS, vecQuery, PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;
    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in second is %f \n", estTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("sCEOs Est TopK Time in second is %f \n", (float)durTime.count() / 1000000);

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

    // Create struct: PROJECT_X_SORTED_IDX of size N x Dup
    // For each dimension, we keep track the index of points sorted by its values from large to small
    CEOs_TA_SORTED_IDX = MatrixXi(PARAM_DATA_N, PARAM_CEOs_D_UP);

    #pragma omp parallel for
    for (int d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        vector<int> idx(PARAM_DATA_N);

        // 0..n
        iota(idx.begin(), idx.end(), 0);

        // sort indexes based on PROJECT_X.col(d), smallest at n - 1, largest at 0
        sort(idx.begin(), idx.end(), compare_PROJECTED_X(d));

        //printVector(PROJECTED_X.col(d));
        //printVector(idx);

        // Write to GREEDY_STRUCT
        //for (n = 0; n < PARAM_DATA_N; ++n)
            //PROJECT_X_SORTED_IDX(d, n) = idx[n]; // Only contain the indexes of point
        CEOs_TA_SORTED_IDX.col(d) = Map<VectorXi>(idx.data(), PARAM_DATA_N);
    }

    // We have to access each level
    // n = 0, check all maxIdx and minIdx.
    // Hence, D_UP x N is cache-efficient since we iterate each level (e.g. column)
    CEOs_TA_SORTED_IDX.transposeInPlace();
}

/** \brief Return approximate TopK for each query (using TA algorithm)
 - We check each col of PROJECT_X_SORTED_IDX for the row in TA algorithm
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

    //#pragma omp parallel for reduction(+:projectTime, topBTime, topKTime, iNumAccessedRows, iNumProducts)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1
        VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_CEOs_D_UP);

        if (PARAM_CEOs_NUM_ROTATIONS > 0)
        {
            vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;
            for (int i = 0; i < PARAM_CEOs_NUM_ROTATIONS; ++i)
            {
                vecProjectedQuery = vecProjectedQuery.cwiseProduct(HD3.col(i).cast<float>());
                fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_CEOs_D_UP);
                //FWHT(vecProjectedQuery);
            }
        }
        else
            vecProjectedQuery = MATRIX_G * vecQuery;

        // Get the minimum and maximum indexes
        VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        extract_TopK_MinMax_Idx(vecProjectedQuery, PARAM_CEOs_S0, vecMinIdx, vecMaxIdx);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() / 1000000;

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
                int iDim = vecMaxIdx(s);
                int iPointIdx = CEOs_TA_SORTED_IDX(iDim, n); // get the index Xi

                // Project X is N x Dup
                fThreshold += PROJECTED_X(iPointIdx, iDim); // Project X is N x Dup

                // Compute inner product estimate if we have not computed it
                //if (bitsetHist.find(iPointIdx) == bitsetHist.end())
                if (~bitsetHist[iPointIdx])
                {
                    bitsetHist[iPointIdx] = 1;
                    //bitsetHist.insert(iPointIdx);

                    float fEstProduct = 0.0;
                    for (int d = 0; d < PARAM_CEOs_S0; ++d)
                        fEstProduct += (PROJECTED_X(iPointIdx, vecMaxIdx(d)) - PROJECTED_X(iPointIdx, vecMinIdx(d)));

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
                iDim = vecMinIdx(s);
                iPointIdx = CEOs_TA_SORTED_IDX(iDim, PARAM_DATA_N - n - 1); // get the index Xi at the end
                fThreshold -= PROJECTED_X(iPointIdx, iDim); // Project X is N x Dup

                //if (bitsetHist.find(iPointIdx) == bitsetHist.end())
                if (~bitsetHist[iPointIdx])
                {
                    // Compute inner product estimate
                    bitsetHist[iPointIdx] = 1;
                    //bitsetHist.insert(iPointIdx);

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
        topBTime += (float)durTime.count() / 1000000;

        // Top-k
        startTime = chrono::high_resolution_clock::now();

        if (PARAM_INTERNAL_NOT_STORE_MATRIX_X && PARAM_CEOs_NUM_ROTATIONS)
            extract_TopK_MIPS(vecProjectedQuery, vecTopB, PARAM_MIPS_TOP_K, matTopK.col(q));
        else
            extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Projection time in second is %f \n", projectTime);
    printf("TopB time in second is %f \n", topBTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("Number of accessing rows per query is %f \n", (float)iNumAccessedRows / PARAM_QUERY_Q);
    printf("Number of inner products per query is %f \n", (float)iNumProducts / PARAM_QUERY_Q);

    printf("sCEOs TA TopK Time in second is %f \n", (float)durTime.count() / 1000000);

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

    // Clear projection
    PROJECTED_X.resize(0, 0);

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

void coCEOs_Map_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float projectTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    //#pragma omp parallel for  reduction(+:projectTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1
        VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_CEOs_D_UP);

        if (PARAM_CEOs_NUM_ROTATIONS > 0)
        {
            vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;
            for (int i = 0; i < PARAM_CEOs_NUM_ROTATIONS; ++i)
            {
                vecProjectedQuery = vecProjectedQuery.cwiseProduct(HD3.col(i).cast<float>());
                fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_CEOs_D_UP);
                //FWHT(vecProjectedQuery);
            }
        }
        else
            vecProjectedQuery = MATRIX_G * vecQuery;

        // Get the minimum and maximum indexes
        VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        extract_TopK_MinMax_Idx(vecProjectedQuery, PARAM_CEOs_S0, vecMinIdx, vecMaxIdx);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() / 1000000;

        // Partially estimating inner product
        startTime = chrono::high_resolution_clock::now();

        unordered_map<int, float> mapCounter; // counting histogram of N points
        mapCounter.reserve(10 * PARAM_MIPS_NUM_SAMPLES);

        // Esimtate MIPS using the minIdx, take negative
        for (const auto& minIdx: vecMinIdx)
        {
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
                    // both (*iter).m_fValue and vecProjectedQuery(minIdx) are negative
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

        extract_TopB_TopK_Histogram(mapCounter, vecQuery, PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Project time in second is %f \n", projectTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("coCEOs-Map TopK Time in second is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "coCEOs_Map_S_" + int2str(PARAM_MIPS_NUM_SAMPLES) + "_upD_" + int2str(PARAM_CEOs_D_UP)
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

    //#pragma omp parallel for reduction(+:projectTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1
        VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_CEOs_D_UP);

        if (PARAM_CEOs_NUM_ROTATIONS > 0)
        {
            vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;
            for (int i = 0; i < PARAM_CEOs_NUM_ROTATIONS; ++i)
            {
                vecProjectedQuery = vecProjectedQuery.cwiseProduct(HD3.col(i).cast<float>());
                fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_CEOs_D_UP);
                //FWHT(vecProjectedQuery);
            }
        }
        else
            vecProjectedQuery = MATRIX_G * vecQuery;

        // Get the minimum and maximum indexes
        VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
        extract_TopK_MinMax_Idx(vecProjectedQuery, PARAM_CEOs_S0, vecMinIdx, vecMaxIdx);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() / 1000000;

        // Partially estimating inner product
        startTime = chrono::high_resolution_clock::now();

        VectorXf vecCounter = VectorXf::Zero(PARAM_DATA_N); // counting histogram of N points

        // Esimtate MIPS using the minIdx, take negative
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

        extract_TopB_TopK_Histogram(vecCounter, vecQuery, PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in second is %f \n", projectTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("coCEOs-Vector TopK Time in second is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "coCEOs_Vector_S_" + int2str(PARAM_MIPS_NUM_SAMPLES) + "_upD_" + int2str(PARAM_CEOs_D_UP)
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
    for (int d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        extract_max_TopK(PROJECTED_X.col(d), PARAM_MIPS_TOP_B, MATRIX_1CEOs.col(d));
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

    //#pragma omp parallel for reduction(+:projectTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1
        VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_CEOs_D_UP);

        if (PARAM_CEOs_NUM_ROTATIONS > 0)
        {
            vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;
            for (int i = 0; i < PARAM_CEOs_NUM_ROTATIONS; ++i)
            {
                vecProjectedQuery = vecProjectedQuery.cwiseProduct(HD3.col(i).cast<float>());
                fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_CEOs_D_UP);
                //FWHT(vecProjectedQuery);
            }
        }
        else
            vecProjectedQuery = MATRIX_G * vecQuery;

        // Find maximum index
        VectorXf::Index maxIndex;
        vecProjectedQuery.maxCoeff(&maxIndex);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() / 1000000;


        // Find top-K
        startTime = chrono::high_resolution_clock::now();
        extract_TopK_MIPS(vecQuery, MATRIX_1CEOs.col(maxIndex), PARAM_MIPS_TOP_K, matTopK.col(q));
        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;
    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in second is %f \n", projectTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("1CEOs TopK Time in second is %f \n", (float)durTime.count() / 1000000);

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

            extract_max_TopK(vecDif, PARAM_MIPS_TOP_B, MATRIX_2CEOs.col(d1 * PARAM_CEOs_D_UP + d2));
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

    //#pragma omp parallel for reduction(+:projectTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1
        VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_CEOs_D_UP);

        if (PARAM_CEOs_NUM_ROTATIONS > 0)
        {
            vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;
            for (int i = 0; i < PARAM_CEOs_NUM_ROTATIONS; ++i)
            {
                vecProjectedQuery = vecProjectedQuery.cwiseProduct(HD3.col(i).cast<float>());
                fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_CEOs_D_UP);
                //FWHT(vecProjectedQuery);
            }
        }
        else
            vecProjectedQuery = MATRIX_G * vecQuery;

        // Find maximum index
        VectorXf::Index maxIndex, minIndex;
        vecProjectedQuery.maxCoeff(&maxIndex);
        vecProjectedQuery.minCoeff(&minIndex);

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        projectTime += (float)durTime.count() / 1000000;

        // Find topK
        startTime = chrono::high_resolution_clock::now();
        extract_TopK_MIPS(vecQuery, MATRIX_2CEOs.col(maxIndex * PARAM_CEOs_D_UP + minIndex), PARAM_MIPS_TOP_K, matTopK.col(q));
        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in second is %f \n", projectTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("2CEOs TopK Time in second is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "2CEOs_upD_" + int2str(PARAM_CEOs_D_UP) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";
        outputFile(matTopK, sFileName);
    }
}

/**
Compute matrix-maxtrix multiplication with Gaussian

Input:
- MATRIX_X: col-wise point set (N x D)
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x PARAM_CEOs_D_DOWN)

Output:
- MatrixXd::ROTATED_X: col-wise matrix of size N x PARAM_CEOs_D_DOWN (col-maj)

**/
void GaussianRP()
{
    gaussGenerator(PARAM_RP_D_DOWN, PARAM_DATA_D);
    PROJECTED_X = MATRIX_G * MATRIX_X; // (Ddown x D) * (D x N) = Ddown x N
}

/** \brief Return approximate TopK for each query (using vector to store samples)
 - We compute all N inner products using D_DOWN dimensions based on random projection
 *
 * \param
 *
 - PROJECTED_X: Gaussian transformation
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K MIPS
 *
 */
void RP_Est_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float estTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    #pragma omp parallel for reduction(+:estTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Rotate query or Gaussian projection
        VectorXf vecQuery = MATRIX_Q.col(q); // d x 1
        VectorXf vecProjectedQuery = MATRIX_G * vecQuery; // Ddown x 1

        // PROJECT_X is Ddown x N
        VectorXf vecMIPS = vecProjectedQuery.transpose() * PROJECTED_X;

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        estTime += (float)durTime.count() / 1000000;

        //----------------------------------------------
        // Find topB & topK together
        startTime = chrono::high_resolution_clock::now();

        extract_TopB_TopK_Histogram(vecMIPS, vecQuery, PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;
    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in second is %f \n", estTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("Conventional RP-Est TopK Time in second is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string sFileName = "RP_Est_downD_" + int2str(PARAM_RP_D_DOWN) +
                      + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";
        outputFile(matTopK, sFileName);
    }
}

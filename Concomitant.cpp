<<<<<<< HEAD
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
=======
#include "Concomitant.h"
#include "Utilities.h"
#include "Header.h"

/**
Compute vector-maxtrix multiplication with Gaussian

Input:
- VectorXd:vecQuery: col-wise D x 1
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x D_UP)

Output:
- VectorXd::Projected_Qj: col-wise matrix of size upD x 1 (if HD3) or 1 x upD (if random projection)

**/
void GaussianRP_Query(Ref<VectorXd> p_vecProjectedQuery, const VectorXd &p_vecQuery)
{
    p_vecProjectedQuery =  VectorXd::Zero(PARAM_CEOs_D_UP); // size upD x 1

    if (PARAM_INTERNAL_GAUSS_HD3)
    {
        // Get data
        p_vecProjectedQuery.segment(0, PARAM_DATA_D) = p_vecQuery;

        //printVector(PROJECTED_X.col(n));
        FWHT(p_vecProjectedQuery, HD1);

        //printVector(PROJECTED_X.col(n));
        FWHT(p_vecProjectedQuery, HD2);

        //printVector(PROJECTED_X.col(n));
        FWHT(p_vecProjectedQuery, HD3);
        // Add additional zero components
    }
    else
        p_vecProjectedQuery = p_vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x upD
}

/**
Compute matrix-maxtrix multiplication with Gaussian

Input:
- MATRIX_X: col-wise point set (N x D)
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x D_UP)

Output:
- MatrixXd::ROTATED_X: col-wise matrix of size N x D_UP (col-maj)

**/
void GaussianRP_Data()
{
    if (PARAM_INTERNAL_GAUSS_HD3)
    {
        // Init HD3
        HD3Generator(PARAM_CEOs_D_UP);
        //printVector(HD1);
        //printVector(HD2);
        //printVector(HD3);

        // size upD x n at the beginning, after that it would be N x upD
        PROJECTED_X =  MatrixXd::Zero(PARAM_CEOs_D_UP, PARAM_DATA_N);

        // Fast Hadamard transform
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            // Get data
            PROJECTED_X.col(n).segment(0, PARAM_DATA_D) = MATRIX_X.row(n);

            //printVector(PROJECTED_X.col(n));
            FWHT(PROJECTED_X.col(n), HD1);

            //printVector(PROJECTED_X.col(n));
            FWHT(PROJECTED_X.col(n), HD2);

            //printVector(PROJECTED_X.col(n));
            FWHT(PROJECTED_X.col(n), HD3);
            // Add additional zero components
        }

        // cout << "X' has rows " << PROJECTED_X.rows() << " and cols " << PROJECTED_X.cols() << endl;

        PROJECTED_X.transposeInPlace();

        // cout << "X' has rows " << PROJECTED_X.rows() << " and cols " << PROJECTED_X.cols() << endl;
    }
    else
    {
        PROJECTED_X =  MatrixXd::Zero(PARAM_DATA_N, PARAM_CEOs_D_UP); // size upD x n at the beginning, after that it would be N x upD

        // Generate normal distribution
        gaussGenerator(PARAM_DATA_D, PARAM_CEOs_D_UP);

        // MATRIX_X is col-major N x d; MATRIX_NORMAL_DISTRIBUTION is col major d x D
        PROJECTED_X = MATRIX_X * MATRIX_NORMAL_DISTRIBUTION; // (N x D)  * (D x upD)
    }
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
    double dStart0 = clock();
    double dStart = 0.0, dProjectionTime = 0.0, dTopBTime = 0.0, dTopKTime = 0.0;

    int q, d;

    VectorXd vecQuery(PARAM_DATA_D);
    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP);
    VectorXd vecMIPS(PARAM_DATA_N);

    IVector vecTopB;
    VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Reset ever
        vecMIPS = VectorXd::Zero(PARAM_DATA_N);

        // No need to normalize the query does not change the result
        // vecQuery = MATRIX_Q.col(q).normalized(); // d x 1
        vecQuery = MATRIX_Q.col(q); // d x 1

        // Rotate query
        GaussianRP_Query(vecProjectedQuery, vecQuery); // 1 x upD
        //vecProjectedQuery = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x D

        // Get the minimum and maximum indexes
        vecMinIdx = extract_TopK_MinIdx(vecProjectedQuery, PARAM_CEOs_S0);
        vecMaxIdx = extract_TopK_MaxIdx(vecProjectedQuery, PARAM_CEOs_S0);

        // Compute estimate MIPS for N points
        for (d = 0; d < PARAM_CEOs_S0; d++)
            vecMIPS = vecMIPS + PROJECTED_X.col(vecMaxIdx(d)) - PROJECTED_X.col(vecMinIdx(d));

        dProjectionTime += clock() - dStart;

        //----------------------------------------------
        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecMIPS, PARAM_MIPS_TOP_B);

        // Store no post-process result
        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "sCEOs_Est_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_downD_" + int2str(PARAM_CEOs_S0) + "_TopK_NoPost_" + int2str(q) + ".txt");
        */

        dTopBTime += clock() - dStart;

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------

        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()

        if (PARAM_INTERNAL_GAUSS_HD3 && PARAM_INTERNAL_NOT_STORE_MATRIX_X)
            extract_TopK_MIPS_Projected_X(vecProjectedQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK); // upD x 1
        else
            extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        //printf("Computing TopK: Time is %f \n", getCPUTime(dTopKTime));

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveQueue(minQueTopK, "sCEOs_Est_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_downD_" + int2str(PARAM_CEOs_S0) + "_TopK_Post_" + int2str(q) + ".txt");

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("sCEOs Est TopK: Time is %f \n", getCPUTime(clock() - dStart0));
}

/**

Input:
- MATRIX_X: point set (N x D)

Output:
- MatrixXi::PROJECT_X_SORTED_IDX: matrix with sorted columns (data structure used for greedy) of size upD x N (col-maj)

**/
void build_sCEOs_TA()
{
    int d;

    GaussianRP_Data();

    // Create struct: PROJECT_X_SORTED_IDX of size upD x N since we will traverse each dimension (each column)
    // For each dimension, we keep track the index of points sorted by its values.
    PROJECT_X_SORTED_IDX = MatrixXi(PARAM_CEOs_D_UP, PARAM_DATA_N);
    vector<int> idx(PARAM_DATA_N);

    // Sort every column - store index in GREEDY_STRUCT
    for (d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        // 0..n
        iota(idx.begin(), idx.end(), 0);

        // sort indexes based on MATRIX_X.row(d), smallest at n - 1, largest at 0
        sort(idx.begin(), idx.end(), compare_PROJECTED_X(d));

        //printVector(PROJECTED_X.col(d));
        //printVector(idx);

        // Write to GREEDY_STRUCT
        //for (n = 0; n < PARAM_DATA_N; ++n)
            //PROJECT_X_SORTED_IDX(d, n) = idx[n]; // Only contain the indexes of point
        PROJECT_X_SORTED_IDX.row(d) = Map<VectorXi>(idx.data(), PARAM_DATA_N);
    }
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
    double dStart0 = clock();
    double dStart = 0, dTopKTime = 0, dTopBTime = 0;

    int q, d, s, b;
    int n = 0;
    int iPointIdx = 0, iMaxDim = 0, iMinDim = 0;

    double dNumAccessedRows = 0.0, dNumProducts = 0.0;

    double dThreshold = 0.0;
    double dEstProduct = 0.0;

    VectorXd vecQuery(PARAM_DATA_D);
    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP);

    VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
    IVector vecTopB(PARAM_MIPS_TOP_B);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopB, minQueTopK;
    unordered_set<int> setCounter; // point already estimate

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Rotate query
        vecQuery = MATRIX_Q.col(q); // d x 1
        GaussianRP_Query(vecProjectedQuery, vecQuery);

        // Reset everything
        // minQueTopB = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        setCounter.clear();

        // Get the minimum and maximum indexes
        vecMinIdx = extract_TopK_MinIdx(vecProjectedQuery, PARAM_CEOs_S0);
        vecMaxIdx = extract_TopK_MaxIdx(vecProjectedQuery, PARAM_CEOs_S0);

        // For each col of PROJECTED_X_SORT_IDX of TA algorithm
        for (n = 0; n < round(PARAM_DATA_N / 2); ++n)
        {
            dThreshold = 0.0;

            // Compute inner product for all point in max & min dimensions
            for (s = 0; s < PARAM_CEOs_S0; ++s)
            {
                // Extract s0 max elements
                iMaxDim = vecMaxIdx[s];
                iPointIdx = PROJECT_X_SORTED_IDX(iMaxDim, n); // get the index Xi

                dThreshold += PROJECTED_X(iPointIdx, iMaxDim); // update threshold for this row

                // Compute inner product estimate if we have not computed it
                if (setCounter.find(iPointIdx) == setCounter.end())
                {
                    setCounter.insert(iPointIdx);

                    dEstProduct = 0.0;
                    for (d = 0; d < PARAM_CEOs_S0; ++d)
                        dEstProduct += (PROJECTED_X(iPointIdx, vecMaxIdx(d)) - PROJECTED_X(iPointIdx, vecMinIdx(d)));

                    // Insert into minQueue
                    if ((int)minQueTopB.size() < PARAM_MIPS_TOP_B)
                        minQueTopB.push(IDPair(iPointIdx, dEstProduct));
                    else
                    {
                        if (dEstProduct > minQueTopB.top().m_dValue)
                        {
                            minQueTopB.pop();
                            minQueTopB.push(IDPair(iPointIdx, dEstProduct));
                        }
                    }
                }


                // Extract s0 min elements
                iMinDim = vecMinIdx[s];
                iPointIdx = PROJECT_X_SORTED_IDX(iMinDim, PARAM_DATA_N - n - 1); // get the index Xi at the end
                dThreshold -= PROJECTED_X(iPointIdx, iMinDim); // update the threshold for this row

                if (setCounter.find(iPointIdx) == setCounter.end())
                {
                    // Compute inner product estimate
                    setCounter.insert(iPointIdx);

                    dEstProduct = 0.0;
                    for (d = 0; d < PARAM_CEOs_S0; ++d)
                        dEstProduct += (PROJECTED_X(iPointIdx, vecMaxIdx[d]) - PROJECTED_X(iPointIdx, vecMinIdx[d]));

                    // Insert into minQueue
                    if ((int)minQueTopB.size() < PARAM_MIPS_TOP_B)
                        minQueTopB.push(IDPair(iPointIdx, dEstProduct));
                    else
                    {
                        if (dEstProduct > minQueTopB.top().m_dValue)
                        {
                            minQueTopB.pop();
                            minQueTopB.push(IDPair(iPointIdx, dEstProduct));
                        }
                    }
                }
            }

            // Compute first threshold
            if (((int)minQueTopB.size() >= PARAM_MIPS_TOP_B) && (minQueTopB.top().m_dValue >= dThreshold))
            {
                dNumAccessedRows += (n + 1);
                dNumProducts += setCounter.size(); // number of inner product computation for each query
                break;
            }

        }

        // Copy to a vector TopB
        for (b = 0; b < PARAM_MIPS_TOP_B; ++b)
        {
            vecTopB[b] = minQueTopB.top().m_iIndex;
            minQueTopB.pop();
        }

        dTopBTime += clock() - dStart;

        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "sCEOs_TA_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_downD_" + int2str(PARAM_CEOs_S0) + "_TopK_NoPost_" + int2str(q) + ".txt");
        */
        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()

        if (PARAM_INTERNAL_GAUSS_HD3 && PARAM_INTERNAL_NOT_STORE_MATRIX_X)
            extract_TopK_MIPS_Projected_X(vecProjectedQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK); // upD x 1
        else
            extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveQueue(minQueTopK, "sCEOs_TA_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_downD_" + int2str(PARAM_CEOs_S0) + "_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Avg number of accessed rows is %f \n", dNumAccessedRows / PARAM_QUERY_Q);
    printf("Avg number of inner products is %f \n", dNumProducts / PARAM_QUERY_Q);

    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));
    printf("sCEOs TA TopK: Time is %f \n", getCPUTime(clock() - dStart0));
}

/**
- Compute matrix-maxtrix multiplication with Gaussian
- Store only B largest values of each UP_D dimensions

Output:
- MatrixXi MATRIX_CEOs_PRECOMPUTED_MIPS: of size TopB x upD containing indexes of top B for each dimension

**/
void build_1CEOs_Hash()
{
    GaussianRP_Data();

    //MAX_COL_SORT_IDX = IVector(PARAM_MIPS_TOP_K * PARAM_CEOs_D_UP);
    MATRIX_CEOs_PRECOMPUTED_MIPS = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_CEOs_D_UP);

    for (int d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        MATRIX_CEOs_PRECOMPUTED_MIPS.col(d) = extract_TopK_MaxIdx(PROJECTED_X.col(d), PARAM_MIPS_TOP_K);
        // copy(vecMaxIdx.begin(), vecMaxIdx.end(), MAX_COL_SORT_IDX.begin() + d * PARAM_MIPS_TOP_K);
    }

    // Since it is hashing, no need computing inner product.
    // Clear MATRIX_X and PROJECTED_X
    MATRIX_X.resize(0, 0);
    PROJECTED_X.resize(0, 0);

}

/** \brief Return approximate TopK for each query
 - Only use the maximum Q(1)
 - B = K, O(1) query time
 *
 * \param
 *
 - MATRIX_CEOs_PRECOMPUTED_MIPS: Top-K indexes of each dimension
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void maxCEOs_Hash()
{
    double dStart0 = clock();
    double dStart = 0, dProjectionTime = 0, dTopKTime = 0;

    int q, d;
    int iMaxIdx;
    double dMaxValue;

    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP);
    VectorXi vecTopK(PARAM_MIPS_TOP_K);

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Rotate query
        GaussianRP_Query(vecProjectedQuery, MATRIX_Q.col(q));

        // Find maximum index
        dMaxValue = vecProjectedQuery(0);
        iMaxIdx = 0;
        for (d = 1; d < PARAM_CEOs_D_UP; ++d)
        {
            if (dMaxValue < vecProjectedQuery(d))
            {
                dMaxValue = vecProjectedQuery(d);
                iMaxIdx = d;
            }
        }

        dProjectionTime += clock() - dStart;

        // Find topK
        dStart = clock();

        // Extract top K
        vecTopK = MATRIX_CEOs_PRECOMPUTED_MIPS.col(iMaxIdx);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopK, "1CEOs_upD_" + int2str(PARAM_CEOs_D_UP) + "_TopK_Hash_" + int2str(q) + ".txt");

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("1CEOs Hash: Estimating time is %f \n", getCPUTime(clock() - dStart0));
}

/**
- Compute matrix-maxtrix multiplication with Gaussian
- Store only B largest values of each UP_D dimensions

Output:
- MATRIX_CEOs_PRECOMPUTED_MIPS of size TopB x D_UP containing index of top B for each dimension

**/
void build_1CEOs_Search()
{
    GaussianRP_Data();

    // MAX_COL_SORT_IDX = IVector(PARAM_MIPS_TOP_B * PARAM_CEOs_D_UP);
    MATRIX_CEOs_PRECOMPUTED_MIPS = MatrixXi::Zero(PARAM_MIPS_TOP_B, PARAM_CEOs_D_UP);

    for (int d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        MATRIX_CEOs_PRECOMPUTED_MIPS.col(d) = extract_TopK_MaxIdx(PROJECTED_X.col(d), PARAM_MIPS_TOP_B);
        // copy(vecMaxIdx.begin(), vecMaxIdx.end(), MAX_COL_SORT_IDX.begin() + d * PARAM_MIPS_TOP_B);
    }

    // Clear projection
    PROJECTED_X.resize(0, 0);

}

/** \brief Return approximate TopK for each query
 - Only use the maximum Q(1)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - MATRIX_CEOs_PRECOMPUTED_MIPS: Top-B indexes of each dimension
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void maxCEOs_Search()
{
    double dStart0 = clock();
    double dStart = 0, dProjectionTime = 0, dTopBTime = 0, dTopKTime = 0;

    int q, d;
    int iMaxIdx;
    double dMaxValue;

    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP), vecQuery(PARAM_DATA_D);
    VectorXi vecTopB(PARAM_MIPS_TOP_B);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();
        vecQuery = MATRIX_Q.col(q);

        // Rotate query
        GaussianRP_Query(vecProjectedQuery, vecQuery);

        // Find maximum index
        dMaxValue = vecProjectedQuery(0);
        iMaxIdx = 0;
        for (d = 1; d < PARAM_CEOs_D_UP; ++d)
        {
            if (dMaxValue < vecProjectedQuery(d))
            {
                dMaxValue = vecProjectedQuery(d);
                iMaxIdx = d;
            }
        }

        dProjectionTime += clock() - dStart;

        // Find topB
        dStart = clock();

        // Extract top B
        vecTopB = MATRIX_CEOs_PRECOMPUTED_MIPS.col(iMaxIdx);

        //vecTopB.assign(MAX_COL_SORT_IDX.begin() + iMaxIdx * PARAM_MIPS_TOP_B,
          //              MAX_COL_SORT_IDX.begin() + (iMaxIdx + 1) * PARAM_MIPS_TOP_B);

        //copy(MAX_COL_SORT_IDX.begin() + iMaxIdx * PARAM_MIPS_DOT_PRODUCTS,
        //     MAX_COL_SORT_IDX.begin() + (iMaxIdx + 1) * PARAM_MIPS_DOT_PRODUCTS,
        //     vecTopB.begin());

        dTopBTime += clock() - dStart;

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
            saveQueue(minQueTopK, "1CEOs_upD_" + int2str(PARAM_CEOs_D_UP) + "_TopK_Post_" + int2str(q) + ".txt");

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("1CEOs Search: Estimating time is %f \n", getCPUTime(clock() - dStart0));
}

/**
- Compute matrix-maxtrix multiplication with Gaussian
- Store only B largest values of each UP_D dimensions

Output:
- vector<int> MAX_COL_SORT_IDX: index of top B for each dimension

**/
void build_2CEOs_Search()
{
    GaussianRP_Data();

    int d1, d2;

    VectorXd vecMaxCol(PARAM_DATA_N), vecMinCol(PARAM_DATA_N), vecDif(PARAM_DATA_N);

    //MAX_COL_SORT_IDX = IVector(PARAM_MIPS_TOP_B * PARAM_CEOs_D_UP * PARAM_CEOs_D_UP);
    MATRIX_CEOs_PRECOMPUTED_MIPS = MatrixXi::Zero(PARAM_MIPS_TOP_B, PARAM_CEOs_D_UP * PARAM_CEOs_D_UP);

    // For each dimension, we need to sort and keep top B
    vector<IDPair> priVec(PARAM_DATA_N);

    for (d1 = 0; d1 < PARAM_CEOs_D_UP; ++d1)
    {
        vecMaxCol = PROJECTED_X.col(d1);

        for (d2 = 0; d2 < PARAM_CEOs_D_UP; ++d2)
        {
            if (d2 == d1)
                continue;

            vecMinCol = PROJECTED_X.col(d2); // N x 1

            vecDif = vecMaxCol - vecMinCol;

            MATRIX_CEOs_PRECOMPUTED_MIPS.col(d1 * PARAM_CEOs_D_UP + d2) = extract_TopK_MaxIdx(vecDif, PARAM_MIPS_TOP_B);
            //copy(vecMaxIdx.begin(), vecMaxIdx.end(), MAX_COL_SORT_IDX.begin() + (d1 * PARAM_CEOs_D_UP + d2) * PARAM_MIPS_TOP_B);
            //printVector(MAX_COL_SORT_IDX);
        }
    }

    // Clear projection
    PROJECTED_X.resize(0, 0);

}

/** \brief Return approximate TopK for each query
 - Only use the maximum Q(1) and minimum Q(D)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - MATRIX_CEOs_PRECOMPUTED_MIPS: Top-B indexes of each dimension
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void minmaxCEOs_Search()
{
    double dStart0 = clock();
    double dStart = 0, dProjectionTime = 0, dTopBTime = 0, dTopKTime = 0;

    int q, d;
    int iMaxIdx, iMinIdx, iIdx;
    double dMaxValue, dMinValue, dValue;

    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP), vecQuery(PARAM_DATA_D);
    VectorXi vecTopB(PARAM_MIPS_TOP_B);

    vector<IDPair>::iterator iter, iterBegin;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q);

        // Rotate query
        GaussianRP_Query(vecProjectedQuery, vecQuery);
        //vecProjectedQuery = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x D

        // Find maximum & minimum index
        dMaxValue = vecProjectedQuery(0);
        dMinValue = vecProjectedQuery(0);
        iMaxIdx = 0;
        iMinIdx = 0;

        for (d = 1; d < PARAM_CEOs_D_UP; ++d)
        {
            dValue = vecProjectedQuery(d);

            if (dMaxValue < dValue)
            {
                dMaxValue = dValue;
                iMaxIdx = d;
            }

            if (dMinValue > dValue)
            {
                dMinValue = dValue;
                iMinIdx = d;
            }
        }

        dProjectionTime += clock() - dStart;

        // Find topB
        dStart = clock();

        // Extract top B
        iIdx = iMaxIdx * PARAM_CEOs_D_UP + iMinIdx;
        vecTopB = MATRIX_CEOs_PRECOMPUTED_MIPS.col(iIdx);

        //vecTopB.assign(MAX_COL_SORT_IDX.begin() + iIdx * PARAM_MIPS_TOP_B,
        //                MAX_COL_SORT_IDX.begin() + (iIdx + 1) * PARAM_MIPS_TOP_B);

        //printVector(vecTopB);

        dTopBTime += clock() - dStart;

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
            saveQueue(minQueTopK, "2CEOs_upD_" + int2str(PARAM_CEOs_D_UP) + "_TopK_Post_" + int2str(q) + ".txt");

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("2CEOs Search: Estimating time is %f \n", getCPUTime(clock() - dStart0));
}

/**
- Compute matrix-maxtrix multiplication with Gaussian
- Truncate PARAM_N_DOWN at the beginning and the end of the order statistics

Input:
- MATRIX_X: col-wise point set (N x D)
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x D_UP)

Output:
- vector<IDPair> MAX_COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N_DOWN x D_UP (col-maj)
- vector<IDPair> MIN_COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N_DOWN x D_UP (col-maj)

**/

void build_coCEOs_Search()
{
    // Generate normal distribution
    GaussianRP_Data();

    int d, n;

    VectorXd vecCol(PARAM_DATA_N);

    MAX_COL_SORT_DATA_IDPAIR = vector<IDPair>(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP);
    MIN_COL_SORT_DATA_IDPAIR = vector<IDPair>(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP);

    // For each dimension, we need to sort and keep top k
    vector<IDPair> priVec(PARAM_DATA_N);

    for (d = 0; d < PARAM_CEOs_D_UP; ++d)
    {
        vecCol = PROJECTED_X.col(d); // N x 1

        for (n = 0; n < PARAM_DATA_N; ++n)
            priVec[n] = IDPair(n, vecCol(n));

        // Sort X1 > X2 > ... > Xn
        sort(priVec.begin(), priVec.end(), greater<IDPair>());
        // printVector(priVec);

        // Store the beginning to N_DOWN
        copy(priVec.begin(), priVec.begin() + PARAM_CEOs_N_DOWN, MAX_COL_SORT_DATA_IDPAIR.begin() + d * PARAM_CEOs_N_DOWN);

        // Store the ending to N_DOWN
        copy(priVec.end() - PARAM_CEOs_N_DOWN, priVec.end(), MIN_COL_SORT_DATA_IDPAIR.begin() + d * PARAM_CEOs_N_DOWN);
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

void coCEOs_Map_Search()
{
    double dStart0 = clock();
    double dStart = 0, dProjectionTime = 0, dTopBTime = 0, dTopKTime = 0;

    int q, d;
    int iPointIdx;
    double dValue;

    VectorXd vecQuery(PARAM_DATA_D);
    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP);

    IVector vecTopB;
    unordered_map<int, double> mapCounter; // counting histogram of N points

    VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
    vector<IDPair>::iterator iter, iterBegin;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // d x 1

        // Rotate query
        GaussianRP_Query(vecProjectedQuery, vecQuery);
        //vecProjectedQuery = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x D

        // Clear map
        mapCounter.clear();
        mapCounter.reserve(10 * PARAM_MIPS_SAMPLES);

        // Get the minimum and maximum indexes
        vecMinIdx = extract_TopK_MinIdx(vecProjectedQuery, PARAM_CEOs_S0);
        vecMaxIdx = extract_TopK_MaxIdx(vecProjectedQuery, PARAM_CEOs_S0);

        // Esimtate MIPS using the minIdx
        for (d = 0; d < PARAM_CEOs_S0; ++d)
        {
            iterBegin = MIN_COL_SORT_DATA_IDPAIR.begin() + vecMinIdx[d] * PARAM_CEOs_N_DOWN;

            for (iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
            {
                // cout << *iter << endl;
                iPointIdx = (*iter).m_iIndex;
                dValue = (*iter).m_dValue;

                auto mapIter = mapCounter.find(iPointIdx);

                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second -= dValue;
                else // not exist
                    mapCounter.insert(make_pair(iPointIdx, -dValue));
            }
        }

        // Esimtate MIPS using the maxIdx
        for (d = 0; d < PARAM_CEOs_S0; ++d)
        {
            iterBegin = MAX_COL_SORT_DATA_IDPAIR.begin() + vecMaxIdx[d] * PARAM_CEOs_N_DOWN;

            for (iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
            {
                // cout << *iter << endl;
                iPointIdx = (*iter).m_iIndex;
                dValue = (*iter).m_dValue;

                auto mapIter = mapCounter.find(iPointIdx);

                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second += dValue;
                else // not exist
                    mapCounter.insert(make_pair(iPointIdx, dValue));
            }
        }

        dProjectionTime += clock() - dStart;

        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_TOP_B);

        dTopBTime += clock() - dStart;

        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "coCEOs_S_" + int2str(PARAM_MIPS_SAMPLES)
                    + "_upD_" + int2str(PARAM_CEOs_D_UP)
                    + "_downD" + int2str(PARAM_CEOs_S0)
                    + "_NoPost_" + int2str(q) + ".txt");
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
            saveQueue(minQueTopK, "coCEOs_Map_S_" + int2str(PARAM_MIPS_SAMPLES)
                    + "_upD_" + int2str(PARAM_CEOs_D_UP)
                    + "_downD_" + int2str(PARAM_CEOs_S0)
                    + "_TopK_Post_" + int2str(q) + ".txt");

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("coCEOs Map Search: Estimating time is %f \n", getCPUTime(clock() - dStart0));
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

void coCEOs_Vector_Search()
{
    double dStart0 = clock();
    double dStart = 0, dProjectionTime = 0, dTopBTime = 0, dTopKTime = 0;

    int q, d;
    int iPointIdx;
    double dValue;

    VectorXd vecQuery(PARAM_DATA_D);
    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP);

    IVector vecTopB;
    DVector vecCounter(PARAM_DATA_N); // counting histogram of N points

    VectorXi vecMinIdx(PARAM_CEOs_S0), vecMaxIdx(PARAM_CEOs_S0);
    vector<IDPair>::iterator iter, iterBegin;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // d x 1

        // Rotate query
        GaussianRP_Query(vecProjectedQuery, vecQuery);
        //vecProjectedQuery = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x D

        // Clear map
        fill(vecCounter.begin(), vecCounter.end(), 0.0);

        // Reset
        // Get the minimum and maximum indexes
        vecMinIdx = extract_TopK_MinIdx(vecProjectedQuery, PARAM_CEOs_S0);
        vecMaxIdx = extract_TopK_MaxIdx(vecProjectedQuery, PARAM_CEOs_S0);

        // Esimtate MIPS using the minIdx
        for (d = 0; d < PARAM_CEOs_S0; ++d)
        {
            iterBegin = MIN_COL_SORT_DATA_IDPAIR.begin() + vecMinIdx[d] * PARAM_CEOs_N_DOWN;

            for (iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
            {
                // cout << *iter << endl;
                iPointIdx = (*iter).m_iIndex;
                dValue = (*iter).m_dValue;

                vecCounter[iPointIdx] -= dValue;
            }
        }

        // Esimtate MIPS using the maxIdx
        for (d = 0; d < PARAM_CEOs_S0; ++d)
        {
            iterBegin = MAX_COL_SORT_DATA_IDPAIR.begin() + vecMaxIdx[d] * PARAM_CEOs_N_DOWN;

            for (iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
            {
                // cout << *iter << endl;
                iPointIdx = (*iter).m_iIndex;
                dValue = (*iter).m_dValue;

                vecCounter[iPointIdx] += dValue;
            }
        }

        dProjectionTime += clock() - dStart;

        // Store no post-process result
//        if (PARAM_INTERNAL_SAVE_OUTPUT)
//            saveVector(vecCounter, "coCEOs_S_" + int2str(PARAM_MIPS_SAMPLES)
//                    + "_upD_" + int2str(PARAM_CEOs_D_UP)
//                    + "_downD" + int2str(PARAM_CEOs_D_DOWN)
//                    + "_NoPost_" + int2str(q) + ".txt");


        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_TOP_B);

        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "coCEOs_Vector_S_" + int2str(PARAM_MIPS_SAMPLES)
                    + "_upD_" + int2str(PARAM_CEOs_D_UP)
                    + "_downD_" + int2str(PARAM_CEOs_S0)
                    + "_TopK_NoPost_" + int2str(q) + ".txt");
        */

        dTopBTime += clock() - dStart;

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
            saveQueue(minQueTopK, "coCEOs_Vector_S_" + int2str(PARAM_MIPS_SAMPLES)
                    + "_upD_" + int2str(PARAM_CEOs_D_UP)
                    + "_downD_" + int2str(PARAM_CEOs_S0)
                    + "_TopK_Post_" + int2str(q) + ".txt");

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("coCEOs Vector Search: Estimating time is %f \n", getCPUTime(clock() - dStart0));
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
    PROJECTED_X =  MatrixXd::Zero(PARAM_DATA_N, PARAM_RP_D_DOWN);

    // Generate normal distribution
    gaussGenerator(PARAM_DATA_D, PARAM_RP_D_DOWN);

    // MATRIX_X is col-major N x d; MATRIX_NORMAL_DISTRIBUTION is col major d x D
    PROJECTED_X = MATRIX_X * MATRIX_NORMAL_DISTRIBUTION; // (N x D)  * (D x downD) = N x downD
}

/** \brief Return approximate TopK for each query (using vector to store samples)
 - We compute all N inner products using D_DOWN dimensions based on random projection
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - PROJECTED_X: Gaussian transformation
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
void RP_Est_TopK()
{
    double dStart0 = clock();
    double dStart = 0.0, dProjectionTime = 0.0, dTopBTime = 0.0, dTopKTime = 0.0;

    int q;

    VectorXd vecQuery(PARAM_DATA_D);
    VectorXd vecProjectedQuery(PARAM_CEOs_D_UP);
    VectorXd vecMIPS(PARAM_DATA_N);

    IVector vecTopB(PARAM_MIPS_TOP_B);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Reset ever
        vecMIPS = VectorXd::Zero(PARAM_DATA_N);

        // No need to normalize the query does not change the result
        // vecQuery = MATRIX_Q.col(q).normalized(); // d x 1
        vecQuery = MATRIX_Q.col(q); // d x 1

        // Project query
        vecProjectedQuery = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // downD x 1: vector is always column wise

        // cout << "Projected query has rows " << vecProjectedQuery.rows() << " and cols " << vecProjectedQuery.cols() << endl;
        vecMIPS = PROJECTED_X * vecProjectedQuery;

        dProjectionTime += clock() - dStart;

        // Store no post-process result
        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecMIPS, "RP_Est_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_downD_" + int2str(PARAM_CEOs_S0) + "_TopK_NoPost_" + int2str(q) + ".txt");
        */
        //----------------------------------------------
        // Find topB
        dStart = clock();

        vecTopB = extract_SortedTopK_Histogram(vecMIPS, PARAM_MIPS_TOP_B);

        dTopBTime += clock() - dStart;

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;
        //printf("Computing TopK: Time is %f \n", getCPUTime(dTopKTime));

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveQueue(minQueTopK, "RP_Est_upD_" + int2str(PARAM_CEOs_D_UP) +
                      "_downD_" + int2str(PARAM_CEOs_S0) + "_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Random Projection Est TopK: Time is %f \n", getCPUTime(clock() - dStart0));
}
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

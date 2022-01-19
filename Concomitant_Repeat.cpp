#include "Concomitant.h"
#include "Utilities.h"
#include "Header.h"

/**
Compute matrix-maxtrix multiplication with Gaussian

Input:
- MATRIX_X: col-wise point set (N x D)
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x (D_UP x REPEAT))

Output:
- MatrixXd::ROTATED_X: col-wise matrix of size N x (D_UP x REPEAT) (col-maj)

**/
void GaussianRP()
{
    // Generate normal distribution
    gaussGenerator(PARAM_DATA_D, PARAM_CEOs_D_UP);

    // MATRIX_X is col-major N x d; MATRIX_NORMAL_DISTRIBUTION is col major d x D
    PROJECTED_X = MATRIX_X * MATRIX_NORMAL_DISTRIBUTION;
}

/**
- Compute matrix-maxtrix multiplication with Gaussian
- Truncate PARAM_N_DOWN at the beginning and the end of the order statistics

Input:
- MATRIX_X: col-wise point set (N x D)
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x (D_UP x REPEAT))

Output:
- vector<IDPair> MAX_COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N_DOWN x (D_UP x REPEAT) (col-maj)
- vector<IDPair> MIN_COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N_DOWN x (D_UP x REPEAT) (col-maj)

**/
void buildGaussCOS_ReduceND_Est()
{
    // Generate normal distribution
    gaussGenerator(PARAM_DATA_D, PARAM_CEOs_D_UP * PARAM_NUM_REPEAT);

    // temp_X is col-wise Nxd so matrix multiplication might be a bit slow
    // It will be deconstructed
    MatrixXd temp_ROTATED_X = MATRIX_X * MATRIX_NORMAL_DISTRIBUTION; // N x (D_UP x REPEAT)

    int d, n, r, iBaseIdx;

    VectorXd vecCol(PARAM_DATA_N);

    MAX_COL_SORT_DATA_IDPAIR = vector<IDPair>(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP * PARAM_NUM_REPEAT);
    MIN_COL_SORT_DATA_IDPAIR = vector<IDPair>(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP * PARAM_NUM_REPEAT);

    // For each dimension, we need to sort and keep top k
    vector<IDPair> priVec(PARAM_DATA_N);

    for (r = 0; r < PARAM_NUM_REPEAT; ++r)
    {
        for (d = 0; d < PARAM_CEOs_D_UP; ++d)
        {
            // BaseIdx in [PARAM_COS_D_UP x REPEAT]
            iBaseIdx = r * PARAM_CEOs_D_UP + d;

            vecCol = temp_ROTATED_X.col(iBaseIdx); // N x 1

            for (n = 0; n < PARAM_DATA_N; ++n)
                priVec[n] = IDPair(n, vecCol(n));

            // Sort X1 > X2 > ... > Xn
            sort(priVec.begin(), priVec.end(), greater<IDPair>());
            // printVector(priVec);

            // Store the beginning to N_DOWN
            copy(priVec.begin(), priVec.begin() + PARAM_CEOs_N_DOWN, MAX_COL_SORT_DATA_IDPAIR.begin() + iBaseIdx * PARAM_CEOs_N_DOWN);

            // Store the ending to N_DOWN
            copy(priVec.end() - PARAM_CEOs_N_DOWN, priVec.end(), MIN_COL_SORT_DATA_IDPAIR.begin() + iBaseIdx * PARAM_CEOs_N_DOWN);
        }
    }
}

/**
- Compute matrix-maxtrix multiplication with Gaussian
- Truncate PARAM_N_DOWN at the beginning and the end of the order statistics
- Only keep indexes of points

Input:
- MATRIX_X: col-wise point set (N x D)
- MATRIX_NORMAL_DISTRIBUTION: Gaussian matrix (D x (D_UP x REPEAT))

Output:
- IVector MAX_COL_SORT_IDX: col-wise matrix with sorted columns of size N_DOWN x (D_UP x REPEAT) (col-maj)
- IVector MIN_COL_SORT_IDX: col-wise matrix with sorted columns of size N_DOWN x (D_UP x REPEAT) (col-maj)

**/
void buildGaussCOS_ReduceND_Freq()
{
    // Generate normal distribution
    gaussGenerator(PARAM_DATA_D, PARAM_CEOs_D_UP * PARAM_NUM_REPEAT);

    // temp_X is col-wise Nxd so matrix multiplication might be a bit slow
    // It will be deconstructed
    MatrixXd temp_ROTATED_X = MATRIX_X * MATRIX_NORMAL_DISTRIBUTION; // N x (D' x REPEAT)

    int d, n, r, iBaseIdx;

    VectorXd vecCol(PARAM_DATA_N);

    MAX_COL_SORT_IDX = IVector(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP * PARAM_NUM_REPEAT);
    MIN_COL_SORT_IDX = IVector(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_UP * PARAM_NUM_REPEAT);

    // For each dimension, we need to sort and keep top k
    vector<IDPair> priVec(PARAM_DATA_N);

    for (r = 0; r < PARAM_NUM_REPEAT; ++r)
    {
        for (d = 0; d < PARAM_CEOs_D_UP; ++d)
        {
            // BaseIdx in [PARAM_COS_D_UP x REPEAT]
            iBaseIdx = r * PARAM_CEOs_D_UP + d;

            vecCol = temp_ROTATED_X.col(iBaseIdx); // N x 1

            for (n = 0; n < PARAM_DATA_N; ++n)
                priVec[n] = IDPair(n, vecCol(n));

            // Sort X1 > X2 > ... > Xn
            sort(priVec.begin(), priVec.end(), greater<IDPair>());
            // printVector(priVec);

            // Store the beginning to N_DOWN
            for (n = 0; n < PARAM_CEOs_N_DOWN; ++n)
            {
                MAX_COL_SORT_IDX[n + iBaseIdx * PARAM_CEOs_N_DOWN] = priVec[n].m_iIndex; // For the max
                MIN_COL_SORT_IDX[n + iBaseIdx * PARAM_CEOs_N_DOWN] = priVec[PARAM_DATA_N - 1 - n].m_iIndex; // For the min
            }
        }
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
    double dStart = 0, dProjectionTime = 0, dTopBTime = 0, dTopKTime = 0;

    int q, d, r;
    int iFirstIdx, iLastIdx;

    VectorXd vecQuery, vecProjectedQuery;
    VectorXd vecMIPS;

    IVector vecTopB;
    IVector vecMinIdx(PARAM_CEOs_D_DOWN), vecMaxIdx(PARAM_CEOs_D_DOWN);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Reset ever
        vecMIPS = VectorXd::Zero(PARAM_DATA_N);

        // No need to normalize the query does not change the result
        // vecQuery = MATRIX_Q.col(q).normalized(); // d x 1
        vecQuery = MATRIX_Q.col(q); // d x 1
        //printVector(vecQuery);
//        if (vecQuery.norm() > 1.0)
//            cout << "There is an error! Norm of query " << q << " is: " << vecQuery.norm() << endl;

        // Convert to 1 x d
        //vecQuery.transposeInPlace();

        // Rotate query
        vecProjectedQuery = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x (D x R)

        // Handle each repeat
        for (r = 0; r < PARAM_NUM_REPEAT; ++r)
        {
            iFirstIdx = r * PARAM_CEOs_D_UP;
            iLastIdx =  (r + 1) * PARAM_CEOs_D_UP;

            // Reset
            fill(vecMinIdx.begin(), vecMinIdx.end(), 0);
            fill(vecMaxIdx.begin(), vecMaxIdx.end(), 0);

            // Get the minimum and maximum indexes
            extract_TopK_MinMaxIdx(vecProjectedQuery, vecMinIdx, vecMaxIdx, iFirstIdx, iLastIdx, PARAM_CEOs_D_DOWN);

            // Compute estimate MIPS for N points
            for (d = 0; d < PARAM_CEOs_D_DOWN; d++)
                vecMIPS = vecMIPS + PROJECTED_X.col(vecMaxIdx[d]) - PROJECTED_X.col(vecMinIdx[d]);

        }

        dProjectionTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecMIPS, "simpleCOS_ReduceD_Est_NoPost" + int2str(q) + ".txt");

        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecMIPS, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(MATRIX_Q.col(q), vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            saveQueue(minQueTopK, "simpleCOS_ReduceD_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("simpleCOS - ReduceD: time is %f \n", getCPUTime(clock() - dStart0));
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

void simpleCOS_ReduceND_Est_TopK()
{
    double dStart0 = clock();
    double dStart = 0, dProjectionTime = 0, dTopBTime = 0, dTopKTime = 0;

    int q, d, r;
    int iFirstIdx, iLastIdx;
    int iPointIdx;
    double dValue;

    VectorXd vecQuery, vecRotated;

    IVector vecTopB;
    unordered_map<int, double> mapCounter; // counting histogram of N points

    IVector vecMinIdx(PARAM_CEOs_D_DOWN), vecMaxIdx(PARAM_CEOs_D_DOWN);
    vector<IDPair>::iterator iter, iterBegin;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Normalize the query does not change the result
        vecQuery = MATRIX_Q.col(q).normalized(); // d x 1
        //printVector(vecQuery);
//        if (vecQuery.norm() > 1.0)
//            cout << "There is an error! Norm of query " << q << " is: " << vecQuery.norm() << endl;

        // Convert to 1 x d
        //vecQuery.transposeInPlace();

        // Rotate query
        vecRotated = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x (newD x R)

        // Clear map
        mapCounter.clear();
        mapCounter.reserve(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_DOWN * PARAM_NUM_REPEAT);

        // Handle each repeats
        for (r = 0; r < PARAM_NUM_REPEAT; ++r)
        {
            iFirstIdx = r * PARAM_CEOs_D_UP;
            iLastIdx =  (r + 1) * PARAM_CEOs_D_UP;

            // Reset
            fill(vecMinIdx.begin(), vecMinIdx.end(), 0);
            fill(vecMaxIdx.begin(), vecMaxIdx.end(), 0);

            // Get the minimum and maximum indexes
            extract_TopK_MinMaxIdx(vecRotated, vecMinIdx, vecMaxIdx, iFirstIdx, iLastIdx, PARAM_CEOs_D_DOWN);

            // Esimtate MIPS using the minIdx
            for (d = 0; d < PARAM_CEOs_D_DOWN; ++d)
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
            for (d = 0; d < PARAM_CEOs_D_DOWN; ++d)
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
        }

        dProjectionTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveMap(mapCounter, "simpleCOS_ReduceND_Est_NoPost_" + int2str(q) + ".txt");

        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(MATRIX_Q.col(q), vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            saveQueue(minQueTopK, "simpleCOS_ReduceND_Est_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("simpleCOS - ReduceND: Estimating time is %f \n", getCPUTime(clock() - dStart0));
}

/** \brief Return approximate TopK for each query (using map to store samples)
 - Counting the frequency of data in the histogram of N_DOWN x D_DOWN based on order statistics
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

void simpleCOS_ReduceND_Freq_TopK()
{
    double dStart0 = clock();
    double dStart = 0, dProjectionTime = 0, dTopBTime = 0, dTopKTime = 0;

    int q, d, r;
    int iFirstIdx, iLastIdx;
    int iPointIdx;

    VectorXd vecQuery, vecRotated;

    IVector vecTopB;
    unordered_map<int, int> mapCounter; // counting histogram of N points

    IVector vecMinIdx(PARAM_CEOs_D_DOWN), vecMaxIdx(PARAM_CEOs_D_DOWN);
    IVector::iterator iter, iterBegin;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        // Normalize the query does not change the result
        vecQuery = MATRIX_Q.col(q).normalized(); // d x 1
        //printVector(vecQuery);
//        if (vecQuery.norm() > 1.0)
//            cout << "There is an error! Norm of query " << q << " is: " << vecQuery.norm() << endl;

        // Convert to 1 x d
        // vecQuery.transposeInPlace();

        // Rotate query
        vecRotated = vecQuery.transpose() * MATRIX_NORMAL_DISTRIBUTION; // of size 1 x (newD x R)

        // Clear map
        mapCounter.clear();
        mapCounter.reserve(PARAM_CEOs_N_DOWN * PARAM_CEOs_D_DOWN * PARAM_NUM_REPEAT);

        // Handle each repeats
        for (r = 0; r < PARAM_NUM_REPEAT; ++r)
        {
            iFirstIdx = r * PARAM_CEOs_D_UP;
            iLastIdx =  (r + 1) * PARAM_CEOs_D_UP;

            // Reset
            fill(vecMinIdx.begin(), vecMinIdx.end(), 0);
            fill(vecMaxIdx.begin(), vecMaxIdx.end(), 0);

            // Get the minimum and maximum indexes
            extract_TopK_MinMaxIdx(vecRotated, vecMinIdx, vecMaxIdx, iFirstIdx, iLastIdx, PARAM_CEOs_D_DOWN);

            // Esimtate MIPS using the minIdx
            for (d = 0; d < PARAM_CEOs_D_DOWN; ++d)
            {
                iterBegin = MIN_COL_SORT_IDX.begin() + vecMinIdx[d] * PARAM_CEOs_N_DOWN;

                for (iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
                {
                    iPointIdx = (*iter);

                    auto mapIter = mapCounter.find(iPointIdx);

                    if (mapIter != mapCounter.end()) // if exists
                        mapIter->second += 1;
                    else // not exist
                        mapCounter.insert(make_pair(iPointIdx, 1));
                }
            }

            // Esimtate MIPS using the maxIdx
            for (d = 0; d < PARAM_CEOs_D_DOWN; ++d)
            {
                iterBegin = MAX_COL_SORT_IDX.begin() + vecMaxIdx[d] * PARAM_CEOs_N_DOWN;

                for (iter = iterBegin; iter != iterBegin + PARAM_CEOs_N_DOWN; ++iter)
                {
                    iPointIdx = (*iter);

                    auto mapIter = mapCounter.find(iPointIdx);

                    if (mapIter != mapCounter.end() ) // if exists
                        mapIter->second += 1;
                    else // not exist
                        mapCounter.insert(make_pair(iPointIdx, 1));
                }
            }
        }

        dProjectionTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveMap(mapCounter, "simpleCOS_ReduceND_Freq_NoPost" + int2str(q) + ".txt");

        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(MATRIX_Q.col(q), vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            saveQueue(minQueTopK, "simpleCOS_ReduceND_Freq_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Projection time is %f \n", getCPUTime(dProjectionTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("simpleCOS - ReduceND: Frequency time is %f \n", getCPUTime(clock() - dStart0));
}

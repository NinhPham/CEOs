

#include "WedgeSampling.h"
#include "Utilities.h"
#include "Header.h"

/**
Presorting data for each dimension

Input:
- MATRIX_X: col-wise point set (N x D)
- p_bSign = 1/0: sort based on the absolute value (used in dWedge) or exact value (Greedy)

Output:
- vector<IDPair> COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N x D (col-maj)
- DVector POS_COL_NORM_1: column norm 1 for dWedge

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

            // True: for WedgeSampling since it uses the |dXij| for sampling
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

/**
Presorting data for each dimension using shifting preprocessing

Input:
- MATRIX_X: col-wise point set (N x D)

Output:
- vector<IDPair> COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N x D (col-maj)
- DVector POS_COL_NORM_1, NEG_COL_NORM_1: column norm 1 of different shifting strategies

**/
void dimensionShiftSort()
{
    int d, n;

    VectorXd vecCol(PARAM_DATA_N), tempCol(PARAM_DATA_N);

    // min and max values of each columns
    VECTOR_COL_MIN = MATRIX_X.colwise().minCoeff();
    //cout << VECTOR_COL_MIN << endl;

    VECTOR_COL_MAX = MATRIX_X.colwise().maxCoeff();
    //cout << VECTOR_COL_MAX << endl;

    COL_SORT_DATA_IDPAIR = vector<IDPair>(PARAM_DATA_D * PARAM_DATA_N);

    // Init for precomputed vector
    POS_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);
    NEG_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);

    // Sort every column based on the value dXij
    vector<IDPair> priVec(PARAM_DATA_N);

    for(d = 0; d < PARAM_DATA_D; ++d)
    {
        vecCol = MATRIX_X.col(d); // N x 1

        //-------------------------
        // Processing Xpos
        //-------------------------
        // Update Xpos by y_j - min(y_j)
        tempCol = vecCol - VectorXd::Ones(PARAM_DATA_N, 1) * VECTOR_COL_MIN(d); // VectorXd - scalar is not possible
        POS_COL_NORM_1[d] = tempCol.sum();

        //-------------------------
        // Processing Xneg
        //-------------------------
        // Update Xneg by -y_j + max(y_j)
        tempCol = VectorXd::Ones(PARAM_DATA_N, 1) * VECTOR_COL_MAX(d) - vecCol; // VectorXd - scalar is not possible
        NEG_COL_NORM_1[d] = tempCol.sum();


        // Create an array of Xi/ui
        for (n = 0; n < PARAM_DATA_N; ++n)
           priVec[n] = IDPair(n, vecCol(n));

        // Sort X1 > X2 > ... > Xn
        sort(priVec.begin(), priVec.end(), greater<IDPair>());
        // printVector(priVec);

        // Store
        copy(priVec.begin(), priVec.end(), COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N);
    }
}

/**
Presorting data for each dimension using positive preprocessing

Input:
- MATRIX_X: col-wise point set (N x D)

Output:
- vector<IDPair> COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N x D (col-maj)
- DVector POS_COL_NORM_1, NEG_COL_NORM_1: column norm 1 of positive and negative elements

**/
void dimensionPosSort()
{
    int d, n;

    VectorXd vecCol(PARAM_DATA_N), tempCol(PARAM_DATA_N);

    COL_SORT_DATA_IDPAIR = vector<IDPair>(PARAM_DATA_D * PARAM_DATA_N);

    // Init for precomputed vector
    POS_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);
    NEG_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);

    // Sort every column based on the value dXij
    vector<IDPair> priVec(PARAM_DATA_N);

    for(d = 0; d < PARAM_DATA_D; ++d)
    {
        vecCol = MATRIX_X.col(d); // N x 1

        // Processing Xpos & Xneg
        for (n = 0; n < PARAM_DATA_N; ++n)
        {
            priVec[n] = IDPair(n, vecCol(n));

            if (vecCol(n) >= 0.0)
                POS_COL_NORM_1[d] += vecCol(n);
            else
                NEG_COL_NORM_1[d] -= vecCol(n);
        }

        if ((POS_COL_NORM_1[d] < 0.0) || (NEG_COL_NORM_1[d] < 0.0))
             cout << "There is an error on preprocessing phase!" << endl;

        // Sort X1 > X2 > ... > Xn
        sort(priVec.begin(), priVec.end(), greater<IDPair>());
        // printVector(priVec);

        // Store
        copy(priVec.begin(), priVec.end(), COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N);
    }
}

/** \brief Presampling for wedge
 - Pre-sampling point indexes for each column based on its discrete distribution, note that now q -> |q|
 - Pre-compute norm 1 of each column
 *
 * \param
 - MatrixXd::MATRIX_X of size N x D
 *
 * \return
 - IVector: POS_PRESAMPLES: PARAM_DATA_D x PARAM_WEDGE_MAX_S as a pre-sample set for positive case
 - Dvector: POS_COL_NORM_1: norm 1 of each column of MATRIX_X_POS (1 x D)
 *
 */

void wedge_PreSampling()
{
    cout << "Max number of samples per dimension: " << PARAM_MIPS_PRESAMPLES_MAX << endl;

    // Initialize the PRESAMPLES vector
    POS_PRESAMPLES = IVector(PARAM_DATA_D * PARAM_MIPS_PRESAMPLES_MAX, 0);

    // Initialize the norm 1 of each column
    POS_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);

    IVector vecSamples;
    VectorXd vecCol(PARAM_DATA_N); // N x 1


    for (int d = 0; d < PARAM_DATA_D; ++d)
    {
        vecCol = MATRIX_X.col(d); // N x 1

        // Get abs() and norm 1
        POS_COL_NORM_1[d] = vecCol.cwiseAbs().sum();

        // Generate discrete random variables
        vecSamples = preSampling(vecCol, PARAM_TEST_SAMPLING_RANDOM_GENERATOR, true);
        copy(vecSamples.begin(), vecSamples.end(), POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX);
    }
}

/** \brief Presampling for shiftWedge
 - For each dimension: Xd - min(Xd) for positive matrix
 - For each dimension: max(Xd) - Xd for negative matrix
 - Pre-sampling point indexes for each column based on its discrete distribution
 - Pre-compute norm 1 of each column of positive and negative matrixes
 *
 * \param
 - MatrixXd::MATRIX_X of size N x D
 *
 * \return
 - IVector: POS_PRESAMPLES: PARAM_WEDGE_MAX_S x PARAM_DATA_D as a pre-sample set for positive case
 - IVector: NEG_PRESAMPLES: PARAM_WEDGE_MAX_S x PARAM_DATA_D as a pre-sample set for negative case

 - Dvector: POS_COL_NORM_1: norm 1 of each column of MATRIX_X_POS (1 x D)
 - Dvector: NEG_COL_NORM_1: norm 1 of each column of MATRIX_X_NEG (1 x D)

 - VectorXd: VECTOR_C_MIN: minimum value of each dimension (1 x D)
 - VectorXd: VECTOR_C_MAX: maximum value of each dimension (1 x D)
 *
 */

void shift_Wedge_PreSampling()
{
    cout << "Max number of samples per dimension: " << PARAM_MIPS_PRESAMPLES_MAX << endl;

    // Init for presample matrixes
    POS_PRESAMPLES = IVector(PARAM_DATA_D * PARAM_MIPS_PRESAMPLES_MAX, 0);
    NEG_PRESAMPLES = IVector(PARAM_DATA_D * PARAM_MIPS_PRESAMPLES_MAX, 0);

    // Init for precomputed vector
    POS_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);
    NEG_COL_NORM_1 = DVector(PARAM_DATA_D, 0.0);

    // min and max values of each columns
    VECTOR_COL_MIN = MATRIX_X.colwise().minCoeff();
    //cout << VECTOR_COL_MIN << endl;

    VECTOR_COL_MAX = MATRIX_X.colwise().maxCoeff();
    //cout << VECTOR_COL_MAX << endl;

    IVector vecSamples;
    VectorXd tempCol(PARAM_DATA_N), vecCol(PARAM_DATA_N); // N x 1

    for (int d = 0; d < PARAM_DATA_D; ++d)
    {
        vecCol = MATRIX_X.col(d); // N x 1

        //-------------------------
        // Processing Xpos
        //-------------------------
        // Update Xpos by y_j - min(y_j)
        tempCol = vecCol - VectorXd::Ones(PARAM_DATA_N, 1) * VECTOR_COL_MIN(d); // VectorXd - scalar is not possible

        POS_COL_NORM_1[d] = tempCol.sum();

        // Generate discrete random variables
        vecSamples = preSampling(tempCol, PARAM_TEST_SAMPLING_RANDOM_GENERATOR);
        copy(vecSamples.begin(), vecSamples.end(), POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX);

        //-------------------------
        // Processing Xneg
        //-------------------------
        // Update Xneg by -y_j + max(y_j)
        tempCol = VectorXd::Ones(PARAM_DATA_N, 1) * VECTOR_COL_MAX(d) - vecCol; // VectorXd - scalar is not possible
        NEG_COL_NORM_1[d] = tempCol.sum();

        // Generate discrete random variables
        vecSamples = preSampling(tempCol, PARAM_TEST_SAMPLING_RANDOM_GENERATOR);
        copy(vecSamples.begin(), vecSamples.end(), NEG_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX);
    }
}

/** \brief Return approximate TopK for each query (using vector to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - POS_PRESAMPLES: pre-sampled set for each column
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1: norm-1 of each column/dimension
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void wedge_Vector_TopK()
{
    double dStart0 = clock();

    int q, d;
    double dStart = 0, dSamTime = 0, dTopBTime = 0, dTopKTime = 0;

    double dQj = 0.0, dSignQj;

    int iColSamples = 0;

    DVector vecWeight(PARAM_DATA_D, 0.0); // number of samples for each dimension
    IVector vecCounter(PARAM_DATA_N, 0); // counting histogram of N points

    VectorXd vecQuery(PARAM_DATA_D); // vector of query
    IVector::iterator iter, iterBegin;

    IVector vecTopB;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        //---------------------------
        // Compute the sum of N dot products and its dimension weight
        //---------------------------

        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeWeight(vecQuery, vecWeight);

        //---------------------------
        // Access samples and update counter
        //---------------------------

        fill(vecCounter.begin(), vecCounter.end(), 0);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            dSignQj = sgn(dQj);

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);

            iterBegin = POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;

            // Wedge sampling use only positive
            for (iter = iterBegin; iter != iterBegin + iColSamples; ++iter)
            {
                // cout << *iter << endl;
                vecCounter[abs(*iter)] += sgn(*iter) * dSignQj;
            }
        }

        dSamTime += clock() - dStart;

        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "wedgeCounter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);
        dTopBTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "wedge_Vector_TopK_NoPost_" + int2str(q) + ".txt");

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
            saveQueue(minQueTopK, "wedge_Vector_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Vector-Wedge time is %f \n", getCPUTime(clock() - dStart0));
}

/** \brief Return approximate TopK for each query (using unordered_map to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - POS_PRESAMPLES: pre-sampled set for each column
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1: norm-1 of each column/dimension
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void wedge_Map_TopK()
{
    double dStart0 = clock();

    int q, d;
    double dStart = 0, dSamTime = 0, dTopBTime = 0, dTopKTime = 0;

    double dQj = 0.0;

    int iColSamples = 0;
    int iSignQj, iSignXij, iPointIdx;

    DVector vecWeight(PARAM_DATA_D, 0.0); // number of samples for each dimension

    unordered_map<int, int> mapCounter; // counting histogram of N points


    VectorXd vecQuery(PARAM_DATA_D); // vector of query
    IVector::iterator iter, iterBegin;

    IVector vecTopB;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        //---------------------------
        // Compute the sum of N dot products and its dimension weight
        //---------------------------

        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeWeight(vecQuery, vecWeight);

        //---------------------------
        // Access samples and update counter
        //---------------------------

        mapCounter.clear();
        mapCounter.reserve(PARAM_MIPS_SAMPLES);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            iSignQj = sgn(dQj);

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);

            iterBegin = POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;

            // Wedge sampling use only positive
            for (iter = iterBegin; iter != iterBegin + iColSamples; ++iter)
            {
                // cout << *iter << endl;
                iPointIdx = abs(*iter);
                iSignXij = sgn(*iter);

                auto mapIter = mapCounter.find(iPointIdx);

                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second += iSignXij * iSignQj;
                else // not exist
                    mapCounter.insert(make_pair(iPointIdx, iSignXij * iSignQj));
            }
        }

        // cout << mapCounter.size() << endl;

        dSamTime += clock() - dStart;

        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "wedgeCounter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_DOT_PRODUCTS);
        dTopBTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "wedge_Map_TopK_NoPost_" + int2str(q) + ".txt");

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
            saveQueue(minQueTopK, "wedge_Map_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Map-Wedge time is %f \n", getCPUTime(clock() - dStart0));
}


/** \brief Return approximate TopK for each query (using vector to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - POS_PRESAMPLES, NEG_PRESAMPLES: pre-sampled set for each column
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1, NEG_COL_NORM_1: vector of sum of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void shift_Wedge_Vector_TopK()
{
    double dStart0 = clock();

    int q, d;

    double dStart = 0, dSamTime = 0, dTopBTime = 0, dTopKTime = 0;

    double dQj = 0.0;

    int iColSamples = 0;

    DVector vecWeight(PARAM_DATA_D, 0.0); // number of samples for each dimension
    IVector vecCounter(PARAM_DATA_N, 0); // counting histogram of N points

    VectorXd vecQuery(PARAM_DATA_D); // vector of query
    IVector vecTopB;

    IVector::iterator iter, iterBegin;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        //---------------------------
        // Compute the sum of N dot products and its dimension weight
        //---------------------------
        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeShiftWeight(vecQuery, vecWeight);

        //---------------------------
        // Access samples and update counter
        //---------------------------
        fill(vecCounter.begin(), vecCounter.end(), 0);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);
//            if (iColSamples > PARAM_MIPS_PRESAMPLES_MAX)
//                cout << "We need to increase PARAM_MIPS_PRESAMPLES_MAX" << endl;

            // Using positive or negative component
            if (sgn(dQj) > 0)
                iterBegin = POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;
            else
                iterBegin = NEG_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;

            for (iter = iterBegin; iter != iterBegin + iColSamples; ++iter)
            {
                // cout << *iter << endl;
                vecCounter[*iter]++;
            }
        }

        dSamTime += clock() - dStart;

        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "shiftWedgeCounter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);
        dTopBTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "shift_Wedge_Vector_TopK_NoPost_" + int2str(q) + ".txt");

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
            saveQueue(minQueTopK, "shift_Wedge_Vector_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Shift Vector-Wedge time is %f \n", getCPUTime(clock() - dStart0));
}

/** \brief Return approximate TopK for each query (using unordered_map to store samples)
 - Using O(s + n) space overhead to return top B > K with simple std::vector
 - Using post-preprocessing to return top K
 *
 * \param
 *
 - POS_PRESAMPLES, NEG_PRESAMPLES: pre-sampled set for each column
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1, NEG_COL_NORM_1: vector of sum of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void shift_Wedge_Map_TopK()
{
    double dStart0 = clock();

    int q, d;
    int iPointIdx;

    double dStart = 0.0, dSamTime = 0.0, dTopBTime = 0.0, dTopKTime = 0.0;

    double dQj = 0.0;

    int iColSamples;

    DVector vecWeight(PARAM_DATA_D, 0.0); // vector of weights

    IVector vecCounter(PARAM_DATA_N, 0); // counting histogram
    IVector::iterator iter, iterBegin;

    unordered_map<int, int> mapCounter;

    VectorXd vecQuery(PARAM_DATA_D); // vector of query

    IVector vecTopB;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        //---------------------------------------
        // Compute the sum of N dot products and its dimension weight
        //---------------------------------------
        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeShiftWeight(vecQuery, vecWeight);

        //----------------------------------------
        // Access samples
        //----------------------------------------

        mapCounter.clear();
        mapCounter.reserve(PARAM_MIPS_SAMPLES);

        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            // Get number of samples
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);
//            if (iColSamples > PARAM_MIPS_PRESAMPLES_MAX)
//                cout << "We need to increase PARAM_MIPS_PRESAMPLES_MAX" << endl;

            // Using positive or negative part
            if (sgn(dQj) > 0)
                iterBegin = POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;
            else
                iterBegin = NEG_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;

            // Update histograms
            for (iter = iterBegin; iter != iterBegin + iColSamples; ++iter)
            {
                iPointIdx = *iter;

                auto mapIter = mapCounter.find(iPointIdx);

                if (mapIter != mapCounter.end())
                    mapIter->second += 1;
                else
                    mapCounter.insert(make_pair(iPointIdx, 1));
            }
        }

        // cout << mapCounter.size() << endl;

        dSamTime += clock() - dStart;

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_DOT_PRODUCTS);
        dTopBTime += clock() - dStart;

        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "shift_Wedge_Map_TopK_NoPost_" + int2str(q) + ".txt");

        //----------------------------------
        // Compute topK
        //----------------------------------

        dStart = clock();

        // Extract top K
        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>();
        extract_TopK_MIPS(vecQuery, vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        //cout << "Number of candidates: " << b << endl;
        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(minQueTopK);
        if (PARAM_TEST_SAVE_OUTPUT)
            saveQueue(minQueTopK, "shift_Wedge_Map_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step

    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Shift Map-Wedge time is %f \n", getCPUTime(clock() - dStart0));
}


/** \brief Return approximate TopK for each query (using 2 histograms)
 - Using 2 histogram O(s+n) space overhead to return top B > K with simple std::vector
 - Using post-preprocessing to return top K
 *
 * \param
 *
 - POS_PRESAMPLES, NEG_PRESAMPLES: pre-sampled set for each column
 - MATRIX_X: point set (D x N)
 - MATRIX_Q: query set (Q x D)
 - C_POS_SUM, C_NEG_SUM: vector of sum of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

//void shiftWedgeFastTopK_2Hist()
//{
//    double dStart0 = clock();
//
//    int q, d, s;
//    int iPointIdx, iOldTracker;
//
//    double dStart = 0, dSamTime = 0, dDotTime = 0;
//
//    double dQj = 0.0;
//
//    int iColSamples;
//
//    DVector vecWeight(PARAM_DATA_D, 0.0); // vector of weights
//
//    IVector vecCounter(PARAM_DATA_N, 0); // counting histogram
//    IVector::iterator iter, iterBegin;
//
//    vector<IVector> vecTracker; // tracking histogram
//    unordered_set<int> setTopB;
//
//    VectorXd vecQuery(PARAM_DATA_D); // vector of query
//    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;
//
//    // For each query
//    for (q = 0; q < PARAM_QUERY_Q; q++)
//    {
//        dStart = clock();
//
//        vecQuery = MATRIX_Q.col(q); // size D x 1
//
//        //---------------------------------------
//        // Compute the sum of N dot products and its dimension weight
//        //---------------------------------------
//        fill(vecWeight.begin(), vecWeight.end(), 0.0);
//        computeShiftWeight(vecQuery, vecWeight);
//
//        //----------------------------------------
//        // Access samples
//        //----------------------------------------
//
//        fill(vecCounter.begin(), vecCounter.end(), 0); // counting histogram
//        vecTracker.clear();
//
//        for (d = 0; d < PARAM_DATA_D; ++d)
//        {
//            dQj = vecQuery(d);
//
//            if (dQj == 0.0)
//                continue;
//
//            // Get number of samples
//            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);
//
//            // Using positive or negative part
//            if (sgn(dQj) > 0)
//                iterBegin = POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;
//            else
//                iterBegin = NEG_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX;
//
//            // Update histograms
//            for (iter = iterBegin; iter != iterBegin + iColSamples; ++iter)
//            {
//                iPointIdx = *iter;
//
//                // Get old value
//                iOldTracker = vecCounter[iPointIdx];
//                if (iOldTracker == (int)vecTracker.size()) // exist
//                {
//                    // new hash table
//                    vecTracker.emplace_back(vector<int>());
//                }
//
//                // Insert it into the tracker
//                vecTracker[iOldTracker].emplace_back(iPointIdx);
//
//                vecCounter[iPointIdx]++;
//            }
//        }
//
//        dSamTime += clock() - dStart;
//
//        // Extract top B
//        dStart = clock();
//
//        setTopB.clear();
//        //setTopB.reserve(PARAM_WEDGE_POST_B);
//
//        for (s = vecTracker.size() - 1; s >= 0; --s)
//        {
//            for (auto iPointIdx: vecTracker[s])
//            {
//                // If already have B points, stop and return
//                if ((int)setTopB.size() < PARAM_MIPS_DOT_PRODUCTS)
//                    setTopB.insert(iPointIdx);
//                else
//                {
//                    s = -1;
//                    break;
//                }
//            }
//        }
//
//        // Extract top K
//        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>();
//        extract_TopK_MIPS(vecQuery, setTopB, PARAM_MIPS_TOP_K, minQueTopK);
//
//        //cout << "Number of candidates: " << b << endl;
//        dDotTime += clock() - dStart;
//
//        // Print out or save
//        //printQueue(minQueTopK);
//        if (PARAM_TEST_SAVE_OUTPUT)
//            saveQueue(minQueTopK, "shiftWedgeFastTopK_Post_" + int2str(q) + ".txt");
//    }
//
//    // Print time complexity of each step
//    printf("Sampling time is %f \n", getCPUTime(dSamTime));
//    printf("Computing dot products time is %f \n", getCPUTime(dDotTime));
//
//    printf("Fast Shift Wedge time is %f \n", getCPUTime(clock() - dStart0));
//}

/** \brief Return approximate TopK with dWedge (using vector to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - vector<IDPair> SORTED_DATA: col-wise matrix with sorted columns of size N x D (col-maj)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1, NEG_COL_NORM_1: vector of norm-1 of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void shift_dWedge_Vector_TopK()
{
    double dStart0 = clock();
    int iColSamples;

    vector<IDPair>::iterator iter;

    int q, d;
    // double dThreshold = 0.0;

    int iSignQj, iCount, iSamples;

    double dQj = 0.0, dShiftValue = 0.0, dColNorm = 0.0;
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

        computeShiftWeight(vecQuery, vecWeight);

        //---------------------------
        // Access samples and update counter
        //---------------------------

        fill(vecCounter.begin(), vecCounter.end(), 0.0);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);
            iSignQj = sgn(dQj);

            // Using positive part
            if (iSignQj > 0)
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N;
                dColNorm = POS_COL_NORM_1[d];
            }
            else
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N + PARAM_DATA_N - 1;
                dColNorm = NEG_COL_NORM_1[d];
            }

            // Reset counting samples
            iCount = 0;
            while (iCount < iColSamples)
            {
                // value after shifing
                if (iSignQj > 0)
                    dShiftValue = (*iter).m_dValue - VECTOR_COL_MIN(d);
                else
                    dShiftValue = VECTOR_COL_MAX(d) - (*iter).m_dValue;

                iSamples = ceil(dShiftValue * iColSamples / dColNorm);

                vecCounter[(*iter).m_iIndex] += iSamples;
                iCount += iSamples;

                iter = iter + iSignQj;
            }
        }

        dSamTime += clock() - dStart;


        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "shift_dWedge_Counter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "shift_dWedge_Vector_TopK_NoPost_" + int2str(q) + ".txt");


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
        {
            saveQueue(minQueTopK, "shift_dWedge_Vector_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Shift Vector-dWedge: Time is %f \n", getCPUTime(clock() - dStart0));
}

/** \brief Return approximate TopK with dWedge (using unordered_map to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - vector<IDPair> SORTED_DATA: col-wise matrix with sorted columns of size N x D (col-maj)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1, NEG_COL_NORM_1: vector of norm-1 of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void shift_dWedge_Map_TopK()
{
    double dStart0 = clock();
    int iColSamples;

    vector<IDPair>::iterator iter;

    int q, d;
    // double dThreshold = 0.0;

    int iSignQj, iCount, iSamples;

    double dQj = 0.0, dShiftValue = 0.0, dColNorm = 0.0;
    double dStart = 0.0, dSamTime = 0.0, dTopBTime = 0.0, dTopKTime = 0.0;

    VectorXd vecQuery(PARAM_DATA_D); // vector of query

    IVector vecTopB;
    unordered_map<int, int> mapCounter;

    DVector vecWeight(PARAM_DATA_D, 0.0);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        // reset everything
        mapCounter.clear();
        mapCounter.reserve(PARAM_MIPS_SAMPLES); // For faster

        //---------------------------
        // Access samples and update counter
        //---------------------------

        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeShiftWeight(vecQuery, vecWeight);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);
            iSignQj = sgn(dQj);

            // Using positive part
            if (iSignQj > 0)
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N;
                dColNorm = POS_COL_NORM_1[d];
            }
            else
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N + PARAM_DATA_N - 1;
                dColNorm = NEG_COL_NORM_1[d];
            }

            // Reset counting samples
            iCount = 0;
            while (iCount < iColSamples)
            {
                // value after shifing
                if (iSignQj > 0)
                    dShiftValue = (*iter).m_dValue - VECTOR_COL_MIN(d);
                else
                    dShiftValue = VECTOR_COL_MAX(d) - (*iter).m_dValue;

                iSamples = ceil(dShiftValue * iColSamples / dColNorm);

                iCount += iSamples;

                auto mapIter = mapCounter.find((*iter).m_iIndex);

                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second += iSamples;
                else // not exist
                    mapCounter.insert(make_pair((*iter).m_iIndex, iSamples));

                iter = iter + iSignQj;
            }
        }

        dSamTime += clock() - dStart;


        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "shift_dWedge_Counter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "shift_dWedge_Map_TopK_NoPost_" + int2str(q) + ".txt");


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
        {
            saveQueue(minQueTopK, "shift_dWedge_Map_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Shift Map-dWedge: Time is %f \n", getCPUTime(clock() - dStart0));
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

        computeWeight(vecQuery, vecWeight);

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

                // update counter
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
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "dWedge_Vector_TopK_NoPost_" + int2str(q) + ".txt");


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

        computeWeight(vecQuery, vecWeight);

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
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "dWedge_Map_TopK_NoPost_" + int2str(q) + ".txt");


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

/** \brief Return approximate TopK with dWedge (using unordered_map to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - vector<IDPair> SORTED_DATA: col-wise matrix with sorted columns of size N x D (col-maj)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1, NEG_COL_NORM_1: vector of norm-1 of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void pos_dWedge_Map_TopK()
{
    double dStart0 = clock();
    int iColSamples;

    vector<IDPair>::iterator iter;

    int q, d;
    // double dThreshold = 0.0;

    int iSignQj, iCount, iSamples;

    double dQj = 0.0, dPosValue = 0.0, dColNorm = 0.0;
    double dStart = 0.0, dSamTime = 0.0, dTopBTime = 0.0, dTopKTime = 0.0;

    VectorXd vecQuery(PARAM_DATA_D); // vector of query

    IVector vecTopB;
    unordered_map<int, int> mapCounter;

    DVector vecWeight(PARAM_DATA_D, 0.0);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    // For each query
    for (q = 0; q < PARAM_QUERY_Q; q++)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        // reset everything
        mapCounter.clear();
        mapCounter.reserve(PARAM_MIPS_SAMPLES); // For faster

        //---------------------------
        // Access samples and update counter
        //---------------------------

        fill(vecWeight.begin(), vecWeight.end(), 0.0);
        computeShiftWeight(vecQuery, vecWeight);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);
            iSignQj = sgn(dQj);

            // Using positive part
            if (iSignQj > 0)
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N;
                dColNorm = POS_COL_NORM_1[d];
            }
            else
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N + PARAM_DATA_N - 1;
                dColNorm = NEG_COL_NORM_1[d];
            }

            // Reset counting samples
            iCount = 0;
            while (iCount < iColSamples)
            {
                // value after shifing
                if (iSignQj > 0)
                    dPosValue = (*iter).m_dValue; // get positive part
                else
                    dPosValue = -(*iter).m_dValue; // get negative part, hence need -

                iSamples = ceil(dPosValue * iColSamples / dColNorm);

                iCount += iSamples;

                auto mapIter = mapCounter.find((*iter).m_iIndex);

                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second += iSamples;
                else // not exist
                    mapCounter.insert(make_pair((*iter).m_iIndex, iSamples));

                iter = iter + iSignQj;
            }
        }

        dSamTime += clock() - dStart;


        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "pos_dWedge_Counter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(mapCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "pos_dWedge_Map_TopK_NoPost_" + int2str(q) + ".txt");


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
        {
            saveQueue(minQueTopK, "pos_dWedge_Map_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Postive Map-dWedge: Time is %f \n", getCPUTime(clock() - dStart0));
}

/** \brief Return approximate TopK with dWedge (using vector to store samples)
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - vector<IDPair> SORTED_DATA: col-wise matrix with sorted columns of size N x D (col-maj)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - POS_COL_NORM_1, NEG_COL_NORM_1: vector of norm-1 of each dimension of positive / negative matrix
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void pos_dWedge_Vector_TopK()
{
    double dStart0 = clock();
    int iColSamples;

    vector<IDPair>::iterator iter;

    int q, d;
    // double dThreshold = 0.0;

    int iSignQj, iCount, iSamples;

    double dQj = 0.0, dPosValue = 0.0, dColNorm = 0.0;
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

        computeShiftWeight(vecQuery, vecWeight);

        //---------------------------
        // Access samples and update counter
        //---------------------------

        fill(vecCounter.begin(), vecCounter.end(), 0.0);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            dQj = vecQuery(d);

            if (dQj == 0.0)
                continue;

            // Get number of samples for each dimension
            iColSamples = ceil(vecWeight[d] * PARAM_MIPS_SAMPLES);
            iSignQj = sgn(dQj);

            // Using positive part
            if (iSignQj > 0)
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N;
                dColNorm = POS_COL_NORM_1[d];
            }
            else
            {
                iter = COL_SORT_DATA_IDPAIR.begin() + d * PARAM_DATA_N + PARAM_DATA_N - 1;
                dColNorm = NEG_COL_NORM_1[d];
            }

            // Reset counting samples
            iCount = 0;
            while (iCount < iColSamples)
            {
                // value after shifing
                if (iSignQj > 0)
                    dPosValue = (*iter).m_dValue;
                else
                    dPosValue = -(*iter).m_dValue;

                iSamples = ceil(dPosValue * iColSamples / dColNorm);

                vecCounter[(*iter).m_iIndex] += iSamples;
                iCount += iSamples;

                iter = iter + iSignQj;
            }
        }

        dSamTime += clock() - dStart;


        //if (PARAM_TEST_SAVE_OUTPUT)
            //saveVector(vecCounter, "pos_dWedge_Counter_" + int2str(q) + ".txt");

        //----------------------------------
        // Insert into priority queue TopB
        //----------------------------------

        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_DOT_PRODUCTS);

        dTopBTime += clock() - dStart;

        // Store no post-process result
        if (PARAM_TEST_SAVE_OUTPUT)
            saveVector(vecTopB, "pos_dWedge_Vector_TopK_NoPost_" + int2str(q) + ".txt");


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
        {
            saveQueue(minQueTopK, "pos_dWedge_Vector_TopK_Post_" + int2str(q) + ".txt");
        }

    }

    // Print time complexity of each step
    printf("Sampling time is %f \n", getCPUTime(dSamTime));
    printf("TopB time is %f \n", getCPUTime(dTopBTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Positive Vector-dWedge: Time is %f \n", getCPUTime(clock() - dStart0));
}

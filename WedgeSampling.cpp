<<<<<<< HEAD
#include "WedgeSampling.h"
#include "Utilities.h"
#include "Header.h"


=======


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
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

/**
Presorting data for each dimension

Input:
<<<<<<< HEAD
- MATRIX_X: col-wise point set (D x N)
=======
- MATRIX_X: col-wise point set (N x D)
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134
- p_bSign = 1/0: sort based on the absolute value (in dWedge) or exact value (in MIPS-Greedy)

Output:
- vector<IDPair> COL_SORT_DATA_IDPAIR: col-wise matrix with sorted columns of size N x D (col-maj)
- DVector POS_COL_NORM_1: column 1-norm for dWedge

**/
void dimensionSort(bool p_bSign)
<<<<<<< HEAD
{
    // Note for big data, this data structre eat much RAM
    // Solution: increase the virtual memory
    // Or
    // Keep 10% of points eg PARAM_INTERNAL_dWEDGE_N, since we never reach these sorted col given that MIPS_SAMPLES = N
    // Note: Greedy does not support this compressed data structure since it has to iterate from top (if Qj> 0) or bottom (if Qj < 0)
    // dWedge only iterates from top since we consider abs(Xij)

    WEDGE_SORTED_COL = vector<IFPair>(PARAM_DATA_D * PARAM_INTERNAL_dWEDGE_N);
    WEDGE_COL_NORM_1 = VectorXf::Zero(PARAM_DATA_D);

    #pragma omp parallel for
    for (int d = 0; d < PARAM_DATA_D; ++d)
    {
        // sort the column for greedy & wedge
        vector<IFPair> priVec(PARAM_DATA_N);

        // We process each row corresponding to each dimension
        VectorXf vecRow = MATRIX_X.row(d); // N x 1

        // Create an array of Xi/ui
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            float Xij = vecRow(n);
            int iSign = sgn(Xij);

            WEDGE_COL_NORM_1(d) += iSign * Xij; // abs(dXij);

            // True: for dWedge since it uses the |dXij| for sampling
            if (p_bSign)
                priVec[n] = IFPair(iSign * n, iSign * Xij); // wedge keep both sign and index, so we use sgn(Xij) * n, abs(Xij)
            else // False for Greedy
                priVec[n] = IFPair(n, Xij);
        }

        // Sort X1 > X2 > ... > Xn
        sort(priVec.begin(), priVec.end(), greater<IFPair>());
        // printVector(priVec);

        // Store
        copy(priVec.begin(), priVec.begin() + PARAM_INTERNAL_dWEDGE_N,
                            WEDGE_SORTED_COL.begin() + d * PARAM_INTERNAL_dWEDGE_N);
    }
}

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
void wedge_ColWeight(const Ref<VectorXf> &p_vecQuery, Ref<VectorXf> p_vecWeight)
{
    float fSum = 0.0;
    for (int d = 0; d < PARAM_DATA_D; ++d)
    {
        p_vecWeight(d) = WEDGE_COL_NORM_1(d) * abs(p_vecQuery(d));
        fSum += p_vecWeight(d);
    }

    // Normalize weight
    p_vecWeight /= fSum;

    // p_vecWeight = WEDGE_COL_NORM_1.cwiseProduct(p_vecQuery.cwiseAbs());
    // p_vecWeight /= p_vecWeight.sum();

}

/** \brief Return approximate TopK using dWedge (using map to store histogram)
 *
 * \param
 *
 - vector<IFPair> WEDGE_SORTED_COL: sorted each col and store <pointIdx * sign(Xij), abs(Xij)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - WEDGE_COL_NORM_1: vector of norm-1 of each dimension
=======
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
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

<<<<<<< HEAD
void dWedge_Map_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float estTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    #pragma omp parallel for reduction(+:estTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; q++)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // size D x 1

        // Compute weighted vector
        VectorXf vecWeight = VectorXf::Zero(PARAM_DATA_D);
         //wedge_ColWeight(vecQuery, vecWeight);
        vecWeight = WEDGE_COL_NORM_1.cwiseProduct(vecQuery.cwiseAbs());
        vecWeight /= vecWeight.sum();

        // Accessing samples and update counter value
        unordered_map<int, float> mapCounter;
        mapCounter.reserve(PARAM_MIPS_NUM_SAMPLES);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (int d = 0; d < PARAM_DATA_D; ++d)
        {
            float Qj = vecQuery(d);

            if (Qj == 0)
                continue;

            int iSignQj = sgn(Qj);

            // Prepare sampling

            // 1) Get number of samples for each dimension
            int iColSamples = ceil(vecWeight(d) * PARAM_MIPS_NUM_SAMPLES);
            // 2) Get sorted col value
            // Note that WEDGE_SORT_COL has key = sgn(Xij) * iPointIdx, value = abs(Xij)
            vector<IFPair>::iterator iter = WEDGE_SORTED_COL.begin() + d * PARAM_INTERNAL_dWEDGE_N;

            // Reset counting samples
            int iCount = 0;
            while (iCount <= iColSamples)
            {
                // Get point index with its sign
                int iPointIdx = abs((*iter).m_iIndex);
                int iSignXij = sgn((*iter).m_iIndex);

                // number of samples
                int iSamples = ceil((*iter).m_fValue * iColSamples / WEDGE_COL_NORM_1(d));

                // Update current # samples
                iCount += iSamples;

                auto mapIter = mapCounter.find(iPointIdx);

                if (mapIter != mapCounter.end() ) // if exists
                    mapIter->second += iSamples * iSignXij * iSignQj;
                else // not exist
                    mapCounter.insert(make_pair(iPointIdx, iSamples * iSignXij * iSignQj));

                // Check next iteration
=======
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

>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134
                ++iter;
            }
        }

<<<<<<< HEAD
        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        estTime += (float)durTime.count() / 1000000;

        ///----------------------------------------------
        // Find topB & topK together
        startTime = chrono::high_resolution_clock::now();
        extract_TopB_TopK_Histogram(mapCounter, vecQuery, PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in second is %f \n", estTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("dWedge-Map TopK Time in second is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
        outputFile(matTopK, "dWedge_Map_S_" + int2str(PARAM_MIPS_NUM_SAMPLES) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt");

}

/** \brief Return approximate TopK using dWedge (using vector to store histogram)
 *
 * \param
 *
 - vector<IFPair> WEDGE_SORTED_COL: sorted each col and store <pointIdx * sign(Xij), abs(Xij)
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (D x Q)
 - WEDGE_COL_NORM_1: vector of norm-1 of each dimension
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */

void dWedge_Vector_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float estTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    #pragma omp parallel for reduction(+:estTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; q++)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // size D x 1

        // Compute weighted vector
        VectorXf vecWeight = VectorXf::Zero(PARAM_DATA_D);

        //wedge_ColWeight(vecQuery, vecWeight);

        vecWeight = WEDGE_COL_NORM_1.cwiseProduct(vecQuery.cwiseAbs());
        vecWeight /= vecWeight.sum();

        // Accessing samples and update counter value
        // Since float and int using 4-byte so there is no difference on speed
        VectorXf vecCounter = VectorXf::Zero(PARAM_DATA_N);

        // Get all samples from d dimensions, store in a vector for faster sequential access
        for (int d = 0; d < PARAM_DATA_D; ++d)
        {
            float Qj = vecQuery(d);

            if (Qj == 0)
                continue;

            int iSignQj = sgn(Qj);

            // Prepare sampling

            // 1) Get number of samples for each dimension
            int iColSamples = ceil(vecWeight(d) * PARAM_MIPS_NUM_SAMPLES);
            // 2) Get sorted col value
            // Note that WEDGE_SORT_COL has key = sgn(Xij) * iPointIdx, value = abs(Xij)
            vector<IFPair>::iterator iter = WEDGE_SORTED_COL.begin() + d * PARAM_INTERNAL_dWEDGE_N;

            // Reset counting samples
            int iCount = 0;
            while (iCount <= iColSamples)
            {
                // number of samples on this dimension d
                int iSamples = ceil((*iter).m_fValue * iColSamples / WEDGE_COL_NORM_1(d));

                iCount += iSamples;

                // update counter: we need abs(m_iIndex) since we keep the originial sign in the index
                vecCounter(abs((*iter).m_iIndex)) += iSamples * sgn((*iter).m_iIndex) * iSignQj;

                ++iter;
            }
        }

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        estTime += (float)durTime.count() / 1000000;

        ///----------------------------------------------
        // Find topB & topK together
        startTime = chrono::high_resolution_clock::now();

        extract_TopB_TopK_Histogram(vecCounter, vecQuery, PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Estimation time in second is %f \n", estTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("dWedge-Vector TopK Time in second is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
        outputFile(matTopK, "dWedge_Vector_S_" + int2str(PARAM_MIPS_NUM_SAMPLES) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt");

}

/** \brief Return approximate TopK of MIPS for each query. Implements the Greedy algorithm from the paper NIPS 17
 *
 * \param
 *
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (Q x D)
 - vector<IFPair> WEDGE_SORTED_COL: sorted each col and store <pointIdx, Xij>
 *
 * \return
 - Top K MIPS
 *
 */
void greedy_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float candTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    #pragma omp parallel for reduction(+:candTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        VectorXf vecQuery = MATRIX_Q.col(q); // size D x 1

        vector<int> vecNextPointIdx(PARAM_DATA_D, 0); // contain the index of the point for the next verification
        vector<int> vecCand; // Set with candidates already added.
        boost::dynamic_bitset<> bitsetHist(PARAM_DATA_N);

        priority_queue<IFPair, vector<IFPair>> maxCandQueue; // Queue used to store candidates.

        // Get the pointIdx with max value for each dimension
        for (int d = 0; d < PARAM_DATA_D; ++d)
        {
            // First, set up vecNextPointIdx (0 if signQ < 0 else n-1)
            int iSignQj = sgn(vecQuery(d));
            if (iSignQj < 0)
                vecNextPointIdx[d] = PARAM_DATA_N - 1;

            // Get the point index whose value is largest
            vector<IFPair>::iterator iter = WEDGE_SORTED_COL.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

            float fValue = (*iter).m_fValue * vecQuery(d); // Value of point
            maxCandQueue.push(IFPair(d, fValue)); // Add to queue
        }

        // Extract candidates
        while ((int)vecCand.size() < PARAM_MIPS_TOP_B) // Will do at most Bd rounds
        {
            // Extract the dimension d with the max partial product
            int d = maxCandQueue.top().m_iIndex;
            maxCandQueue.pop();

            // Get pointIdx and add to candidate set if not visited
            vector<IFPair>::iterator iter = WEDGE_SORTED_COL.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

            int pointIdx = (*iter).m_iIndex; // get next index

            // If not visited
            if (~bitsetHist[pointIdx])
            {
                vecCand.push_back(pointIdx); // Add to set
                bitsetHist[pointIdx] = 1;
            }

            while (true)
            {
                vecNextPointIdx[d] += sgn(vecQuery(d)); // next index In-/decrement counter

                // Add next element for this dimension to candQueue if any more left
                if (vecNextPointIdx[d] >= 0 && vecNextPointIdx[d] < PARAM_DATA_N)
                {
                    vector<IFPair>::iterator iter = WEDGE_SORTED_COL.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

                    if (~bitsetHist[(*iter).m_iIndex]) // if not exist
                    {
                        float fValue = (*iter).m_fValue * vecQuery(d); // Value of next
                        maxCandQueue.push(IFPair(d, fValue));      // Add to queue

                        break;
                    }
                }
                else
                    break;
            }

        }

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        candTime += (float)durTime.count() / 1000000;

        // Top-k
        startTime = chrono::high_resolution_clock::now();

        extract_TopK_MIPS(vecQuery, Map<VectorXi>(vecCand.data(), vecCand.size()), PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);

    // Print time complexity of each step
    printf("Candidate time in second is %f \n", candTime);
    printf("TopK time in second is %f \n", topKTime);

    printf("Greedy TopK Time in second is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
        outputFile(matTopK, "Greedy_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt");
=======
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
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134
}


#include "CEOs.h"
#include "Header.h"
#include "Utilities.h"

// __builtin_popcount function
// #include <bits/stdc++.h>

/**
 * Build index of CEOs for estimating inner product
 *
 * @param matX: Note that CEOs::matrix_X has not been initialized, so we need to send param matX
 * We can avoid duplicated memory by loading data directly from the filename if dataset is big.
 *
 * Passing reference Eigen: https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
 */

void CEOs::build_CEOs(const Ref<const RowMajorMatrixXf> & matX)
{
    cout << "Building CEOs index..." << endl;
    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_repeats: " << CEOs::n_repeats << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;


    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(CEOs::n_threads);

    auto copy_start = chrono::high_resolution_clock::now();

    // NOTE: Do not need centering since it does not affect the inner product estimation
    // if (CEOs::centering)
    // {
    //     VectorXf vec_centerX = matX.rowwise().mean(); // Centering the data
    //     CEOs::matrix_X = matX.rowwise() - vec_centerX; // Centered data
    // }
    // else
    // {
    //     CEOs::matrix_X = matX; // No centering
    // }

    CEOs::matrix_X = matX; // No centering
    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - copy_start);
    cout << "Copying data time (in seconds): " << (float)duration.count() / 1000 << endl;

    // Note that if fhtDim > n_proj, then we need to retrieve the first n_proj columns of the projections
    // This will save memory footprint if fhtDim is much larger than n_proj
    // We need N x (proj * repeat) since query phase will access each column corresponding each random vector
    CEOs::matrix_P = MatrixXf::Zero(CEOs::n_points, CEOs::n_proj * CEOs::n_repeats);

    bitHD3Generator(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD);

    int log2_FHT = log2(CEOs::fhtDim);

#pragma omp parallel for
    for (int n = 0; n < CEOs::n_points; ++n)
    {
        VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);
        tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n);

        // For each repeat
        for (int r = 0; r < CEOs::n_repeats; ++r) {

            VectorXf rotatedX = tempX;
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * r;

            for (int i = 0; i < CEOs::n_rotate; ++i)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(CEOs::bitHD[baseIdx + i * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            // Store it into the matrix_P of size N x (n_proj * n_repeats)
//            cout << CEOs::n_proj * r + 0 << " " << CEOs::n_proj * r + CEOs::n_proj << endl;
//            cout << rotatedX.segment(0, CEOs::n_proj).transpose() << endl;

            // Note for segment(i, size) where i is starting index, size is segment size
            CEOs::matrix_P.row(n).segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj); // only get up to #n_proj
//            CEOs::matrix_P.block(n, CEOs::n_proj * r + 0, 1, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj).transpose();
//            cout << matrix_P.row(n) << endl;
        }
    }

    double dSize = 1.0 * (CEOs::matrix_P.rows() * CEOs::matrix_P.cols() + CEOs::matrix_X.rows() * CEOs::matrix_X.cols()) * sizeof(float) / (1 << 30);
    cout << "Size of CEOs index in GB: " << dSize << endl;

    dSize = 1.0 * CEOs::matrix_X.rows() * CEOs::matrix_X.cols() * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Index construction time (in seconds): " << (float)duration.count() / 1000 << endl;
}

/**
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_CEOs(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    int n_queries = matQ.rows();

    if (verbose)
    {
        cout << "n_queries: " << n_queries << endl;
        cout << "n_probed_vectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probed_points: " << CEOs::n_probed_points << endl;
        cout << "n_cand: " << CEOs::n_cand << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();
//    double itime = omp_get_wtime();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors );
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int log2_FHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime)

    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        VectorXf vecProject = VectorXf (CEOs::n_proj * CEOs::n_repeats);

        // For each repeat
        for (int r = 0; r < CEOs::n_repeats; ++r)
        {
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * r;
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);

            rotatedQ.segment(0, CEOs::n_features) = vecQuery;

            for (int i = 0; i < CEOs::n_rotate; ++i)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(CEOs::bitHD[baseIdx + i * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedQ.segment(0, CEOs::n_proj); // only get up to #n_proj

        }

        // Now look for the top-s closest random vector -- note that we consider both close/far vectors
        priority_queue< IFPair, vector<IFPair>, greater<> > minQue;

        /**
         * matrix_P contains the projection values, of size n x (e * n_proj)
         * For query, we apply a simple trick to restore the furthest/closest vector
         * We increase index by 1 to get rid of the case of 0, and store a sign to differentiate the closest/furthest
         * Remember to convert this value back to the corresponding index of matrix_P
         * This fix is only for the rare case where r_0 at exp = 0 has been selected, which happen with very tiny probability
         */
        for (int d = 0; d < CEOs::n_repeats * CEOs::n_proj; ++d)
        {
            float fAbsProjValue = vecProject(d);

            // Hack: We increase by 1 since the index start from 0 and cannot have +/-
            // This trick would make implementation simpler compared to d + n_proj if negative since we have many repeats
            int iCol_plus_one = d + 1;

            if (fAbsProjValue < 0)
            {
                fAbsProjValue = -fAbsProjValue; // get abs
                iCol_plus_one = -iCol_plus_one; // use minus to indicate furthest vector
            }

            if ((int)minQue.size() < CEOs::n_probed_vectors)
                minQue.emplace(iCol_plus_one, fAbsProjValue);

            // queue is full
            else if (fAbsProjValue > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(iCol_plus_one, fAbsProjValue);
            }

        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Get top-r position out of n_proj * n_repeats and compute estimates
        /**
         * Note: This step is not efficient with large number of threads, e.g. 32 threads with n = 1M
         * E.g. Cache size is not big enough to load an array of n elements for each thread
         * coCEOs is thread-friendly since we can use top-point and top-proj to control the cache-size for each thread
         */
        startTime = chrono::high_resolution_clock::now();
        VectorXf vecEst = VectorXf::Zero(CEOs::n_points);
        for (int i = 0; i < CEOs::n_probed_vectors; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();

            // Negative, means furthest away
            if (ifPair.m_iIndex < 0)
                vecEst -= CEOs::matrix_P.col(-ifPair.m_iIndex - 1);  // We need to subtract 1 since we increased by 1 for the case of selected r_0
            else
                vecEst += CEOs::matrix_P.col(ifPair.m_iIndex - 1); // We need to subtract 1 since we increased by 1 for the case of selected r_0
        }

        estTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Extract top candidate
        startTime = chrono::high_resolution_clock::now();
        for (int n = 0; n < CEOs::n_points; ++n)
        {
            if ((int)minQue.size() < CEOs::n_cand)
                minQue.emplace(n, vecEst(n));

                // queue is full
            else if (vecEst(n) > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(n, vecEst(n));
            }
        }

        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Distance computation
        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        for (int i = 0; i < CEOs::n_cand; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();
            int iPointIdx = ifPair.m_iIndex;

            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
            }
        }

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);
//    double exeTime = omp_get_wtime() - itime;

    if (verbose)
    {
        cout << "Projecting and extracting top-vectors time: " << projTime << " ms" << endl;
        cout << "Estimating time: " << estTime << " ms" << endl;
        cout << "Extracting candidates time: " << candTime << " ms" << endl;
        cout << "Computing distance time: " << distTime << " ms" << endl;
        cout << "Querying time: " << (float)durTime.count() << " ms" << endl;

        // string sFileName = "CEOs_Est_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
     }

    return make_tuple(matTopK, matTopDist);
}

/**
 * Build index of coCEOs for estimating inner product
 *
 * @param matX: Note that CEOs::matrix_X has not been initialized, so we need to send param matX
 * We can avoid duplicated memory by loading data directly from the filename if dataset is big.
 * This implementation computes matrix_P first, then reduce its size.
 * Though using more temporary memory but faster parallel, and perhaps cache-friendly
 *
 * Caveat: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders
 * While this approach allows fully optimized vectorized calculations in Eigen,
 * it cannot be used with array slices (i.e. when sending matX as an array slices from numpy)
 */
void CEOs::build_coCEOs_Est(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "Building coCEOs-Estimate index..." << endl;

    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_repeats: " << CEOs::n_repeats << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "top_m: " << CEOs::top_m << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(CEOs::n_threads);

    CEOs::matrix_X = matX;
    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Copying data time (in seconds): " << (float)duration.count() / 1000 << endl;

    // CEOs::matrix_P has (4 * top-m) x (proj * repeats) since query phase will access each column corresponding each random vector
    // We need 2 * top-points position for (index, value)
    CEOs::matrix_P = MatrixXf::Zero(4 * CEOs::top_m, CEOs::n_proj * CEOs::n_repeats);

    // mat_pX is identical to CEOs estimation, has (n_points) x (proj * repeats)
    MatrixXf mat_pX = MatrixXf::Zero(CEOs::n_points, CEOs::n_proj * CEOs::n_repeats);
    bitHD3Generator(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD);

    int log2_FHT = log2(CEOs::fhtDim);

    float extractTopPointsTime = 0.0, projTime = 0.0;

    // First, we compute the projection of each point, store it into mat_pX
#pragma omp parallel for reduction(+:projTime)
    for (int n = 0; n < CEOs::n_points; ++n) {

        auto startTime = chrono::high_resolution_clock::now();

        VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);
        tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n);

        // For each repeat
        for (int r = 0; r < CEOs::n_repeats; ++r) {

            VectorXf rotatedX = tempX;
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * r;

            for (int i = 0; i < CEOs::n_rotate; ++i)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(CEOs::bitHD[baseIdx + i * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            mat_pX.row(n).segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj); // only get up to #n_proj
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }


    // Second, we extract indexBucketSize points closest/furthest to each random vector (i.e. for each column)
    // Since number of repeats is small, we parallel on number of projection vectors (i.e. second loop)
#pragma omp parallel for reduction(+: extractTopPointsTime)
    for (int iCol = 0; iCol < CEOs::n_repeats * CEOs::n_proj; ++iCol)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Note that minQueFar has to deal with minus
        // Since indexBuckeSize << n, using priority queue is faster than make_heap and pop_heap
        // Also, we will deque to get top-points closest/furthest to random vector, so the queues will be empty in the end
        priority_queue<IFPair, vector<IFPair>, greater<> > minQueClose, minQueFar; // for closest

        VectorXf projectedVec = mat_pX.col(iCol);

        for (int n = 0; n < CEOs::n_points; ++n){
            float fProjectedValue = projectedVec(n);

            // Closest points
            if ((int)minQueClose.size() < CEOs::top_m)
                minQueClose.emplace(n, fProjectedValue);
            else if (fProjectedValue > minQueClose.top().m_fValue)
            {
                minQueClose.pop();
                minQueClose.emplace(n, fProjectedValue);
            }

            // Furthest points: need to deal with -
            if ((int)minQueFar.size() < CEOs::top_m)
                minQueFar.emplace(n, -fProjectedValue);
            else
            {
                if (-fProjectedValue > minQueFar.top().m_fValue)
                {
                    minQueFar.pop();
                    minQueFar.emplace(n, -fProjectedValue);
                }
            }
        }

        // Now deque and store into matrix_P:
        // the first top-point is idx, the second top-points is value for closest
        // the third top-point is idx, the fourth top-points is value for furthest
        for (int m = CEOs::top_m - 1; m >= 0; --m)
        {
            // Close: 1st is index, 2nd is projected value
            CEOs::matrix_P(m, iCol) = minQueClose.top().m_iIndex;
            CEOs::matrix_P(m + 1 * CEOs::top_m, iCol) = minQueClose.top().m_fValue;
            minQueClose.pop();

            // Far: 3rd is index, 4th is projected value
            CEOs::matrix_P(m + 2 * CEOs::top_m, iCol) = minQueFar.top().m_iIndex;
            CEOs::matrix_P(m + 3 * CEOs::top_m, iCol) = minQueFar.top().m_fValue; // Store -projectedValue
            minQueFar.pop();
        }

        extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }

    double dSize = 1.0 * (CEOs::matrix_P.rows() * CEOs::matrix_P.cols() + CEOs::matrix_X.rows() * CEOs::matrix_X.cols()) * sizeof(float) / (1 << 30);
    cout << "Size of coCEOs-Est index in GB: " << dSize << endl;

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Projecting time: " << projTime << " ms" << endl;
    cout << "Extracting top-m points time: " << extractTopPointsTime << " ms" << endl;
    cout << "Constructing time (in seconds): " << (float)duration.count() / 1000 << endl;
}

/**
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_coCEOs_Est(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (CEOs::n_probed_points > CEOs::top_m)
    {
        cerr << "Error: Number of probed points must be smaller than number of indexed top-m points !" << endl;
        exit(1);
    }
    if (CEOs::n_probed_vectors > CEOs::n_proj * CEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than n_proj * n_repeats !" << endl;
        exit(1);
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_queries: " << n_queries << endl;
        cout << "n_probedVectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probedPoints: " << CEOs::n_probed_points << endl;
        cout << "n_cand: " << CEOs::n_cand << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int log2_FHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime)

    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        VectorXf vecProject = VectorXf (CEOs::n_proj * CEOs::n_repeats);

        // For each repeat
        for (int r = 0; r < CEOs::n_repeats; ++r)
        {
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * r;
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);

            rotatedQ.segment(0, CEOs::n_features) = vecQuery;

            for (int i = 0; i < CEOs::n_rotate; ++i)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(CEOs::bitHD[baseIdx + i * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedQ.segment(0, CEOs::n_proj); // only get up to #n_proj

        }

        // Now look for the top-k (i.e. n_probedBucket) closest/furtheest random vector to the query
        // The minQue will be empty after extracting top-k closest/furthest random vector to the query
        priority_queue< IFPair, vector<IFPair>, greater<> > minQue;

        /**
         * matrix_P contains the projection values, of size n x (e * n_proj)
         * For query, we apply a simple trick to restore the furthest/closest vector
         * We increase index by 1 to get rid of the case of 0, and store a sign to differentiate the closest/furthest
         * Remember to convert this value back to the corresponding index of matrix_P
         * This fix is only for the rare case where r_0 at exp = 0 has been selected, which happen with very tiny probability
         */
        for (int d = 0; d < CEOs::n_repeats * CEOs::n_proj; ++d)
        {
            float fAbsProjValue = vecProject(d);

            // Must increase by 1 after getting the value
            int iCol = d + 1; // Hack: We increase by 1 since the index start from 0 and cannot have +/-

            if (fAbsProjValue < 0)
            {
                fAbsProjValue = -fAbsProjValue; // get abs
                iCol = -iCol; // use minus to indicate furthest vector
            }

            if ((int)minQue.size() < CEOs::n_probed_vectors)
                minQue.emplace(iCol, fAbsProjValue);

            // queue is full
            else if (fAbsProjValue > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(iCol, fAbsProjValue);
            }
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Get top-r position out of n_proj * n_repeats and compute estimates
        startTime = chrono::high_resolution_clock::now();

        // If we want to estimate distance, then using vecEst.
        // Otherwise, using unordered_map
//        VectorXf vecEst = VectorXf::Zero(CEOs::n_points);
        tsl::robin_map<int, float> mapEst;
//        tsl::robin_map<int, pair<int, float> > mapEst;
        mapEst.reserve(CEOs::n_probed_vectors * CEOs::n_probed_points);

        for (int i = 0; i < CEOs::n_probed_vectors; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();

            // Negative, means furthest away
            if (ifPair.m_iIndex < 0)
            {
                int iCol = -ifPair.m_iIndex - 1; // decrease by one due to the trick of increase by 1 and use the sign for closest/furthest
                for (int m = 0; m < CEOs::n_probed_points; ++m) // only consider up to probed_points
                {
                    int iPointIdx = int(CEOs::matrix_P(m + 2 * CEOs::top_m, iCol));
                    float fValue = CEOs::matrix_P(m + 3 * CEOs::top_m, iCol); // we store -projectedValue

                    if (mapEst.find(iPointIdx) == mapEst.end())
                        mapEst[iPointIdx] = fValue;
//                        mapEst[iPointIdx] = make_pair(1, fValue);
                    else
                    {
                        mapEst[iPointIdx] += fValue;
//                        mapEst[iPointIdx] = make_pair(mapEst[iPointIdx].first + 1, mapEst[iPointIdx].second + fValue);
//                        mapEst[iPointIdx].first++;
//                        mapEst[iPointIdx].second += fValue;
                    }

                }
            }
            else // close
            {
                int iCol = ifPair.m_iIndex - 1; // decrease by one due to the trick of increase by 1 and use the sign for closest/furthest
                for (int m = 0; m < CEOs::n_probed_points; ++m) // only consider up to probed_points
                {
                    int iPointIdx = int(CEOs::matrix_P(m, iCol));
                    float fValue = CEOs::matrix_P(m + CEOs::top_m, iCol);

                    if (mapEst.find(iPointIdx) == mapEst.end())
                        mapEst[iPointIdx] = fValue;
//                        mapEst[iPointIdx] = make_pair(1, fValue);
                    else
                    {
                        mapEst[iPointIdx] += fValue;
//                        mapEst[iPointIdx] = make_pair(mapEst[iPointIdx].first + 1, mapEst[iPointIdx].second + fValue);
//                        mapEst[iPointIdx].first++;
//                        mapEst[iPointIdx].second += fValue;
                    }

                }
            }
        }

        estTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // If the cost of estTime and exact candidate is large with n = 10^9, then consider google dense map
        assert(minQue.size() == 0);
        startTime = chrono::high_resolution_clock::now();

        // Only for probedPoints and probedBucket inputs
        for (auto& it: mapEst)
        {
            float avgEst = it.second;
//            float avgEst = it.second.second / it.second.first;
            if ((int)minQue.size() < CEOs::n_cand)
                minQue.emplace(it.first, avgEst); // use average value for estimation

            // queue is full
            else if (avgEst > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(it.first, avgEst); // use average value for estimation
            }
        }

        assert(minQue.size() == CEOs::n_cand);

        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;
        for (int i = 0; i < CEOs::n_cand; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();
            int iPointIdx = ifPair.m_iIndex;

            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
            }
        }

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);


    if (verbose)
    {
        cout << "Projecting and extracting top-vectors time: " << projTime << " ms" << endl;
        cout << "Estimating time: " << estTime << " ms" << endl;
        cout << "Extracting candidates time: " << candTime << " ms" << endl;
        cout << "Computing distance time: " << distTime << " ms" << endl;
        cout << "Querying time: " << (float)durTime.count() << " ms" << endl;

        // string sFileName = "coCEOs_Est_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_topPoints_" + int2str(CEOs::n_probed_points) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK, matTopDist);
}

/**
 * Build index of coCEOs for estimating inner product
 *
 * @param matX: Note that CEOs::matrix_X has not been initialized, so we need to send param matX
 * We can avoid duplicated memory by loading data directly from the filename if dataset is big.
 * This implementation computes matrix_P first, then reduce its size.
 * Though using more temporary memory but faster parallel, and perhaps cache-friendly
 *
 * Caveat: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders
 * While this approach allows fully optimized vectorized calculations in Eigen,
 * it cannot be used with array slices (i.e. when sending matX as an array slices from numpy)
 */
void CEOs::build_CEOs_Hash(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "top_m: " << CEOs::top_m << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    omp_set_num_threads(CEOs::n_threads);

    auto start = chrono::high_resolution_clock::now();
    CEOs::matrix_X = matX;

    if (CEOs::centering) {

        VectorXf vecCenter = matX.rowwise().mean();

#pragma omp parallel for
        for (int n = 0; n < CEOs::n_points; ++n)
            CEOs::matrix_X.row(n) -= vecCenter;  // CEOs::matrix_X = matX.array().rowwise() - vecCenter.array(); // must add colwise()
    }

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count();
    cout << "Copying and centering data time in seconds: " << (float)duration / 1000 << endl;

    // CEOs::matrix_H has (2 * top-points) x (proj * repeats) since query phase will access each column corresponding each random vector
    // We need 2 * top-points position for (index, value)
    CEOs::matrix_H = MatrixXi::Zero(2 * CEOs::top_m, CEOs::n_proj * CEOs::n_repeats);

    // mat_pX is identical to CEOs estimation, has (n_points) x (proj * repeats)
    MatrixXf mat_pX = MatrixXf::Zero(CEOs::n_points, CEOs::n_proj * CEOs::n_repeats);

    bitHD3Generator(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD);
    int log2_FHT = log2(CEOs::fhtDim);

    float extractTopPointsTime = 0.0, projTime = 0.0;

    // First, we compute the projection of each points, store it into mat_pX
#pragma omp parallel for reduction(+:projTime)
    for (int n = 0; n < CEOs::n_points; ++n) {

        auto startTime = chrono::high_resolution_clock::now();

        VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);
        tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n);

        // For each repeat
        for (int r = 0; r < CEOs::n_repeats; ++r) {

            VectorXf rotatedX = tempX;
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * r;

            for (int i = 0; i < CEOs::n_rotate; ++i)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(CEOs::bitHD[baseIdx + i * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            mat_pX.row(n).segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj); // only get up to #n_proj
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }


    // Second, we extract top-m points closest/furthest to each random vector (i.e. for each column)
#pragma omp parallel for reduction(+: extractTopPointsTime)
    for (int iCol = 0; iCol < CEOs::n_repeats * CEOs::n_proj; ++iCol)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Note that minQueFar has to deal with minus
        // Since top-m << n, using priority queue is faster than make_heap and pop_heap
        // Also, we will deque to get top-points closest/furthest to random vector, so the queues will be empty in the end
        priority_queue<IFPair, vector<IFPair>, greater<> > minQueClose, minQueFar; // for closest and furthest

        VectorXf projectedVec = mat_pX.col(iCol);

        for (int n = 0; n < CEOs::n_points; ++n){

            float fProjectedValue = projectedVec(n);

            // Closest points
            if ((int)minQueClose.size() < CEOs::top_m)
                minQueClose.emplace(n, fProjectedValue);
            else if (fProjectedValue > minQueClose.top().m_fValue)
            {
                minQueClose.pop();
                minQueClose.emplace(n, fProjectedValue);
            }

            // Furthest points: need to deal with -
            if ((int)minQueFar.size() < CEOs::top_m)
                minQueFar.emplace(n, -fProjectedValue);
            else
            {
                if (-fProjectedValue > minQueFar.top().m_fValue)
                {
                    minQueFar.pop();
                    minQueFar.emplace(n, -fProjectedValue);
                }
            }
        }

        // Now deque and store into matrix_H:
        // the first top-m is idx for closest, the second top-m is idx for furthest
        for (int m = CEOs::top_m - 1; m >= 0; --m)
        {
            // Close:
            CEOs::matrix_H(m, iCol) = minQueClose.top().m_iIndex;
            minQueClose.pop();

            // Far:
            CEOs::matrix_H(m + CEOs::top_m, iCol) = minQueFar.top().m_iIndex;
            minQueFar.pop();
        }

        extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    double dSize = 1.0 * (CEOs::matrix_H.rows() * CEOs::matrix_H.cols() * sizeof(int) + CEOs::matrix_X.rows() * CEOs::matrix_X.cols() * sizeof(float) ) / (1 << 30);
    cout << "Size of CEOs-Hash index in GB: " << dSize << endl;

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count();
    cout << "Projecting time: " << projTime << " ms" << endl;
    cout << "Extracting top-points time: " << extractTopPointsTime << " ms" << endl;
    cout << "Constructing time (in seconds): " << (float)duration / 1000 << endl;
}


/**
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_CEOs_Hash(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (CEOs::n_probed_points > CEOs::top_m)
    {
        cerr << "Error: Number of probed points must be smaller than number of indexed top-points !" << endl;
        exit(1);
    }
    if (CEOs::n_probed_vectors > CEOs::n_proj * CEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than n_proj * n_repeats !" << endl;
        exit(1);
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_queries: " << n_queries << endl;
        cout << "n_probedVectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probedPoints: " << CEOs::n_probed_points << endl;
        // cout << "n_cand: " << CEOs::n_cand << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    RowMajorMatrixXi matTopK = MatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = MatrixXf::Zero(n_queries, n_neighbors);

    int log2_FHT = log2(CEOs::fhtDim);

    float candSize = 0.0;

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime, candSize)
    for (int q = 0; q < n_queries; ++q) {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        VectorXf vecProject = VectorXf(CEOs::n_proj * CEOs::n_repeats);

        // For each repeat
        for (int r = 0; r < CEOs::n_repeats; ++r)
        {
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * r;
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);

            rotatedQ.segment(0, CEOs::n_features) = vecQuery;

            for (int i = 0; i < CEOs::n_rotate; ++i)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(CEOs::bitHD[baseIdx + i * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedQ.segment(0, CEOs::n_proj); // only get up to #n_proj

        }

        // Now look for the top-r closest random vector
        priority_queue<IFPair, vector<IFPair>, greater<> > minQue;

        /**
         * matrix_H contains the point IDx
         * For query, we apply a simple trick to restore the furthest/closest vector
         * We increase index by 1 to get rid of the case of 0, and store a sign to differentiate the closest/furthest
         * Remember to convert this value back to the corresponding index of matrix_H
         * This fix is only for the rare case where r_0 at exp = 0 has been selected, which happen with very tiny probability
         */
        for (int d = 0; d < CEOs::n_repeats * CEOs::n_proj; ++d) {

            float fAbsProjValue = vecProject(d);

            // Must increase by 1 after getting the value
            int iCol = d + 1; // Hack: We increase by 1 since the index start from 0 and cannot have +/-

            if (fAbsProjValue < 0) {
                fAbsProjValue = -fAbsProjValue; // get abs
                iCol = -iCol; // use minus to indicate furthest vector
            }

            if ((int) minQue.size() < CEOs::n_probed_vectors)
                minQue.emplace(iCol, fAbsProjValue);

            // queue is full
            else if (fAbsProjValue > minQue.top().m_fValue) {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(iCol, fAbsProjValue);
            }
        }


        projTime += (float) chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        IVector vecRandIdx = IVector(CEOs::n_probed_vectors);

        // Get index of closest/furthest random vector from minQueue
        // We already increased by 1 to get rid of 0, and used the sign for closest/furthest
        for (int i = CEOs::n_probed_vectors - 1; i >= 0 ; --i)
        {
            IFPair ifPair = minQue.top(); // This contains Ri which is closest/furthest to the query q
            minQue.pop();

            vecRandIdx[i] = ifPair.m_iIndex; // pos = close, neg = far
        }

        // Find candidate whose has estimated dot product largest and set its bit in boost::bitset
        // Practical heuristic implementation: For each vector, get top-(n_cand / n_proj) - cache-friendly
        // Theory: Insert points with its projection value into a minQueue, and remember the points which has been added into the candidate
        // However, if n_cand is large, then the extra cost of using minQueue might make it less efficient than the practical implementation.
        // Hashing observation: Try to reduce the cost of getting candidate and spend more cost on dist computation

        // tsl::robin_set<int> setHist;
        // setHist.reserve(CEOs::n_probed_vectors * CEOs::n_probed_points);

        boost::dynamic_bitset<> bitsetHist(CEOs::n_points);

        // This implementation is more cache-efficient though has more or less n_cand points due to duplicates
        // Note that duplicate ratio is approximate 2 as we consider close & far
        // So non-duplicate candidate is ~ n_cand / 2
        // int bucketSize = ceil(1.0 * CEOs::n_cand / CEOs::n_probed_vectors);

        for (int i = 0; i < CEOs::n_probed_vectors; ++i)
        {
            int iCol = vecRandIdx[i];
            if (iCol > 0) // closest
            {
                iCol = iCol - 1; // get the right index col from matrix_P
                for (int m = 0; m < CEOs::n_probed_points; ++m) {
                    // setHist.insert(int(CEOs::matrix_H(m, iCol)));

                    if (~bitsetHist[CEOs::matrix_H(m, iCol)])
                        bitsetHist[CEOs::matrix_H(m, iCol)] = true; // set bit to true if not set
                }
            }
            else // furthest
            {
                iCol = -iCol - 1; // get the right index col from matrix_P
                for (int m = 0; m < CEOs::n_probed_points; ++m) {
                    // setHist.insert(int(CEOs::matrix_H(m + CEOs::top_m, iCol)));
                    if (~bitsetHist[CEOs::matrix_H(m + CEOs::top_m, iCol)])
                        bitsetHist[CEOs::matrix_H(m + CEOs::top_m, iCol)] = true; // set bit to true if not set
                }

            }
        }

        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        candSize += bitsetHist.count(); // setHist.size();

        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        // for (const auto& iPointIdx: setHist)
        int iPointIdx = bitsetHist.find_first();
        while (iPointIdx != (int)boost::dynamic_bitset<>::npos)
        {
            // Get dot product
            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
            }

            iPointIdx = bitsetHist.find_next(iPointIdx);
        }

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);


    if (verbose)
    {
        cout << "Projecting and extracting top-vectors time: " << projTime << " ms" << endl;
        cout << "Estimating time: " << estTime << " ms" << endl;
        cout << "Extracting candidates time: " << candTime << " ms" << endl;
        cout << "Avg candidate size : " << candSize / n_queries << endl;
        cout << "Computing distance time: " << distTime << " ms" << endl;
        cout << "Querying time: " << (float)durTime.count() << " ms" << endl;

        // string sFileName = "coCEOs_Hash_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_topPoints_" + int2str(CEOs::n_probed_points) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK, matTopDist);
}
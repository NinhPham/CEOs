//
// Created by npha145 on 22/09/24.
//

#include "coCEOs.h"
#include "Header.h"
#include "Utilities.h"

/**
 * Build index of coCEOs for estimating inner product
 *
 * @param matX: Note that CEOs::matrix_X has not been initialized, so we need to send param matX
 * Caveat: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders
 * While this approach allows fully optimized vectorized calculations in Eigen,
 * it cannot be used with array slices (i.e. when sending matX as an array slices from numpy)
 *
 * We can avoid duplicated memory by loading data directly from the filename if dataset is big.
 * This implementation computes matrix_P first, then reduce its size.
 * Though using more temporary memory but faster parallel, and perhaps cache-friendly
 *
 */
void coCEOs::build(const Ref<const Eigen::MatrixXf> &matX)
{
    cout << "n_features: " << coCEOs::n_features << endl;
    cout << "n_proj: " << coCEOs::n_proj << endl;
    cout << "iTopPoints: " << coCEOs::iTopPoints << endl;
    cout << "fhtDim: " << coCEOs::fhtDim << endl;
    cout << "centering: " << coCEOs::centering << endl;

    auto start = chrono::high_resolution_clock::now();

    coCEOs::n_points = matX.cols();
    cout << "n_points: " << coCEOs::n_points << endl;

    // NOTE: If n_points is large enough, then we can apply the centering heuristic to improve the hash accuracy
    // vec_centerX must not be updated in add_remove function.
    if (coCEOs::centering) // a heuristic threshold to determining centering the data
        coCEOs::vec_centerX = matX.rowwise().mean();

    // Not sure how to do this in multi-thread
    for (int n = 0; n < coCEOs::n_points; ++n)
    {
        if (coCEOs::centering)
            coCEOs::deque_X.emplace_back(matX.col(n) - coCEOs::vec_centerX);
        else
            coCEOs::deque_X.push_back(matX.col(n)); //emplace_back() causes error if calling with only matX.col(n)
    }

    float addTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count() / 1000.0;
    cout << "Copying data time (in ms): " << addTime << " ms" << endl;

    // coCEOs::matrix_P has (4 * top-points) x (proj * repeats) since query phase will access each column corresponding each random vector
    // We need 2 * top-points position for (index, value)
    // We need 1st and 2nd for closest vector, and 3rd and 4nd for furthest vectors
    coCEOs::matrix_P = MatrixXf::Zero(4 * coCEOs::iTopPoints, coCEOs::n_proj * coCEOs::n_repeats);

    // mat_pX is the projection matrix with (n_points) x (n_proj * repeats)
    MatrixXf mat_pX = MatrixXf::Zero(coCEOs::n_points, coCEOs::n_proj * coCEOs::n_repeats);

    coCEOs::bitGenerator(coCEOs::fhtDim, coCEOs::n_repeats); // only generate once while building the index

    float extractTopPointsTime = 0.0, projTime = 0.0;

    int log2_FWHT = log2(coCEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(coCEOs::n_threads);
#pragma omp parallel for reduction(+:projTime)
    for (int n = 0; n < coCEOs::n_points; ++n) {

        auto startTime = chrono::high_resolution_clock::now();

        VectorXf tempX = VectorXf::Zero(coCEOs::fhtDim); // fhtDim >= n_features since we pad 0 to have power of 2 format
        tempX.segment(0, coCEOs::n_features) = coCEOs::deque_X[n]; //matX.col(n); // coCEOs::deque_X[n];

        // For each exponent
        for (int r = 0; r < coCEOs::n_repeats; ++r) {

            // For each random rotation
            VectorXf rotatedX = tempX;

            for (int i = 0; i < coCEOs::n_rotate; ++i) {

                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = coCEOs::vecHD1[r];
                else if (i == 1)
                    randSign = coCEOs::vecHD2[r];
                else if (i == 2)
                    randSign = coCEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < coCEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FWHT);

            }

            // Note for segment(i, size) where i is starting index, size is segment size
            mat_pX.row(n).segment(coCEOs::n_proj * r + 0, coCEOs::n_proj) = rotatedX.segment(0, coCEOs::n_proj); // only get up to #n_proj
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }

#pragma omp parallel for reduction(+: extractTopPointsTime)
    for (int iCol = 0; iCol < coCEOs::n_repeats * coCEOs::n_proj; ++iCol)
    {
        // repeatIdx = iCol / coCEOs::n_proj;
        // projIdx = iCol % coCEOs::n_proj;

        auto startTime = chrono::high_resolution_clock::now();

        // Note that minQueFar has to deal with minus
        // These two minQueue are used to get top-points and will be dequeued later
        // NOTE: Use priQue to ensure extra space is O(topPoints) and time of O(n log(topPoints))
        // If we use heap of O(n) space, running time is O(n) to make heap and O(topPoint log(n))
        priority_queue<IFPair, vector<IFPair>, greater<> > minQueClose, minQueFar; // for closest & furthest

        VectorXf projectedVec = mat_pX.col(iCol);

        for (int n = 0; n < coCEOs::n_points; ++n){
            float fProjectedValue = projectedVec(n);

            // Closest points
            if ((int)minQueClose.size() < coCEOs::iTopPoints)
                minQueClose.emplace(n, fProjectedValue);
            else if (fProjectedValue > minQueClose.top().m_fValue)
            {
                minQueClose.pop();
                minQueClose.emplace(n, fProjectedValue);
            }

            // Furthest points: need to deal with -
            if ((int)minQueFar.size() < coCEOs::iTopPoints)
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
        for (int m = coCEOs::iTopPoints - 1; m >= 0; --m) // Ensure that matrix_P is sorted based on the projectedValue
        {
            // Close
            coCEOs::matrix_P(m, iCol) = minQueClose.top().m_iIndex;
            coCEOs::matrix_P(m + 1 * coCEOs::iTopPoints, iCol) = minQueClose.top().m_fValue;
            minQueClose.pop();

            // Far
            coCEOs::matrix_P(m + 2 * coCEOs::iTopPoints, iCol) = minQueFar.top().m_iIndex;
            coCEOs::matrix_P(m + 3 * coCEOs::iTopPoints, iCol) = minQueFar.top().m_fValue; // Store -projectedValue
            minQueFar.pop();
        }

        assert(minQueClose.size() == 0);
        assert(minQueFar.size() == 0);

        extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }

    double dSize = 1.0 * coCEOs::matrix_P.rows() * coCEOs::matrix_P.cols() * sizeof(float) / (1 << 30);
    dSize += 1.0 * coCEOs::deque_X.size() * coCEOs::deque_X[0].size() * sizeof(float) / (1 << 30);
    cout << "Size of coCEOs-Est index in GB: " << dSize << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Projecting time (in ms): " << projTime << " ms" << endl;
    cout << "Extracting top-points time (in ms): " << extractTopPointsTime << " ms" << endl;
    cout << "Constructing time (in seconds): " << (float)duration.count() << " seconds" << endl;
}

/**
 * Number of new points should be large enough such that curSize + newPoint - delPoint >= num-points
 * to ensure the data structure coCEOs has sufficient top-points for each dimension
 *
 * First, insert coCEOs points into the priority queue
 * Second, insert new points with pointIdx + n_points into the queue
 *
 * @param mat_newX : new data to be added
 * @param n_delPoints: number of points to be remove from the front (i.e. index 0 to n_delPoints - 1)
 *
 */
void coCEOs::update(const Ref<const Eigen::MatrixXf> & mat_newX, int n_delPoints)
{
    int n_newPoints = mat_newX.cols();
    int n_curSize = coCEOs::deque_X.size();

    if (n_delPoints > n_curSize)
    {
        cerr << "Error: Number of removed points must be smaller than the current number of points !" << endl;
        exit(1);
    }

    if (coCEOs::iTopPoints > n_curSize + n_newPoints - n_delPoints)
    {
        cerr << "Error: There is not enough indexed top-points for coCEOs after update !" << endl;
        exit(1);
    }

//    cout << "n_features: " << coCEOs::n_features << endl;
//    cout << "n_proj: " << coCEOs::n_proj << endl;
//    cout << "iTopPoints: " << coCEOs::iTopPoints << endl;
//    cout << "fhtDim: " << coCEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    for (int n = 0; n < n_newPoints; ++n)
    {
        if (coCEOs::centering)
            deque_X.emplace_back(mat_newX.col(n) - coCEOs::vec_centerX); // add new points into the back
        else
            deque_X.push_back(mat_newX.col(n)); // Note: emplace_back(mat_newX.col(n)) causes bug
    }

//    cout << "Size of databases after adding new points: " << deque_X.size() << endl;
    for (int n = 0; n < n_delPoints; ++n)
        deque_X.pop_front(); // remove old points from the front
//    cout << "Size of databases after removing old points: " << deque_X.size() << endl;

    coCEOs::n_points = deque_X.size();
    cout << "n_points (after update): " << coCEOs::n_points << endl;

    float durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count() / 1000.0;
    cout << "Updating data time (in ms): " << durTime << " ms" << endl;

    // mat_proj_newX is projection matrix of the new data, has (n_newPoints) x (proj * repeats)
    // Must reuse vecHD1, vecHD2, vecHD3 to update coCEOs
    MatrixXf mat_proj_newX = MatrixXf::Zero(n_newPoints, coCEOs::n_proj * coCEOs::n_repeats);
    int log2_FWHT = log2(coCEOs::fhtDim);

    float extractTopPointsTime = 0.0, projTime = 0.0;

    /**
     * Project new_X
     */
    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(coCEOs::n_threads);
#pragma omp parallel for reduction(+:projTime)
    for (int n = 0; n < n_newPoints; ++n) {

        auto startTime  = chrono::high_resolution_clock::now();

        VectorXf tempX = VectorXf::Zero(coCEOs::fhtDim);
//        tempX.segment(0, coCEOs::n_features) = mat_newX.col(n) - coCEOs::vec_centerX; // be careful when we center
        tempX.segment(0, coCEOs::n_features) = deque_X[n + n_curSize - n_delPoints]; // n --> n + curSize - n_delPoints, dequeX already centered

        // For each exponent
        for (int r = 0; r < coCEOs::n_repeats; ++r) {
            // For each random rotation
            VectorXf rotatedX = tempX;

            for (int i = 0; i < coCEOs::n_rotate; ++i) {

                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = coCEOs::vecHD1[r];
                else if (i == 1)
                    randSign = coCEOs::vecHD2[r];
                else if (i == 2)
                    randSign = coCEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < coCEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FWHT);

            }

            // Note for segment(i, size) where i is starting index, size is segment size
            mat_proj_newX.row(n).segment(coCEOs::n_proj * r + 0, coCEOs::n_proj) = rotatedX.segment(0, coCEOs::n_proj); // only get up to #n_proj
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }


    /**
     * For each dimension, remove old points from the current sorted data structure
     * Insert new points into the sorted list
     * Note: We still use the index from 0 to (n_points + n_newPoints)
     */
#pragma omp parallel for reduction(+:extractTopPointsTime)
    for (int iCol = 0; iCol < coCEOs::n_repeats * coCEOs::n_proj; ++iCol)
    {
        // repeatIdx = iCol / coCEOs::n_proj;
        // projIdx = iCol % coCEOs::n_proj
        auto startTime  = chrono::high_resolution_clock::now();

        // Create a sorted list to store close and far away
        priority_queue<IFPair, vector<IFPair>, greater<> > minQueClose, minQueFar; // for closest & furthest

        // Process old data: Since matrix_P is already sorted, we push_back()
        for (int m = 0; m < coCEOs::iTopPoints; ++m)
        {
            // Close list
            int pointIdx = coCEOs::matrix_P(m, iCol); // m + 0 * CEOs::indexBucketSize
            float fProjectedValue = coCEOs::matrix_P(m + 1 * coCEOs::iTopPoints, iCol);

            if (pointIdx >= n_delPoints) // since we remove pointIdx from 0 to (n_delPoints - 1)
                minQueClose.emplace(pointIdx, fProjectedValue);

            // Far list
            pointIdx = coCEOs::matrix_P(m + 2 * coCEOs::iTopPoints, iCol);
            fProjectedValue = coCEOs::matrix_P(m + 3 * coCEOs::iTopPoints, iCol); // Already store -projectedValue

            if (pointIdx >= n_delPoints) // since we remove pointIdx from 0 to (n_delPoints - 1)
                minQueFar.emplace(pointIdx, fProjectedValue); // Already store -projectedValue
        }

        assert(minQueClose.size() <= coCEOs::iTopPoints);
        assert(minQueFar.size() <= coCEOs::iTopPoints);

        // Process the new data
        VectorXf projectedVec = mat_proj_newX.col(iCol);
        for (int n = 0; n < n_newPoints; ++n)
        {
            float fProjectedValue = projectedVec(n);
            int newPointIdx = n + n_curSize; // new point idx from oldSize to the end, we will decrease after de-list

            // Insert into the sorted list and delete the smallest (last) element of the list if the size is larger than iTopPoints
            // https://cplusplus.com/reference/algorithm/upper_bound/

            // Close
            if ((int)minQueClose.size() < coCEOs::iTopPoints) // smaller
            {
                minQueClose.emplace(newPointIdx, fProjectedValue);
            }
            else if (fProjectedValue > minQueClose.top().m_fValue)
            {
                minQueClose.pop();
                minQueClose.emplace(newPointIdx, fProjectedValue);
            }

            // Far: need -fProjectedValue
            if ((int)minQueFar.size() < coCEOs::iTopPoints) // smaller
            {
                minQueFar.emplace(newPointIdx, -fProjectedValue); // insert
            }
            else if (-fProjectedValue > minQueFar.top().m_fValue)
            {
                minQueFar.pop();
                minQueFar.emplace(newPointIdx, -fProjectedValue);
            }
        }

        // Very rare case: we do not have enough iTopPoints
        // Safe query if iTopPoints >> probedPoints
        assert(minQueClose.size() <= coCEOs::iTopPoints);
        assert(minQueFar.size() <= coCEOs::iTopPoints);

        // Now iterate the list to store information into matrix_P
        for (int m = (int)minQueClose.size() - 1; m >= 0; --m) // in rare case where we do not have enough iTopPoints, missing points will have idx = 0
        {
            // Close
            coCEOs::matrix_P(m, iCol) = minQueClose.top().m_iIndex - n_delPoints; // To synchronize with the index of deque_X, starting from 0
            coCEOs::matrix_P(m + 1 * coCEOs::iTopPoints, iCol) = minQueClose.top().m_fValue;
            minQueClose.pop();
        }

        // Far
        for (int m = (int)minQueFar.size() - 1; m >= 0; --m) // in rare case where we do not have enough iTopPoints, missing points will have idx = 0
        {
            // Far
            coCEOs::matrix_P(m + 2 * coCEOs::iTopPoints, iCol) = minQueFar.top().m_iIndex - n_delPoints; // To synchronize with the index of deque_X, starting from 0
            coCEOs::matrix_P(m + 3 * coCEOs::iTopPoints, iCol) = minQueFar.top().m_fValue;
            minQueFar.pop();
        }

        extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        assert(minQueClose.size() == 0);
        assert(minQueFar.size() == 0);
    }

    double dSize = 1.0 * coCEOs::matrix_P.rows() * coCEOs::matrix_P.cols() * sizeof(float) / (1 << 30);
    dSize += 1.0 * coCEOs::deque_X.size() * coCEOs::deque_X[0].size() * sizeof(float) / (1 << 30);
    cout << "Size of coCEOs-Est index in GB: " << dSize << endl;

    durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count() / 1.0;

    cout << "Projecting time (in ms): " << projTime << " ms" << endl;
    cout << "Extracting top-points time (in ms): " <<  extractTopPointsTime << " ms" << endl;
    cout << "Updating time (in ms): " << durTime << " ms" << endl;
}

/**
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<MatrixXi, MatrixXf> coCEOs::estimate_search(const Ref<const MatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (coCEOs::n_probedPoints > coCEOs::iTopPoints)
    {
        cerr << "Error: Number of probed points must be smaller than indexed top-points !" << endl;
        exit(1);
    }

    if (coCEOs::n_probedVectors > coCEOs::n_proj)
    {
        cerr << "Error: Number of probed vectors must be smaller than number of projections !" << endl;
        exit(1);
    }

    int n_queries = matQ.cols();

    if (verbose)
    {
        cout << "n_queries: " << n_queries << endl;
        cout << "n_probedVectors: " << coCEOs::n_probedVectors << endl;
        cout << "n_cand: " << coCEOs::n_cand << endl;
        cout << "n_threads: " << coCEOs::n_threads << endl;
    }


    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(n_neighbors, n_queries);
    MatrixXf matTopDist = MatrixXf::Zero(n_neighbors, n_queries);

    int log2_FWHT = log2(coCEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(coCEOs::n_threads);
#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime)

    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.col(q);
        VectorXf vecProject = VectorXf (coCEOs::n_proj * coCEOs::n_repeats);

        // For each exponent
        for (int r = 0; r < coCEOs::n_repeats; ++r)
        {
            VectorXf rotatedQ = VectorXf::Zero(coCEOs::fhtDim);
            rotatedQ.segment(0, coCEOs::n_features) = vecQuery;

            for (int i = 0; i < coCEOs::n_rotate; ++i) {
                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = coCEOs::vecHD1[r];
                else if (i == 1)
                    randSign = coCEOs::vecHD2[r];
                else if (i == 2)
                    randSign = coCEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < coCEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedQ.data(), log2_FWHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(coCEOs::n_proj * r + 0, coCEOs::n_proj) = rotatedQ.segment(0, coCEOs::n_proj); // only get up to #n_proj

        }

        // Now look for the top-r closest random vector
        priority_queue< IFPair, vector<IFPair>, greater<> > minQue;

        /**
         * matrix_P contains the projection values, of size n x (e * n_proj)
         * For query, we apply a simple trick to restore the furthest/closest vector
         * We increase index by 1 to get rid of the case of 0, and store a sign to differentiate the closest/furthest
         * Remember to convert this value back to the corresponding index of matrix_P
         * This fix is only for the rare case where r_0 at exp = 0 has been selected, which happen with very tiny probability
         */

        // vecProject has n_proj * r columns
        // We cannot use one for loop as indexing since we need to separate between close and far
        // We add iCol by 1 to get rid of the case of index 0
        // and then use + for close, - for far
        for (int d = 0; d < coCEOs::n_repeats * coCEOs::n_proj; ++d)
        {
            float fAbsProjValue = vecProject(d);

            // Must increase by 1 after getting the value
            int iCol = d + 1; // Hack: We increase by 1 since the index start from 0 and cannot have +/-

            if (fAbsProjValue < 0) // We find top abs value (reflecting both closest and furthest)
            {
                fAbsProjValue = -fAbsProjValue; // get abs
                iCol = -iCol; // use minus to indicate furthest vector
            }

            if ((int)minQue.size() < coCEOs::n_probedVectors)
                minQue.emplace(iCol, fAbsProjValue);

            // queue is full
            else if (fAbsProjValue > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(iCol, fAbsProjValue);
            }
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        assert(minQue.size() == coCEOs::n_probedVectors);

        // Get top-r position out of n_proj * n_repeats and compute estimates
        startTime = chrono::high_resolution_clock::now();
        tsl::robin_map<int, float> mapEst;
        mapEst.reserve(coCEOs::n_probedVectors * coCEOs::n_probedPoints); // must reserve to increase the efficiency

        for (int i = 0; i < coCEOs::n_probedVectors; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();

            // Negative, means furthest away
            if (ifPair.m_iIndex < 0)
            {
                int iCol = -ifPair.m_iIndex - 1; // decrease by one due to the trick of increase by 1 and use the sign for closest/furthest
                for (int m = 0; m < coCEOs::n_probedPoints; ++m) // only consider up to probed_points
                {
                    int iPointIdx = int(coCEOs::matrix_P(m + 2 * coCEOs::iTopPoints, iCol));
                    float fValue = coCEOs::matrix_P(m + 3 * coCEOs::iTopPoints, iCol); // Already store -projectedValue

                    if (mapEst.find(iPointIdx) == mapEst.end())
                        mapEst[iPointIdx] = fValue;
                    else
                        mapEst[iPointIdx] += fValue;
                }
            }
            else
            {
                int iCol = ifPair.m_iIndex - 1; // decrease by one due to the trick of increase by 1 and use the sign for closest/furthest
                for (int m = 0; m < coCEOs::n_probedPoints; ++m) // only consider up to probed_points
                {
                    int iPointIdx = int(coCEOs::matrix_P(m, iCol));
                    float fValue = coCEOs::matrix_P(m + coCEOs::iTopPoints, iCol);

                    if (mapEst.find(iPointIdx) == mapEst.end())
                        mapEst[iPointIdx] = fValue;
                    else
                        mapEst[iPointIdx] += fValue;
                }
            }
        }

        estTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        assert(minQue.size() == 0);

        // If the cost of estTime and exact candidate is large with n = 10^9, then consider google dense map
        startTime = chrono::high_resolution_clock::now();

        // Only for probedPoints and probedBucket inputs
        for (auto& it: mapEst)
        {
            if ((int)minQue.size() < coCEOs::n_cand)
                minQue.emplace(it.first, it.second);

                // queue is full
            else if (it.second > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(it.first, it.second);
            }
        }

        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        assert(minQue.size() == coCEOs::n_cand);


        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;
        for (int i = 0; i < coCEOs::n_cand; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();
            int iPointIdx = ifPair.m_iIndex;

            float fInnerProduct = vecQuery.dot(coCEOs::deque_X[iPointIdx]);

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
            }
        }

        assert(minQue.size() == 0);

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(k, q) = minQueTopK.top().m_iIndex;
            matTopDist(k, q) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);


    if (verbose)
    {
        cout << "Projecting time: " << projTime << " ms" << endl;
        cout << "Estimating time: " << estTime << " ms" << endl;
        cout << "Extracting candidates time: " << candTime << " ms" << endl;
        cout << "Computing distance time: " << distTime << " ms" << endl;
        cout << "Querying time: " << (float)durTime.count() << " ms" << endl;

        string sFileName = "coCEOs_Est_" + int2str(n_neighbors) +
                           "_numProj_" + int2str(coCEOs::n_proj) +
                           "_numRepeat_" + int2str(coCEOs::n_repeats) +
                           "_topProj_" + int2str(coCEOs::n_probedVectors) +
                           "_topPoints_" + int2str(coCEOs::n_probedPoints) +
                           "_cand_" + int2str(n_cand) + ".txt";


        outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK.transpose(), matTopDist.transpose());
}

/**
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<MatrixXi, MatrixXf> coCEOs::hash_search(const Ref<const MatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (coCEOs::n_probedPoints > coCEOs::iTopPoints)
    {
        cerr << "Error: Number of probed points must be smaller than indexed top-points !" << endl;
        exit(1);
    }

    if (coCEOs::n_probedVectors > coCEOs::n_proj)
    {
        cerr << "Error: Number of probed vectors must be smaller than number of projections !" << endl;
        exit(1);
    }

    int n_queries = matQ.cols();
    if (verbose)
    {
        cout << "n_queries: " << n_queries << endl;
        cout << "n_probedVectors: " << coCEOs::n_probedVectors << endl;
        cout << "n_probedPoints: " << coCEOs::n_probedPoints << endl;
        cout << "n_cand: " << coCEOs::n_cand << endl;
        cout << "n_threads: " << coCEOs::n_threads << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(n_neighbors, n_queries);
    MatrixXf matTopDist = MatrixXf::Zero(n_neighbors, n_queries);

    int log2_FWHT = log2(coCEOs::fhtDim);
    float candSize = 0.0;

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(coCEOs::n_threads);
#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime, candSize)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.col(q);
        VectorXf vecProject = VectorXf(coCEOs::n_proj * coCEOs::n_repeats);

        // For each exponent
        for (int r = 0; r < coCEOs::n_repeats; ++r) {
            VectorXf rotatedQ = VectorXf::Zero(coCEOs::fhtDim);
            rotatedQ.segment(0, coCEOs::n_features) = vecQuery;

            for (int i = 0; i < coCEOs::n_rotate; ++i) {
                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = coCEOs::vecHD1[r];
                else if (i == 1)
                    randSign = coCEOs::vecHD2[r];
                else if (i == 2)
                    randSign = coCEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < coCEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedQ.data(), log2_FWHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(coCEOs::n_proj * r + 0, coCEOs::n_proj) = rotatedQ.segment(0,
                                                                                          coCEOs::n_proj); // only get up to #n_proj

        }

        /**
         * matrix_P contains the projection values, of size n x (n_repeats * n_proj)
         * For query, we apply a simple trick to restore the furthest/closest vector
         * We increase index by 1 to get rid of the case of 0, and store a sign to differentiate the closest/furthest
         * Remember to convert this value back to the corresponding index of matrix_P
         * This fix is only for the rare case where r_0 at exp = 0 has been selected, which happen with very tiny probability
         */
        priority_queue<IFPair, vector<IFPair>, greater<> > minQue;

        // vecProject has n_proj * r columns
        // We cannot use one for loop as indexing since we need to separate between close and far
        // We add iCol by 1 to get rid of the case of index 0
        // and then use + for close, - for far
        for (int d = 0; d < coCEOs::n_repeats * coCEOs::n_proj; ++d) {

            float fAbsProjValue = vecProject(d);

            // Must increase by 1 after getting the value
            int iCol = d + 1; // Hack: We increase by 1 since the index start from 0 and cannot have +/-

            if (fAbsProjValue < 0) // We find top abs value (reflecting both closest and furthest)
            {
                fAbsProjValue = -fAbsProjValue; // get abs
                iCol = -iCol; // use minus to indicate furthest vector
            }

            if ((int) minQue.size() < coCEOs::n_probedVectors)
                minQue.emplace(iCol, fAbsProjValue);

            // queue is full
            else if (fAbsProjValue > minQue.top().m_fValue) {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(iCol, fAbsProjValue);
            }
        }

        projTime += (float) chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        assert(minQue.size() == coCEOs::n_probedVectors);

        startTime = chrono::high_resolution_clock::now();
        IVector vecRandIdx = IVector(coCEOs::n_probedVectors);

        // Get index of closest/furthest random vector from minQueue
        // We already increased by 1 to get rid of 0, and used the sign for closest/furthest
        for (int i = coCEOs::n_probedVectors - 1; i >= 0 ; --i)
        {
            IFPair ifPair = minQue.top(); // This contains Ri which is closest/furthest to the query q
            minQue.pop();

            vecRandIdx[i] = ifPair.m_iIndex; // pos = close, neg = far
        }

        assert(minQue.size() == 0);

        // Find candidate whose has estimated dot product largest and set its bit in boost::bitset
        // Practical heuristic implementation: For each vector, get top-(n_cand / n_proj) - cache-friendly
        // Theory: Insert points with its projection value into a minQueue, and remember the points which has been added into the candidate
        // However, if n_cand is large, then the extra cost of using minQueue might makes it less efficient than the practical implementation.
        // Hashing observation: Try to reduce the cost of getting candidate and spend more cost on dist computation

        tsl::robin_set<int> setHist;
        setHist.reserve(coCEOs::n_cand);

        // This implementation ensures n_cand points to be computed distance
        // But not cache-friendly
//        bool bStop = false;
//        for (int m = 0; m < CEOs::indexBucketSize; ++m)
//        {
//            if (bStop)
//                break;
//
//            for (int i = 0; i < CEOs::n_probedBuckets; ++i)
//            {
//                //if ((int)bitsetHist.count() > CEOs::n_cand) // enough candidates
//                if ((int)setHist.size() > CEOs::n_cand)
//                {
//                    bStop = true;
//                    break;
//                }
//
//                int iCol = vecRandIdx[i];
//                if (iCol > 0) // closest
//                {
//                    iCol = iCol - 1; // get the right index col from matrix_P
//                    int iPointIdx = int(CEOs::matrix_H(m, iCol));
//                    setHist.insert(iPointIdx);
//                }
//                else // furthest
//                {
//                    iCol = -iCol - 1; // get the right index col from matrix_P
//                    int iPointIdx = int(CEOs::matrix_H(m + 1 * CEOs::indexBucketSize, iCol));
//                    setHist.insert(iPointIdx);
//                }
//            }
//        }

        // TODO: Might be a better approach to get top-cand from sorted projected value of each random vectors
        // This implementation is more cache-efficient though has more or lest n_cand points due to duplicates
        // Note that duplicate ratio is approximate 2 as we consider close & far
        // So non-duplicate candidate is ~ n_cand / 2
        int bucketSize = ceil(1.0 * coCEOs::n_cand / coCEOs::n_probedVectors);

        for (int i = 0; i < coCEOs::n_probedVectors; ++i)
        {
            int iCol = vecRandIdx[i];
            if (iCol > 0) // closest
            {
                iCol = iCol - 1; // get the right index col from matrix_P
                for (int m = 0; m < bucketSize; ++m) {
                    setHist.insert(int(coCEOs::matrix_P(m, iCol)));
                }
            }
            else // furthest
            {
                iCol = -iCol - 1; // get the right index col from matrix_P
                for (int m = 0; m < bucketSize; ++m) {
                    setHist.insert(int(coCEOs::matrix_P(m + 2 * coCEOs::iTopPoints, iCol)));
                }
            }
        }

        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        candSize += setHist.size();

        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        for (const auto& iPointIdx: setHist)
        {
            // Get dot product
            float fInnerProduct = vecQuery.dot(coCEOs::deque_X[iPointIdx]);

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
            matTopK(k, q) = minQueTopK.top().m_iIndex;
            matTopDist(k, q) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);


    if (verbose)
    {
        cout << "Projecting time: " << projTime << " ms" << endl;
        cout << "Estimating time: " << estTime << " ms" << endl;
        cout << "Extracting candidates time: " << candTime << " ms" << endl;
        cout << "Avg candidate size : " << candSize / n_queries << endl;
        cout << "Computing distance time: " << distTime << " ms" << endl;
        cout << "Querying time: " << (float)durTime.count() << " ms" << endl;

        string sFileName = "coCEOs_Hash_" + int2str(n_neighbors) +
                           "_numProj_" + int2str(coCEOs::n_proj) +
                           "_numRepeat_" + int2str(coCEOs::n_repeats) +
                           "_topProj_" + int2str(coCEOs::n_probedVectors) +
                           "_topPoints_" + int2str(coCEOs::n_probedPoints) +
                           "_cand_" + int2str(n_cand) + ".txt";


        outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK.transpose(), matTopDist.transpose());
}

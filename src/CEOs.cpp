
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
 */

void CEOs::build_CEOs(const Ref<const Eigen::MatrixXf> &matX)
{
    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    CEOs::matrix_X = matX;

    // Note that if fhtDim > n_proj, then we need to retrieve the first n_proj columns of the projections
    // This will save memory footprint if fhtDim is much larger than n_proj
    // We need N x (proj * exponent) since query phase will access each column corresponding each random vector
    CEOs::matrix_P = MatrixXf::Zero(CEOs::n_points, CEOs::n_proj * CEOs::n_repeats);
//    cout << CEOs::matrix_P.rows() << " " << CEOs::matrix_P.cols() << endl;

    CEOs::bitGenerator(CEOs::fhtDim, CEOs::n_repeats);

    int log2_FWHT = log2(CEOs::fhtDim);

     // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for
    for (int n = 0; n < CEOs::n_points; ++n) {
        VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);
        tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.col(n);

        // For each exponent
        for (int r = 0; r < CEOs::n_repeats; ++r) {

            VectorXf rotatedX = tempX;

            for (int i = 0; i < CEOs::n_rotate; ++i)
            {

                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = CEOs::vecHD1[r];
                else if (i == 1)
                    randSign = CEOs::vecHD2[r];
                else if (i == 2)
                    randSign = CEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FWHT);
            }

            // Store it into the matrix_P of size N x (exponent * n_project)
//            cout << CEOs::n_proj * r + 0 << " " << CEOs::n_proj * r + CEOs::n_proj << endl;
//            cout << rotatedX.segment(0, CEOs::n_proj).transpose() << endl;

            // Note for segment(i, size) where i is starting index, size is segment size
            CEOs::matrix_P.row(n).segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj); // only get up to #n_proj
//            CEOs::matrix_P.block(n, CEOs::n_proj * r + 0, 1, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj).transpose();
//            cout << matrix_P.row(n) << endl;
        }
    }

    cout << "Finish building index for estimating inner product. " << endl;


    double dSize = 1.0 * (CEOs::matrix_P.rows() * CEOs::matrix_P.cols() + CEOs::matrix_X.rows() * CEOs::matrix_X.cols()) * sizeof(float) / (1 << 30);
    cout << "Size of CEOs index in GB: " << dSize << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Construct CEOs-Est index Wall Time (in seconds): " << (float)duration.count() << " seconds" << endl;
}

/**
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<MatrixXi, MatrixXf> CEOs::search_CEOs(const Ref<const MatrixXf> & matQ, int n_neighbors, bool verbose)
{
    int n_queries = matQ.cols();

    if (verbose)
    {
        cout << "number of queries: " << n_queries << endl;
        cout << "top-project: " << CEOs::top_proj << endl;
        cout << "number of cand: " << CEOs::n_cand << endl;
        cout << "number of threads: " << CEOs::n_threads << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();
//    double itime = omp_get_wtime();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(n_neighbors, n_queries);
    MatrixXf matTopDist = MatrixXf::Zero(n_neighbors, n_queries);

    int log2_FWHT = log2(CEOs::fhtDim);
//    int log2_proj = log2(CEOs::n_proj);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime)

    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.col(q);
        VectorXf vecProject = VectorXf (CEOs::n_proj * CEOs::n_repeats);

        // For each exponent
        for (int r = 0; r < CEOs::n_repeats; ++r)
        {
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);
            rotatedQ.segment(0, CEOs::n_features) = vecQuery;

            for (int i = 0; i < CEOs::n_rotate; ++i) {
                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = CEOs::vecHD1[r];
                else if (i == 1)
                    randSign = CEOs::vecHD2[r];
                else if (i == 2)
                    randSign = CEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedQ.data(), log2_FWHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedQ.segment(0, CEOs::n_proj); // only get up to #n_proj

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
        for (int r = 0; r < CEOs::n_repeats; ++r)
        {
            // vecProject has n_proj * r columns
            for (int d = 0; d < CEOs::n_proj; ++d)
            {
                int iCol = CEOs::n_proj * r + d;
                float fAbsProjValue = vecProject(iCol);

                // Must increase by 1 after getting the value
                iCol = iCol + 1; // Hack: We increase by 1 since the index start from 0 and cannot have +/-

                if (fAbsProjValue < 0)
                {
                    fAbsProjValue = -fAbsProjValue; // get abs
                    iCol = -iCol; // use minus to indicate furthest vector
                }

                if ((int)minQue.size() < CEOs::top_proj)
                    minQue.emplace(iCol, fAbsProjValue);

                // queue is full
                else if (fAbsProjValue > minQue.top().m_fValue)
                {
                    minQue.pop(); // pop max, and push min hash distance
                    minQue.emplace(iCol, fAbsProjValue);
                }
            }
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Get top-r position out of n_proj * n_repeats and compute estimates
        /**
         * TODO: This step is not efficient with large number of threads, e.g. 32 threads with n = 1M
         * E.g. Cache size is not big enough to load an array of n elements for each thread
         * coCEOs is thread-friendly since we can use top-point and top-proj to control the cache-size for each thread
         */
        startTime = chrono::high_resolution_clock::now();
        VectorXf vecEst = VectorXf::Zero(CEOs::n_points);
        for (int i = 0; i < CEOs::top_proj; ++i)
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

            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.col(iPointIdx));

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
//    double exeTime = omp_get_wtime() - itime;

    if (verbose)
    {
        cout << "Projecting Time: " << projTime << " ms" << endl;
        cout << "Estimate Time: " << estTime << " ms" << endl;
        cout << "Extract Cand Time: " << candTime << " ms" << endl;
        cout << "Distance Time: " << distTime << " ms" << endl;
        cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;

        string sFileName = "CEOs_Est_" + int2str(n_neighbors) +
                           "_numProj_" + int2str(CEOs::n_proj) +
                           "_numRepeat_" + int2str(CEOs::n_repeats) +
                           "_topProj_" + int2str(CEOs::top_proj) +
                            "_cand_" + int2str(n_cand) + ".txt";


        outputFile(matTopK, sFileName);
     }

    return make_tuple(matTopK.transpose(), matTopDist.transpose());
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
void CEOs::build_coCEOs(const Ref<const Eigen::MatrixXf> &matX)
{
    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "top_points: " << CEOs::top_points << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    CEOs::matrix_X = matX;
    float addTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count() / 1000.0;
    cout << "Adding into matrix_X Wall Time (in ms): " << addTime << " ms" << endl;

    // CEOs::matrix_P has (4 * top-points) x (proj * repeats) since query phase will access each column corresponding each random vector
    // We need 2 * top-points position for (index, value)
    CEOs::matrix_P = MatrixXf::Zero(4 * CEOs::top_points, CEOs::n_proj * CEOs::n_repeats);

    // mat_pX is identical to CEOs estimation, has (n_points) x (proj * repeats)
    MatrixXf mat_pX = MatrixXf::Zero(CEOs::n_points, CEOs::n_proj * CEOs::n_repeats);

    CEOs::bitGenerator(CEOs::fhtDim, CEOs::n_repeats);

    float dequeTime = 0.0, projTime = 0.0;

    int log2_FWHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for reduction(+:projTime)
    for (int n = 0; n < CEOs::n_points; ++n) {

        auto startTime = chrono::high_resolution_clock::now();

        VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);
        tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.col(n);

        // For each exponent
        for (int r = 0; r < CEOs::n_repeats; ++r) {
            // For each random rotation
            VectorXf rotatedX = tempX;

            for (int i = 0; i < CEOs::n_rotate; ++i) {

                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = CEOs::vecHD1[r];
                else if (i == 1)
                    randSign = CEOs::vecHD2[r];
                else if (i == 2)
                    randSign = CEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FWHT);

            }

            // Note for segment(i, size) where i is starting index, size is segment size
            mat_pX.row(n).segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj); // only get up to #n_proj
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }



    for (int r = 0; r < CEOs::n_repeats; ++r) {

#pragma omp parallel for reduction(+: dequeTime)
        for (int d = 0; d < CEOs::n_proj; d++) {

            auto startTime = chrono::high_resolution_clock::now();

            // Note that minQueFar has to deal with minus
            priority_queue<IFPair, vector<IFPair>, greater<> > minQueClose, minQueFar; // for closest

            int iCol = CEOs::n_proj * r + d; // col idx of matrix project
            VectorXf projectedVec = mat_pX.col(iCol);

            for (int n = 0; n < CEOs::n_points; ++n){
                float fProjectedValue = projectedVec(n);

                // Closest points
                if ((int)minQueClose.size() < CEOs::top_points)
                    minQueClose.emplace(n, fProjectedValue);
                else if (fProjectedValue > minQueClose.top().m_fValue)
                {
                    minQueClose.pop();
                    minQueClose.emplace(n, fProjectedValue);
                }

                // Furthest points: need to deal with -
                if ((int)minQueFar.size() < CEOs::top_points)
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
            for (int m = CEOs::top_points - 1; m >= 0; --m)
            {
                // Close
                CEOs::matrix_P(m, iCol) = minQueClose.top().m_iIndex;
                CEOs::matrix_P(m + 1 * CEOs::top_points, iCol) = minQueClose.top().m_fValue;
                minQueClose.pop();

                // Far
                CEOs::matrix_P(m + 2 * CEOs::top_points, iCol) = minQueFar.top().m_iIndex;
                CEOs::matrix_P(m + 3 * CEOs::top_points, iCol) = -minQueFar.top().m_fValue; // we should store the correct projected value
                minQueFar.pop();
            }

            dequeTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        }
    }

    cout << "Finish building index for coCEOs. " << endl;


    double dSize = 1.0 * (CEOs::matrix_P.rows() * CEOs::matrix_P.cols() + CEOs::matrix_X.rows() * CEOs::matrix_X.cols()) * sizeof(float) / (1 << 30);
    cout << "Size of coCEOs index in GB: " << dSize << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "projTime coCEOs index Wall Time (in ms): " << projTime << " ms" << endl;
    cout << "dequeTime coCEOs index Wall Time (in ms): " << dequeTime << " ms" << endl;
    cout << "Construct coCEOs index Wall Time (in seconds): " << (float)duration.count() << " seconds" << endl;
}


/**
 * Number of new points should be larger than deleted points
 * to ensure the data structure coCEOs has sufficient top-points for each dimension
 *
 * First, insert coCEOs points into the priority queue
 * Second, insert new points with pointIdx + n_points into the queue
  *
 * @param mat_newX : new data to be added
  */
void CEOs::add_coCEOs(const Ref<const Eigen::MatrixXf> & mat_newX)
{
    int n_newPoints = mat_newX.cols();
    int n_curPoints = CEOs::n_points;

    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "top_points: " << CEOs::top_points << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    CEOs::matrix_X.conservativeResize(CEOs::n_features, n_curPoints + n_newPoints); // increase the size
    CEOs::matrix_X.rightCols(n_newPoints) = mat_newX; // add new data into the end
    CEOs::n_points = CEOs::matrix_X.cols(); // update n_points
    cout << "n_points: " << CEOs::n_points << endl;

    float durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count() / 1000.0;
    cout << "Update matrix X Wall Time (in ms): " << durTime << " ms" << endl;

    // mat_proj_newX is projection matrix of the new data, has (n_newPoints) x (proj * repeats)
    // Must reuse vecHD1, vecHD2, vecHD3 to update coCEOs
    MatrixXf mat_proj_newX = MatrixXf::Zero(n_newPoints, CEOs::n_proj * CEOs::n_repeats);
    int log2_FWHT = log2(CEOs::fhtDim);

    float dequeTime = 0.0, projTime = 0.0;

    /**
     * Project new_X
     */
    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for reduction(+:projTime)
    for (int n = 0; n < n_newPoints; ++n) {

        auto startTime  = chrono::high_resolution_clock::now();

        VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);
        tempX.segment(0, CEOs::n_features) = mat_newX.col(n);

        // For each exponent
        for (int r = 0; r < CEOs::n_repeats; ++r) {
            // For each random rotation
            VectorXf rotatedX = tempX;

            for (int i = 0; i < CEOs::n_rotate; ++i) {

                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = CEOs::vecHD1[r];
                else if (i == 1)
                    randSign = CEOs::vecHD2[r];
                else if (i == 2)
                    randSign = CEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FWHT);

            }

            // Note for segment(i, size) where i is starting index, size is segment size
            mat_proj_newX.row(n).segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj); // only get up to #n_proj
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    }


    /**
     * For each dimension, insert old points and then new points into priority queue
     * Only insert old points to queue if pointIdx > n_delPoints
     * We still use the index from 0 to (n_points + n_newPoints)
     */
    for (int r = 0; r < CEOs::n_repeats; ++r) {

        // mat_proj_newX has size (n_newPoints) x (proj * repeats)
        // parallel on dimension will not access the same memory
#pragma omp parallel for reduction(+:dequeTime)
        for (int d = 0; d < CEOs::n_proj; d++) {

            auto startTime  = chrono::high_resolution_clock::now();

            // Note that minQueFar has to deal with minus
            priority_queue<IFPair, vector<IFPair>, greater<> > minQueClose, minQueFar; // for closest

            int iCol = CEOs::n_proj * r + d; // col idx of matrix project

            // First we process all points in coCEOs matrix_P.
            // Since we only have top-points, we only insert into the queue
            for (int m = 0; m < CEOs::top_points; ++m)
            {
                // Close points
                int pointIdx = CEOs::matrix_P(m, iCol); // m + 0 * CEOs::top_points
                float fProjectedValue = CEOs::matrix_P(m + 1 * CEOs::top_points, iCol);

                minQueClose.emplace(pointIdx, fProjectedValue);

                // Far points
                pointIdx = CEOs::matrix_P(m + 2 * CEOs::top_points, iCol);
                fProjectedValue = CEOs::matrix_P(m + 3 * CEOs::top_points, iCol);

                minQueFar.emplace(pointIdx, -fProjectedValue); // remember to add minus
            }

            // Now we process the new data
            VectorXf projectedVec = mat_proj_newX.col(iCol);

            for (int n = 0; n < n_newPoints; ++n)
            {
                float fProjectedValue = projectedVec(n);

                // Note: we need to increase index from n --> n + n_oldPoints

                if (fProjectedValue > minQueClose.top().m_fValue)
                {
                    minQueClose.pop();
                    minQueClose.emplace(n + n_curPoints, fProjectedValue); // must change index of point
                }

                // Furthest points: need to deal with -
                if (-fProjectedValue > minQueFar.top().m_fValue) // since the minQue.size() = top_point
                {
                    minQueFar.pop();
                    minQueFar.emplace(n + n_curPoints, -fProjectedValue);
                }
            }

            assert(minQueClose.size() == CEOs::top_points);
            assert(minQueFar.size() == CEOs::top_points);

            // Now deque and store into matrix_P
            // We need to subtract the n_delPoints to synchronize with the index in matrix_X

            // the first top-point is idx, the second top-points is value for closest
            // the third top-point is idx, the fourth top-points is value for furthest
            for (int m = CEOs::top_points - 1; m >= 0; --m)
            {
                // Close
                CEOs::matrix_P(m, iCol) = minQueClose.top().m_iIndex; // To synchronize with the index of matrix_X
                CEOs::matrix_P(m + 1 * CEOs::top_points, iCol) = minQueClose.top().m_fValue;
                minQueClose.pop();

                // Far
                CEOs::matrix_P(m + 2 * CEOs::top_points, iCol) = minQueFar.top().m_iIndex; // To synchronize with the index of matrix_X
                CEOs::matrix_P(m + 3 * CEOs::top_points, iCol) = -minQueFar.top().m_fValue; // we should store the correct projected value
                minQueFar.pop();
            }

            dequeTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        }
    }

//    cout << "Finish updating index for coCEOs. " << endl;


    double dSize = 1.0 * (CEOs::matrix_P.rows() * CEOs::matrix_P.cols() + CEOs::matrix_X.rows() * CEOs::matrix_X.cols()) * sizeof(float) / (1 << 30);
    cout << "Size of coCEOs index in GB: " << dSize << endl;

    durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count() / 1.0;

    cout << "projectTime coCEOs index Wall Time (in ms): " << projTime << " ms" << endl;
    cout << "dequeTime coCEOs index Wall Time (in ms): " << dequeTime << " ms" << endl;
    cout << "Update coCEOs index Wall Time (in ms): " << durTime << " ms" << endl;
}

/**
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<MatrixXi, MatrixXf> CEOs::search_coCEOs(const Ref<const MatrixXf> & matQ, int n_neighbors, bool verbose)
{
    int n_queries = matQ.cols();

    if (verbose)
    {
        cout << "number of queries: " << n_queries << endl;
        cout << "top-project: " << CEOs::top_proj << endl;
        cout << "number of cand: " << CEOs::n_cand << endl;
        cout << "number of threads: " << CEOs::n_threads << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(n_neighbors, n_queries);
    MatrixXf matTopDist = MatrixXf::Zero(n_neighbors, n_queries);

    int log2_FWHT = log2(CEOs::fhtDim);
//    int log2_proj = log2(CEOs::n_proj);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);
#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime)

    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.col(q);
        VectorXf vecProject = VectorXf (CEOs::n_proj * CEOs::n_repeats);

        // For each exponent
        for (int r = 0; r < CEOs::n_repeats; ++r)
        {
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);
            rotatedQ.segment(0, CEOs::n_features) = vecQuery;

            for (int i = 0; i < CEOs::n_rotate; ++i) {
                // Multiply with random sign
                boost::dynamic_bitset<> randSign;
                if (i == 0)
                    randSign = CEOs::vecHD1[r];
                else if (i == 1)
                    randSign = CEOs::vecHD2[r];
                else if (i == 2)
                    randSign = CEOs::vecHD3[r];
                else {
                    cerr << "Error: Not support more than 3 random rotations !" << endl;
                    exit(1);
                }

                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(randSign[d]) - 1);
                }

                fht_float(rotatedQ.data(), log2_FWHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(CEOs::n_proj * r + 0, CEOs::n_proj) = rotatedQ.segment(0, CEOs::n_proj); // only get up to #n_proj

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
        for (int r = 0; r < CEOs::n_repeats; ++r)
        {
            // vecProject has n_proj * r columns
            for (int d = 0; d < CEOs::n_proj; ++d)
            {
                int iCol = CEOs::n_proj * r + d;
                float fAbsProjValue = vecProject(iCol);

                // Must increase by 1 after getting the value
                iCol = iCol + 1; // Hack: We increase by 1 since the index start from 0 and cannot have +/-

                if (fAbsProjValue < 0)
                {
                    fAbsProjValue = -fAbsProjValue; // get abs
                    iCol = -iCol; // use minus to indicate furthest vector
                }

                if ((int)minQue.size() < CEOs::top_proj)
                    minQue.emplace(iCol, fAbsProjValue);

                    // queue is full
                else if (fAbsProjValue > minQue.top().m_fValue)
                {
                    minQue.pop(); // pop max, and push min hash distance
                    minQue.emplace(iCol, fAbsProjValue);
                }
            }
        }

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Get top-r position out of n_proj * n_repeats and compute estimates
        startTime = chrono::high_resolution_clock::now();
        VectorXf vecEst = VectorXf::Zero(CEOs::n_points);
        for (int i = 0; i < CEOs::top_proj; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();

            // Negative, means furthest away
            if (ifPair.m_iIndex < 0)
            {
                int iCol = -ifPair.m_iIndex - 1;
                for (int m = 0; m < CEOs::top_points; ++m)
                {
//                    int iPointIdx = int(CEOs::matrix_P(m + 2 * CEOs::top_points, iCol));
//                    float fValue = CEOs::matrix_P(m + 3 * CEOs::top_points, iCol);
                    vecEst[int(CEOs::matrix_P(m + 2 * CEOs::top_points, iCol))] -= CEOs::matrix_P(m + 3 * CEOs::top_points, iCol);
                }
                    // m + 2 * CEOs::top_points: point Idx
                    // m + 3 * CEOs::top_points: projected value

            }
            else
            {
                int iCol = ifPair.m_iIndex - 1;
                for (int m = 0; m < CEOs::top_points; ++m)
                    // m + 2 * CEOs::top_points: point Idx
                    // m + 3 * CEOs::top_points: projected value
                    vecEst[int(CEOs::matrix_P(m, iCol))] += CEOs::matrix_P(m + CEOs::top_points, iCol);
            }
        }

        estTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // If the cost of estTime and exact candidate is large with n = 10^9, then consider google dense map
        startTime = chrono::high_resolution_clock::now();
        for (int n = 0; n < CEOs::n_points; ++n)
        {
            // Consider points with different 0 estimated inner product --> only useful if # n_proj * top_points << n
//            if (vecEst(n) == 0.0)
//                continue;

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

        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;
        for (int i = 0; i < CEOs::n_cand; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();
            int iPointIdx = ifPair.m_iIndex;

            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.col(iPointIdx));

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
        cout << "Projecting Time: " << projTime << " ms" << endl;
        cout << "Estimate Time: " << estTime << " ms" << endl;
        cout << "Extract Cand Time: " << candTime << " ms" << endl;
        cout << "Distance Time: " << distTime << " ms" << endl;
        cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;

        string sFileName = "coCEOs_Est_" + int2str(n_neighbors) +
                           "_numProj_" + int2str(CEOs::n_proj) +
                           "_numRepeat_" + int2str(CEOs::n_repeats) +
                           "_topProj_" + int2str(CEOs::top_proj) +
                            "_topPoints_" + int2str(CEOs::top_points) +
                           "_cand_" + int2str(n_cand) + ".txt";


        outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK.transpose(), matTopDist.transpose());
}


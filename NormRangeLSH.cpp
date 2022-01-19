<<<<<<< HEAD
#include "NormRangeLSH.h"
#include "Utilities.h"
#include "Header.h"

/**
Generate N(0, 1) from a normal distribution using C++ function
Generate [Prime] from a uniform distribution using C++ function
Output:
- DVector: vector contains normal random variables
- IVector: vector contains random integers
**/
void simHashGenerator()
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    normal_distribution<float> normDist(0.0, 1.0);

    // For SimHash
    int newD = PARAM_DATA_D + 1;
    int iKL = PARAM_LSH_NUM_TABLES * PARAM_LSH_NUM_HASH;

    MATRIX_G = MatrixXf::Zero(iKL, newD);

    // Generate N(0, 1)
    #pragma omp parallel for
    for (int d = 0; d < newD; ++d)
        for (int k = 0; k < iKL; ++k )
            MATRIX_G(k, d) = normDist(generator);
}

/**
Init LSH info (simHash) and construct hash tables
We need to scale the point X by the maximum norm 2 such that norm 2 of X is smaller than 1.

Output:
- Index VECTOR_LSH_TABLES of size L x 2^K x <IVector>
**/
void build_RangeLSH_Table()
{
    cout << "Constructing index..." << endl;

    // Init hash functions
    simHashGenerator();

    // TODO: Using a flat vector if knowing how to fix the bucket size
    // 1 table has 2^K buckets
    VECTOR_LSH_TABLES = vector<unordered_map<uint64_t, IVector>>(PARAM_LSH_NUM_TABLES);

    // Reserve space for hash table
    for (int l = 0; l < PARAM_LSH_NUM_TABLES; ++l)
        VECTOR_LSH_TABLES[l].reserve(1 * PARAM_DATA_N);

    // Precompute the maximum norm 2 to scale down X such that |X|< 1.
    vector<IFPair> vecSortedNorm(PARAM_DATA_N);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
        vecSortedNorm[n] = IFPair(n, MATRIX_X.col(n).norm()); // Matrix X is col-wise D x N

    // Descent sort point by its 2-norm X1 > X2 > ... > Xn
    sort(vecSortedNorm.begin(), vecSortedNorm.end(), greater<IFPair>());
    cout << "Max norm is: " << vecSortedNorm[0].m_fValue << endl;
    // printVector(vecSortedNorm);

    // Precompute number of points/tables on each partition
    int iNumPoints_PerPart = ceil((float)PARAM_DATA_N / PARAM_LSH_NUM_PARTITIONS);
    int iNumTables_PerPart = PARAM_LSH_NUM_TABLES / PARAM_LSH_NUM_PARTITIONS; // note #tables must be divisible #parts

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get pointIdx, start from point with maxNorm first
        int iPointIdx = vecSortedNorm[n].m_iIndex;

        // Get partition of the point
        int iPartIdx = n / iNumPoints_PerPart;

        // One point will be on [first, last] tables since it will be on a specific norm range.
        // First and last table indexes where the point locates
        int iFirstTable = iPartIdx * iNumTables_PerPart;
        int iLastTable = iFirstTable + (iNumTables_PerPart - 1); // [0..9], [10..19]

        // cout << "First Table: " << iFirstTable << " and Last Table: " << iLastTable << endl << endl;

        // Get maximum norm 2 of each partition from the beginning element of the partition
        // Hack: Plus EPSILON to make sure square of 1 - norm2(X) is always larger than 0
        float fMaxNorm = vecSortedNorm[iPartIdx * iNumPoints_PerPart].m_fValue + EPSILON;

        // Scale vector X
//        cout << "Before devided by maxNorm, norm is: " << MATRIX_X.col(iPointIdx).norm() << endl;
        // NOTE: if a point X is all-zero, then / fMaxNorm cause -Nan
        // There is the case where point's norm is larger than fMaxNorm due to the float issue.
        VectorXf vecPoint = MATRIX_X.col(iPointIdx) / fMaxNorm;

        // NOTE: we must use temp variable fTemp. Using directly vecPoint.squaredNorm() does not work
        // Note fTemp might be larger than 1.0 due to the float rounding issue
        float fTemp = vecPoint.squaredNorm();

        if (fTemp > 1.0)
        {
            cout << "There is a weird case where norm is larger than 1" << endl;
            printf("Norm of point is %f\n", fTemp);
        }

        // Add extra 1 dimensions
        vecPoint.conservativeResize(PARAM_DATA_D + 1, NoChange);

        // in case: (1) there are two points share the same norm, or (2) float cannot distinguish the tiny difference
        if (fTemp <= 1.0 - EPSILON)
            vecPoint(PARAM_DATA_D) = sqrt(1.0 - fTemp); // Note that fTemp is |x| / maxNorm^2 since dTemp is computed after scaling
        else
            vecPoint(PARAM_DATA_D) = 0.0;


        // Compute all SimHash values: Matrix G is KL x (D + 1)
        VectorXf vecProject = MATRIX_G * vecPoint; // of size KL x 1
        //cout << "Point: " << iPointIdx << ": " << endl;
        //cout << vecProject.head(5) << endl;;

        // Construct hash tables from iFirstTable to and including iLastTable
        for (int l = iFirstTable; l <= iLastTable; ++l)
        {
            //assert(l < PARAM_LSH_NUM_TABLES);

            uint64_t iHashValue = 0; // max = 2^K
            for (int k = 0; k < PARAM_LSH_NUM_HASH; ++k)
            {
                //int iIdx = l * PARAM_LSH_HASH_FUNCTIONS + k;

//                if (l == 0)
//                    cout << vecHash(iIdx) << " * " << VECTOR_LSH_UNIVERSAL_HASH[iIdx] << endl;

                // assert(l * PARAM_LSH_NUM_HASH + k < PARAM_LSH_NUM_TABLES * PARAM_LSH_NUM_HASH);

                if (vecProject(l * PARAM_LSH_NUM_HASH + k) > 0)
                    iHashValue |= 1UL << k; // add to 2^k = set bit at position k
            }

            //if (VECTOR_LSH_TABLES[l].find(iHashValue) != VECTOR_LSH_TABLES[l].end())
			#pragma omp critical // since VECTOR_LSH_TABLES[l][iHashValue] is access by several thread
            {
                VECTOR_LSH_TABLES[l][iHashValue].push_back(iPointIdx);
			}
            //else
            //{
                //VECTOR_LSH_TABLES[l].insert({iHashValue, IVector(1, iPointIdx)});
            //}
        }
    }

    cout << "Finish constructing index..." << endl;

    // Print number of buckets and bucket size for each hash table to see the skewness
//    for (int l = 0; l < PARAM_LSH_NUM_TABLES; ++l)
//    {
//        // Number of buckets
//        cout << "Number of buckets: " << VECTOR_LSH_TABLES[l].size()  << endl;
//
//        // printMap(VECTOR_LSH_TABLES[l]);
//
//        // Each bucket size
//        for (auto& it: VECTOR_LSH_TABLES[l])
//        {
//            if (it.second.size() > 0)
//                cout << it.second.size() << endl;
//        }
//
//        cout << endl;
//    }

}

/**
Query using RangeLSH Table

Input:
- MATRIX_G: for simhash

**/
void rangeLSH_Table_TopK()
{
    cout << "Start querying..." << endl;
    auto startTime = chrono::high_resolution_clock::now();
    float hashTime = 0.0, lookupTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);
    int iCandSize = 0, iUniqueCand = 0;

    #pragma omp parallel for reduction(+:hashTime, lookupTime, topKTime, iCandSize, iUniqueCand)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        unordered_set<int> setTopB;
        // Need to normalize and add extra dimension to use SimHash
        VectorXf vecQuery = MATRIX_Q.col(q).normalized();

        //printVector(vecQuery);
//        if (vecQuery.norm() != 1.0)
//            cout << "There is an error! Norm of query is: " << vecQuery.norm() << endl;

        // Add extra 1 dimension
        vecQuery.conservativeResize(PARAM_DATA_D + 1, NoChange);
        vecQuery(PARAM_DATA_D) = 0.0;

        // Compute all SimHash values
        VectorXf vecProject = MATRIX_G * vecQuery; // of size KL x 1

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        hashTime += (float)durTime.count() / 1000000;

        for (int l = 0; l < PARAM_LSH_NUM_TABLES; ++l)
        {
            auto startTime = chrono::high_resolution_clock::now();

            uint64_t iHashValue = 0;

            for (int k = 0; k < PARAM_LSH_NUM_HASH; ++k)
            {
                //int iIdx = l * PARAM_LSH_HASH_FUNCTIONS + k;

                if (vecProject(l * PARAM_LSH_NUM_HASH + k) > 0)
                    iHashValue |= 1UL << k;
            }

            auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
            hashTime += (float)durTime.count() / 1000000;

            startTime = chrono::high_resolution_clock::now();

            // Empty bucket or existing points in the bucket
            if (VECTOR_LSH_TABLES[l].find(iHashValue) != VECTOR_LSH_TABLES[l].end())
            {
                IVector vecBucket = VECTOR_LSH_TABLES[l][iHashValue];
                iCandSize += vecBucket.size();

                for (const auto& element: vecBucket)
                {
                    // insert into setTopB
                    // If already have B points, stop and return
                    if ((int)setTopB.size() < PARAM_MIPS_TOP_B)
                        setTopB.insert(element);
                    else // if we have enough candidates
                    {
                        if (PARAM_INTERNAL_SAVE_OUTPUT)
                            cout << "We have enough top-B candidates. We stop at the table " << l << endl;

                        l = PARAM_LSH_NUM_TABLES; // set this counter to quit the outer loop of tables
                        break;
                    }
                }
            }

            durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
            lookupTime += (float)durTime.count() / 1000000;
        }

        iUniqueCand += setTopB.size();

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------

        startTime = chrono::high_resolution_clock::now();
        extract_TopK_MIPS(MATRIX_Q.col(q), setTopB, PARAM_MIPS_TOP_K, matTopK.col(q));
        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    // Print time complexity of each step
    printf("Average number of unique candidates is %f \n", (float)iUniqueCand / PARAM_QUERY_Q);
    printf("Average number of candidates is %f \n", (float)iCandSize / PARAM_QUERY_Q);
    printf("Hash evaluation time is %f \n", hashTime);
    printf("Lookup time is %f \n", lookupTime);
    printf("Top K time is %f \n", topKTime);

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
    printf("Range LSH-Table query time is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string filename = "RangeLSH_Table_K_" + int2str(PARAM_LSH_NUM_HASH) +
                      "_L_" + int2str(PARAM_LSH_NUM_TABLES) + "_P_" + int2str(PARAM_LSH_NUM_PARTITIONS) +
                      "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";

        outputFile(matTopK, filename);
    }
}


/**
Construct Range LSH code

Output:
- vector of size N
**/
void build_RangeLSH_Code()
{
    // Init hash functions
    simHashGenerator();

    VECTOR_LSH_PARTITION_NORM = VectorXf::Zero(PARAM_DATA_N);

    // Each point has *PARAM_LSH_NUM_DECIMAL* uint_64t hash values
    VECTOR_LSH_CODES = vector<boost::dynamic_bitset<uint64_t>>(PARAM_DATA_N,
                                                       boost::dynamic_bitset<uint64_t>(PARAM_LSH_NUM_HASH));

    // Precompute the maximum norm 2 to scale down X such that |X|< 1.
    vector<IFPair> vecSortedNorm(PARAM_DATA_N);
    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
        vecSortedNorm[n] = IFPair(n, MATRIX_X.col(n).norm()); // Matrix X is col-wise D x N

    // Descent sort point by its 2-norm X1 > X2 > ... > Xn
    sort(vecSortedNorm.begin(), vecSortedNorm.end(), greater<IFPair>());
    cout << "Max norm is: " << vecSortedNorm[0].m_fValue << endl;
    // printVector(vecSortedNorm);

    int iNumPoints_PerPart = ceil((float)PARAM_DATA_N / PARAM_LSH_NUM_PARTITIONS);

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get pointIdx
        int iPointIdx = vecSortedNorm[n].m_iIndex;

        // Get partition
        int iPartIdx = n / iNumPoints_PerPart;

        // Get maximum norm 2 of each partition from the beginning element of partition
        // Hack: Plus EPSILON to make sure square of 1 - norm2(X) is always larger than 0
        float fMaxNorm = vecSortedNorm[iPartIdx * iNumPoints_PerPart].m_fValue;

        // store max Norm of partition for inner product estimation
        VECTOR_LSH_PARTITION_NORM(iPointIdx) = fMaxNorm;

        // Scale vector X
//        cout << "Before devided by maxNorm, norm is: " << MATRIX_X.col(iPointIdx).norm() << endl;
        VectorXf vecPoint = MATRIX_X.col(iPointIdx) / fMaxNorm;

//        cout << "After devided by maxNorm, norm is: " << vecRow.norm() << endl;
         float fTemp = vecPoint.squaredNorm();

         // Add extra 1 dimensions
        vecPoint.conservativeResize(PARAM_DATA_D + 1, NoChange);

        // in case: (1) there are two points share the same norm, or (2) float cannot distinguish the tiny difference
        if (fTemp <= 1.0 - EPSILON)
            vecPoint(PARAM_DATA_D) = sqrt(1.0 - fTemp); // Note that fTemp is |x| / maxNorm^2 since dTemp is computed after scaling
        else
            vecPoint(PARAM_DATA_D) = 0.0;

        // Compute all SimHash values
        VectorXf vecProject = MATRIX_G * vecPoint; // of size K*L x 1

        // convert to boost::bitset
        for (int k = 0; k < PARAM_LSH_NUM_HASH; ++k)
        {
            if (vecProject(k) > 0)
                VECTOR_LSH_CODES[iPointIdx][k] = 1; // NOTE: we iterate based on the sortedNorm
        }
    }
}

/**
Query using NormRange-LSH

Input:
- MATRIX_G: for simhash
- VECTOR_LSH_CODES: containing all data points' codes.

**/
void rangeLSH_Code_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float hashTime = 0.0, estTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    #pragma omp parallel for reduction(+:hashTime, estTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Need to normalize and add extra dimension to use SimHash
        VectorXf vecQuery = MATRIX_Q.col(q).normalized();

        //printVector(vecQuery);
//        if (vecQuery.norm() != 1.0)
//            cout << "There is an error! Norm of query is: " << vecQuery.norm() << endl;

        // Add extra 1 dimension
        vecQuery.conservativeResize(PARAM_DATA_D + 1, NoChange);
        vecQuery(PARAM_DATA_D) = 0.0;

        // Compute all SimHash values
        VectorXf vecProject = MATRIX_G * vecQuery; // of size K*L x 1

        // Compute hash values
        boost::dynamic_bitset<uint64_t> bitsetQuery(PARAM_LSH_NUM_HASH);
        for (int k = 0; k < PARAM_LSH_NUM_HASH; ++k)
        {
            if (vecProject(k) > 0)
                bitsetQuery[k] = 1;
        }

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        hashTime += (float)durTime.count() / 1000000;

        // Estimate inner products
        startTime = chrono::high_resolution_clock::now();

        VectorXf vecCounter = VectorXf::Zero(PARAM_DATA_N);
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            boost::dynamic_bitset<uint64_t> tempQuery(PARAM_LSH_NUM_HASH);
            tempQuery.operator= (bitsetQuery); //copy-- must use ()
            tempQuery.operator^=(VECTOR_LSH_CODES[n]); // XOR

            vecCounter(n) = tempQuery.count();

            // cout << "Point idx: " << n << " has value: " << vecCounter(n) << endl;

            if (PARAM_LSH_NUM_PARTITIONS == 1) // simpleLSH: points are all divided by the same const: fMaxNorm
                vecCounter(n) = PARAM_LSH_NUM_HASH - vecCounter(n);
            else // Use adjusted measure suggested in NIPS 18 paper: Uj cos (pi * (1 - eps) * (1 - l/L)
            {
                vecCounter(n) = PI *  (1 - EPSILON) * vecCounter(n) / PARAM_LSH_NUM_HASH;
                vecCounter(n) = hackCos(vecCounter(n)) * VECTOR_LSH_PARTITION_NORM(n);
            }

        }

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        estTime += (float)durTime.count() / 1000000;

        //cout << "Finish estimation step..." << endl;

        // Find topB-topK
        startTime = chrono::high_resolution_clock::now();

        extract_TopB_TopK_Histogram(vecCounter, MATRIX_Q.col(q), PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    // Print time complexity of each step
    printf("Hash evaluation time is %f \n", hashTime);
    printf("Estimation time is %f \n", estTime);
    printf("Top K time is %f \n", topKTime);

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
    printf("Range LSH-Code query time is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string filename = "RangeLSH_Code_K_" + int2str(PARAM_LSH_NUM_HASH) + "_P_" + int2str(PARAM_LSH_NUM_PARTITIONS)
                            + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";

        outputFile(matTopK, filename);
    }
}


/**
Construct SimpleLSH code without using any norm range.

Output:
- vector of size N
- vector of 2-norm of size N
**/
void build_SimHash_Code()
{
    // Init hash functions
    simHashGenerator();

    VECTOR_LSH_PARTITION_NORM = VectorXf::Zero(PARAM_DATA_N);

    VECTOR_LSH_CODES = vector<boost::dynamic_bitset<uint64_t>>(PARAM_DATA_N,
                                                       boost::dynamic_bitset<uint64_t>(PARAM_LSH_NUM_HASH));

    #pragma omp parallel for
    for (int n = 0; n < PARAM_DATA_N; ++n)
    {
        VectorXf vecPoint = MATRIX_X.col(n);

        // Precompute the maximum norm 2 to scale down X such that |X| = 1.
        VECTOR_LSH_PARTITION_NORM(n)= vecPoint.norm();

        // Scale vector X
//        cout << "Before devided by maxNorm, norm is: " << MATRIX_X.col(iPointIdx).norm() << endl;
        vecPoint = vecPoint / VECTOR_LSH_PARTITION_NORM(n);

//        cout << "After devided by maxNorm, norm is: " << vecRow.norm() << endl;
        // dTemp = vecRow.squaredNorm();

        // Add extra 1 dimensions
        vecPoint.conservativeResize(PARAM_DATA_D + 1, NoChange);
        vecPoint(PARAM_DATA_D) = 0.0; // We need to add 0 since the default filled value is not 0

        // Compute all SimHash values
        VectorXf vecProject = MATRIX_G * vecPoint; // of size K*L x 1

        // Compute binary code
        for (int k = 0; k < PARAM_LSH_NUM_HASH; ++k)
        {
            if (vecProject(k) > 0)
                VECTOR_LSH_CODES[n][k] = 1;
        }
    }
}

/**
Query using SimHash combined with 2-norm of X

Input:
- MATRIX_G: for simhash
- VECTOR_LSH_CODES: SimHash codes

**/
void simHash_Code_TopK()
{
    auto startTime = chrono::high_resolution_clock::now();
    float hashTime = 0.0, estTime = 0.0, topKTime = 0.0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    #pragma omp parallel for reduction(+:hashTime, estTime, topKTime)
    for (int q = 0; q < PARAM_QUERY_Q; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Need to normalize and add extra dimension to use SimHash
        VectorXf vecQuery = MATRIX_Q.col(q).normalized();

        //printVector(vecQuery);
//        if (vecQuery.norm() != 1.0)
//            cout << "There is an error! Norm of query is: " << vecQuery.norm() << endl;

        // Add extra 1 dimension
        vecQuery.conservativeResize(PARAM_DATA_D + 1, NoChange);
        vecQuery(PARAM_DATA_D) = 0.0;

        // Compute all SimHash values
        VectorXf vecProject = MATRIX_G * vecQuery; // of size K*L x 1

        // Compute hash values
        // Compute uint64_t decimal value
        boost::dynamic_bitset<uint64_t> bitsetQuery(PARAM_LSH_NUM_HASH);
        for (int k = 0; k < PARAM_LSH_NUM_HASH; ++k)
        {
            if (vecProject(k) > 0)
                bitsetQuery[k] = 1;
        }

        auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        hashTime += (float)durTime.count() / 1000000;

        // Estimate inner products
        startTime = chrono::high_resolution_clock::now();

        VectorXf vecCounter = VectorXf::Zero(PARAM_DATA_N);
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            boost::dynamic_bitset<uint64_t> tempQuery(PARAM_LSH_NUM_HASH);
            tempQuery.operator=(bitsetQuery); //copy-- must use ()

            tempQuery.operator^=(VECTOR_LSH_CODES[n]); // xor
            vecCounter(n) = tempQuery.count();

            vecCounter(n) = PI *  vecCounter(n) / PARAM_LSH_NUM_HASH;
            vecCounter(n) = hackCos(vecCounter(n)) * VECTOR_LSH_PARTITION_NORM(n);
        }

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        estTime += (float)durTime.count() / 1000000;

        // Find topB-topK
        startTime = chrono::high_resolution_clock::now();

        extract_TopB_TopK_Histogram(vecCounter, MATRIX_Q.col(q), PARAM_MIPS_TOP_B, PARAM_MIPS_TOP_K, matTopK.col(q));

        durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
        topKTime += (float)durTime.count() / 1000000;

    }

    // Print time complexity of each step
    printf("Hash evaluation time is %f \n", hashTime);
    printf("Estimation time is %f \n", estTime);
    printf("Top K time is %f \n", topKTime);

    auto durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime);
    printf("SimHash LSH-Code query time is %f \n", (float)durTime.count() / 1000000);

    if (PARAM_INTERNAL_SAVE_OUTPUT)
    {
        string filename = "SimHash_Code_K_" + int2str(PARAM_LSH_NUM_HASH) + "_Cand_" + int2str(PARAM_MIPS_TOP_B) + ".txt";

        outputFile(matTopK, filename);
    }
}
=======
#include "NormRangeLSH.h"
#include "Utilities.h"
#include "Header.h"

/**
Construct LSH code

Output:
- vector of size N
**/
void build_RangeLSH_Code()
{
    cout << "Number of decimal values: " << PARAM_INTERNAL_LSH_NUM_DECIMAL << endl;
    cout << "Number of partitions: " << PARAM_LSH_PARTITIONS << endl;
    cout << "Attention!!! Number of hash tables must be divisible by number of partition" << endl;
    cout << "Number of hash functions: " << PARAM_LSH_HASH_FUNCTIONS << endl;

    // Init hash functions
    simHashGenerator();

    VECTOR_LSH_PARTITION_NORM = DVector(PARAM_DATA_N, 0.0);

    // Each point has *PARAM_LSH_NUM_DECIMAL* uint_64t hash values
    VECTOR_LSH_CODES = I64Vector(PARAM_DATA_N * PARAM_INTERNAL_LSH_NUM_DECIMAL);

    int n, k, l, iMaxK;
    int iPointIdx, iPartIdx;
    uint64_t iHashValue;

    double dMaxNorm = 0.0;
    double dTemp = 0.0;

    VectorXd vecRow;
    VectorXd vecHash(PARAM_LSH_HASH_FUNCTIONS);

    vector<IDPair> vecSortedNorm(PARAM_DATA_N);

    // Precompute the maximum norm 2 to scale down X such that |X|< 1.
    for (n = 0; n < PARAM_DATA_N; ++n)
        vecSortedNorm[n] = IDPair(n, MATRIX_X.row(n).norm());

    // Descent sort point by its norm 2
    sort(vecSortedNorm.begin(), vecSortedNorm.end(), greater<IDPair>());

    cout << "Max norm is: " << vecSortedNorm[0].m_dValue << endl;
    // printVector(vecSortedNorm);

    int iSubDataSize = ceil((double)PARAM_DATA_N / (double)PARAM_LSH_PARTITIONS);

    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get pointIdx
        iPointIdx = vecSortedNorm[n].m_iIndex;

        // Get partition
        iPartIdx = round(n / iSubDataSize);

        // Get maximum norm 2 of each partition from the beginning element of partition
        // Hack: Plus EPSILON to make sure square of 1 - norm2(X) is always larger than 0
        dMaxNorm = vecSortedNorm[iPartIdx * iSubDataSize].m_dValue + EPSILON;

        // store max Norm of partition
        VECTOR_LSH_PARTITION_NORM[iPointIdx] = dMaxNorm;

        // Scale vector X
//        cout << "Before devided by maxNorm, norm is: " << MATRIX_X.col(iPointIdx).norm() << endl;
        vecRow = MATRIX_X.row(iPointIdx) / dMaxNorm;

//        cout << "After devided by maxNorm, norm is: " << vecRow.norm() << endl;
        dTemp = vecRow.squaredNorm();

//        cout << "After devided by maxNorm, squaredNorm is: " << dTemp << endl;

//        if (vecRow.norm() > 1.0)
//        {
//            cout << "There is an error. The norm of " << iPointIdx << " is: " << vecRow.norm() << endl;
//        }


        //cout << endl << "Before adding extra 1 dimensions " << endl;
        //printVector(vecRow);

        // Add extra 1 dimensions
        vecRow.conservativeResize(vecRow.rows() + 1, NoChange);
        vecRow(PARAM_DATA_D) = sqrt(1 - dTemp); // Note that dTemp is |x| / maxNorm^2 since dTemp is computed after scaling

//        if (vecRow.norm() > 1.0 || vecRow.norm() < 1.0)
//        {
//            cout << "There is an error after adding extra dimension. The norm of " << iPointIdx << " is: " << vecRow.norm() << endl;
//        }

        //cout << endl << "After adding extra 1 dimensions " << endl;
        //printVector(vecRow);

        // Compute all SimHash values
        vecHash = MATRIX_LSH_SIM_HASH * vecRow; // of size K*L x 1

        // Compute uint64_t decimal value
        for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
        {
            iHashValue = 0;
            iMaxK = min(64 * (l + 1), PARAM_LSH_HASH_FUNCTIONS);

            for (k = 64 * l; k < iMaxK; ++k) // We use 64 bits
            {
                if (vecHash(k) > 0)
                   iHashValue += pow(2, k % 64);
            }

            VECTOR_LSH_CODES[iPointIdx * PARAM_INTERNAL_LSH_NUM_DECIMAL + l] = iHashValue;
        }
    }
}

/**
Query using NormRange-LSH

Input:
- MATRIX_LSH_SIM_HASH: for simhash
- VECTOR_LSH_UNIVERSAL_HASH: for universal hashing

**/
void rangeLSH_Code_TopK()
{
    double dStart0 = clock();
    double dStart, dHashTime = 0, dTopKTime = 0, dLookupTime = 0;

    int q, k, n, l, iHamDist, iMaxK;

    uint64_t iHashValue;

    VectorXd vecQuery;
    VectorXd vecHash(PARAM_LSH_HASH_FUNCTIONS);

    IVector vecTopB;
    DVector vecCounter(PARAM_DATA_N); // Set with candidates already added.

    // Each query points has *PARAM_LSH_NUM_DECIMAL* decimal hash values
    I64Vector vecI64Hash(PARAM_INTERNAL_LSH_NUM_DECIMAL);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        // Need to normalize and add extra dimension to use SimHash
        vecQuery = MATRIX_Q.col(q).normalized();

        //printVector(vecQuery);
//        if (vecQuery.norm() != 1.0)
//            cout << "There is an error! Norm of query is: " << vecQuery.norm() << endl;

        // Add extra 1 dimension
        vecQuery.conservativeResize(vecQuery.rows() + 1, NoChange);
        vecQuery(PARAM_DATA_D) = 0.0;

        fill(vecCounter.begin(), vecCounter.end(), 0.0);
        fill(vecI64Hash.begin(), vecI64Hash.end(), 0);

        dStart = clock();

        // Compute all SimHash values
        vecHash = MATRIX_LSH_SIM_HASH * vecQuery; // of size K*L x 1

        // Compute hash values
        // Compute uint64_t decimal value
        for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
        {
            iHashValue = 0;
            iMaxK = min(64 * (l + 1), PARAM_LSH_HASH_FUNCTIONS);

            for (k = 64 * l; k < iMaxK; ++k) // We use 64 bits
            {
                if (vecHash(k) > 0)
                   iHashValue += pow(2, k % 64);
            }

            vecI64Hash[l] = iHashValue;
        }

        // Estimate inner products
        for (n = 0; n < PARAM_DATA_N; ++n)
        {
            iHamDist = 0;
            for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
                iHamDist += __builtin_popcountll(vecI64Hash[l] ^ VECTOR_LSH_CODES[n * PARAM_INTERNAL_LSH_NUM_DECIMAL + l]); // ^ XOR

            // Use adjusted measure suggested in NIPS 18 paper: Uj cos (pi * (1 - eps) * (1 - l/L)
            vecCounter[n] = PI *  (1 - EPSILON) * iHamDist / PARAM_LSH_HASH_FUNCTIONS;
            vecCounter[n] = hackCos(vecCounter[n]) * VECTOR_LSH_PARTITION_NORM[n];
        }

        dHashTime += clock() - dStart;

        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_TOP_B);

        dLookupTime += clock() - dStart;

        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "rangeLSH_Code_TopK_NoPost_" + int2str(q) + ".txt");
        */

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
            saveQueue(minQueTopK, "rangeLSH_Code_TopK_Post_" + int2str(q) + ".txt");
        }
    }

    // Print time complexity of each step
    printf("Hash evaluation time is %f \n", getCPUTime(dHashTime));
    printf("Top B time is %f \n", getCPUTime(dLookupTime));
    printf("Top K time is %f \n", getCPUTime(dTopKTime));

    printf("Range LSH: Code time is %f \n", getCPUTime(clock() - dStart0));
}

/**
Init LSH info (simHash and universalHash) and construct hash tables
We need to scale the point X by the maximum norm 2 such that norm 2 of X is smaller than 1.

Output:
- vector of size L x hashMap
**/
void build_RangeLSH_Table()
{
    int iNumTablesPerPart = ceil(PARAM_LSH_HASH_TABLES / PARAM_LSH_PARTITIONS);

    cout << "Number of hash tables: " << PARAM_LSH_HASH_TABLES << endl;
    cout << "Number of partitions: " << PARAM_LSH_PARTITIONS << endl;
    cout << "Attention!!! Number of hash tables must be divisible by number of partition" << endl;
    cout << "Number of hash tables for each partition: " << iNumTablesPerPart << endl;
    cout << "Number of hash functions: " << PARAM_LSH_HASH_FUNCTIONS << endl;


    int l, n, k;
    int iIdx, iPointIdx, iPartIdx;
    uint64_t iHashValue;

    // Init hash functions
    simHashGenerator();

    VECTOR_LSH_TABLES = vector<unordered_map<uint64_t, IVector>>(PARAM_LSH_HASH_TABLES);

    // Reserve space for hash table
    for (l = 0; l < PARAM_LSH_HASH_TABLES; ++l)
        VECTOR_LSH_TABLES[l].reserve(10 * PARAM_DATA_N);


    double dMaxNorm = 0.0;
    double dTemp = 0.0;

    VectorXd vecRow;
    VectorXd vecHash(PARAM_LSH_HASH_TABLES * PARAM_LSH_HASH_FUNCTIONS);

    vector<IDPair> vecSortedNorm(PARAM_DATA_N);

    // Precompute the maximum norm 2 to scale down X such that |X|< 1.
    for (n = 0; n < PARAM_DATA_N; ++n)
        vecSortedNorm[n] = IDPair(n, MATRIX_X.row(n).norm());

    // Descent sort point by its 2-norm X1 > X2 > ... > Xn
    sort(vecSortedNorm.begin(), vecSortedNorm.end(), greater<IDPair>());

    cout << "Max norm is: " << vecSortedNorm[0].m_dValue << endl;
    // printVector(vecSortedNorm);

    int iSubDataSize = ceil((double)PARAM_DATA_N / (double)PARAM_LSH_PARTITIONS);

    int iFirstTable = 0;
    int iLastTable = 0;

    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        // Get pointIdx
        iPointIdx = vecSortedNorm[n].m_iIndex;

        // Get partition
        iPartIdx = n / iSubDataSize;

        // First and last table indexes
        iFirstTable = iPartIdx * iNumTablesPerPart;
        iLastTable = iFirstTable + (iNumTablesPerPart - 1); // [0..9], [10..19]

        // cout << "First Table: " << iFirstTable << " and Last Table: " << iLastTable << endl << endl;

        // Get maximum norm 2 of each partition from the beginning element of partition
        // Hack: Plus EPSILON to make sure square of 1 - norm2(X) is always larger than 0
        dMaxNorm = vecSortedNorm[iPartIdx * iSubDataSize].m_dValue + EPSILON;

        // Scale vector X
//        cout << "Before devided by maxNorm, norm is: " << MATRIX_X.col(iPointIdx).norm() << endl;
        vecRow = MATRIX_X.row(iPointIdx) / dMaxNorm;

//        cout << "After devided by maxNorm, norm is: " << vecRow.norm() << endl;
        dTemp = vecRow.squaredNorm();

//        cout << "After devided by maxNorm, squaredNorm is: " << dTemp << endl;

//        if (vecRow.norm() > 1.0)
//        {
//            cout << "There is an error. The norm of " << iPointIdx << " is: " << vecRow.norm() << endl;
//        }


        //cout << endl << "Before adding extra 1 dimensions " << endl;
        //printVector(vecRow);

        // Add extra 1 dimensions
        vecRow.conservativeResize(vecRow.rows() + 1, NoChange);
        vecRow(PARAM_DATA_D) = sqrt(1 - dTemp); // Note that dTemp is |x| / maxNorm^2 since dTemp is computed after scaling

//        if (vecRow.norm() > 1.0 || vecRow.norm() < 1.0)
//        {
//            cout << "There is an error after adding extra dimension. The norm of " << iPointIdx << " is: " << vecRow.norm() << endl;
//        }

        //cout << endl << "After adding extra 1 dimensions " << endl;
        //printVector(vecRow);

        // Compute all SimHash values
        vecHash = MATRIX_LSH_SIM_HASH * vecRow; // of size 1 x K*L

        // Construct hash tables from iFirstTable to and including iLastTable
        for (l = iFirstTable; l <= iLastTable; ++l)
        {
            iHashValue = 0;
            for (k = 0; k < PARAM_LSH_HASH_FUNCTIONS; ++k)
            {
                iIdx = l * PARAM_LSH_HASH_FUNCTIONS + k;

//                if (l == 0)
//                    cout << vecHash(iIdx) << " * " << VECTOR_LSH_UNIVERSAL_HASH[iIdx] << endl;

                if (vecHash(iIdx) > 0)
                    iHashValue += VECTOR_LSH_UNIVERSAL_HASH[iIdx];
            }

            // iHashValue = iHashValue % PRIME;

            // if there is not exist the hash value, create new bucket and add new point
            // if there exists the hash value, insert new point
            VECTOR_LSH_TABLES[l][iHashValue].push_back(iPointIdx);
        }
    }


    // Print number of buckets and bucket size for each hash table to see the skewness
//    for (l = 0; l < PARAM_LSH_HASH_TABLES; ++l)
//    {
//        // Number of buckets
//        cout << "Number of buckets: " << VECTOR_LSH_TABLES[l].size()  << endl;
//
//        // printMap(VECTOR_LSH_TABLES[l]);
//
//        // Each bucket size
//        for (auto& it: VECTOR_LSH_TABLES[l])
//        {
//            if (it.second.size() > 0)
//                cout << it.second.size() << endl;
//        }
//
//        cout << endl;
//    }

}

/**
Query using Simple-LSH

Input:
- MATRIX_LSH_SIM_HASH: for simhash
- VECTOR_LSH_UNIVERSAL_HASH: for universal hashing

**/
void rangeLSH_Table_TopK()
{
    double dStart0 = clock();
    double dStart, dHashTime = 0, dTopKTime = 0, dLookupTime = 0, dAvgCand = 0.0;

    int q, l, k, n;
    int iIdx, iBucketSize;

    uint64_t iHashValue;

    VectorXd vecQuery;
    VectorXd vecHash(PARAM_LSH_HASH_FUNCTIONS * PARAM_LSH_HASH_TABLES);

    vector<int> vecBucket;
    unordered_set<int> setTopB;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        // Need to normalize and add extra dimension to use SimHash
        vecQuery = MATRIX_Q.col(q).normalized();

        //printVector(vecQuery);
//        if (vecQuery.norm() != 1.0)
//            cout << "There is an error! Norm of query is: " << vecQuery.norm() << endl;

        // Add extra 1 dimension
        vecQuery.conservativeResize(vecQuery.rows() + 1, NoChange);
        vecQuery(PARAM_DATA_D) = 0.0;

        setTopB.clear();

        dStart = clock();

        // Compute all SimHash values
        vecHash = MATRIX_LSH_SIM_HASH * vecQuery; // of size 1 x K*L

        dHashTime += clock() - dStart;

        iBucketSize = 0;

        for (l = 0; l < PARAM_LSH_HASH_TABLES; ++l)
        {
            // Compute hash values
            dStart = clock();
            iHashValue = 0;

            for (k = 0; k < PARAM_LSH_HASH_FUNCTIONS; ++k)
            {
                iIdx = l * PARAM_LSH_HASH_FUNCTIONS + k;

                if (vecHash(iIdx) > 0)
                    iHashValue += VECTOR_LSH_UNIVERSAL_HASH[iIdx];
            }

            // In case iHashValue get negative
            // iHashValue = iHashValue % PRIME;


            dHashTime += clock() - dStart;

            dStart = clock();

            // Empty bucket or exisiting points in the bucket
            if (VECTOR_LSH_TABLES[l].find(iHashValue) != VECTOR_LSH_TABLES[l].end())
            {
                vecBucket = VECTOR_LSH_TABLES[l][iHashValue];
                iBucketSize += vecBucket.size();

                for (n = 0; n < (int)vecBucket.size(); ++n)
                {
                    iIdx = vecBucket[n];

                    // insert into setTopB
                    // If already have B points, stop and return
                    if ((int)setTopB.size() < PARAM_MIPS_TOP_B)
                        setTopB.insert(iIdx);
                    else
                    {
                        if (PARAM_INTERNAL_SAVE_OUTPUT)
                            cout << "The number of checked hash tables is: " << l + 1 << endl;

                        l = PARAM_LSH_HASH_TABLES; //stop lookup
                        break;
                    }
                }
            }

            dLookupTime += clock() - dStart;
        }

        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            // cout << "The number of checked hash tables is: " << l + 1 << endl;
            cout << "The total number of points in checked buckets is: " << iBucketSize << endl;
        }

        dAvgCand += setTopB.size();

        // In case there is no collsion, continue to the next query
        if (setTopB.size() == 0)
            continue;

        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            saveSet(setTopB, "rangeLSH_Table_TopK_NoPost_" + int2str(q) + ".txt");
        }
        */

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(MATRIX_Q.col(q), setTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            printf("Number of inner product computation in LSH %d \n", (int)setTopB.size());
            saveQueue(minQueTopK, "rangeLSH_Table_TopK_Post_" + int2str(q) + ".txt");
        }
    }

    // Print time complexity of each step
    printf("Average number of candidates is %f \n", dAvgCand / PARAM_QUERY_Q);
    printf("Hash evaluation time is %f \n", getCPUTime(dHashTime));
    printf("Lookup time is %f \n", getCPUTime(dLookupTime));
    printf("Top K time is %f \n", getCPUTime(dTopKTime));

    printf("Range LSH: Table time is %f \n", getCPUTime(clock() - dStart0));
}




/**
Construct LSH code without using any norm range.

Output:
- vector of size N
- vector of 2-norm of size N
**/
void build_SimHash_Code()
{
    cout << "Number of decimal values: " << PARAM_INTERNAL_LSH_NUM_DECIMAL << endl;
    cout << "Number of hash functions: " << PARAM_LSH_HASH_FUNCTIONS << endl;

    // Init hash functions
    simHashGenerator();

    VECTOR_LSH_PARTITION_NORM = DVector(PARAM_DATA_N, 0.0);

    // Each point has *PARAM_LSH_NUM_DECIMAL* uint_64t hash values
    VECTOR_LSH_CODES = I64Vector(PARAM_DATA_N * PARAM_INTERNAL_LSH_NUM_DECIMAL);

    int n, k, l, iMaxK;
    uint64_t iHashValue;

    // double dTemp = 0.0;

    VectorXd vecRow;
    VectorXd vecHash(PARAM_LSH_HASH_FUNCTIONS);

    vector<IDPair> vecSortedNorm(PARAM_DATA_N);

    // Precompute the maximum norm 2 to scale down X such that |X| = 1.
    for (n = 0; n < PARAM_DATA_N; ++n)
        VECTOR_LSH_PARTITION_NORM[n] = MATRIX_X.row(n).norm();

    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        // Scale vector X
//        cout << "Before devided by maxNorm, norm is: " << MATRIX_X.col(iPointIdx).norm() << endl;
        vecRow = MATRIX_X.row(n) / VECTOR_LSH_PARTITION_NORM[n];

//        cout << "After devided by maxNorm, norm is: " << vecRow.norm() << endl;
        // dTemp = vecRow.squaredNorm();

        // Add extra 1 dimensions
        vecRow.conservativeResize(vecRow.rows() + 1, NoChange);
        vecRow(PARAM_DATA_D) = 0.0; // We need to add 0 since the default filled value is not 0

        // Compute all SimHash values
        vecHash = MATRIX_LSH_SIM_HASH * vecRow; // of size K*L x 1

        // Compute uint64_t decimal value
        for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
        {
            iHashValue = 0;
            iMaxK = min(64 * (l + 1), PARAM_LSH_HASH_FUNCTIONS);

            for (k = 64 * l; k < iMaxK; ++k) // We use 64 bits
            {
                if (vecHash(k) > 0)
                   iHashValue += pow(2, k % 64); // or set the kth bit: iHashValue |= 1UL << k;;
            }

            VECTOR_LSH_CODES[n * PARAM_INTERNAL_LSH_NUM_DECIMAL + l] = iHashValue;
        }
    }
}

/**
Query using Simple-LSH combined with 2-norm of X

Input:
- MATRIX_LSH_SIM_HASH: for simhash
- VECTOR_LSH_UNIVERSAL_HASH: for universal hashing

**/
void SimHash_Code_TopK()
{
    double dStart0 = clock();
    double dStart, dHashTime = 0, dTopKTime = 0, dLookupTime = 0;

    int q, k, n, l, iHamDist, iMaxK;

    uint64_t iHashValue;

    VectorXd vecQuery;
    VectorXd vecHash(PARAM_LSH_HASH_FUNCTIONS);

    IVector vecTopB;
    DVector vecCounter(PARAM_DATA_N); // Set with candidates already added.

    // Each query points has *PARAM_LSH_NUM_DECIMAL* decimal hash values
    I64Vector vecI64Hash(PARAM_INTERNAL_LSH_NUM_DECIMAL);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        // Need to normalize and add extra dimension to use SimHash
        vecQuery = MATRIX_Q.col(q).normalized();

        //printVector(vecQuery);
//        if (vecQuery.norm() != 1.0)
//            cout << "There is an error! Norm of query is: " << vecQuery.norm() << endl;

        // Add extra 1 dimension
        vecQuery.conservativeResize(vecQuery.rows() + 1, NoChange);
        vecQuery(PARAM_DATA_D) = 0.0;

        fill(vecCounter.begin(), vecCounter.end(), 0.0);
        fill(vecI64Hash.begin(), vecI64Hash.end(), 0);

        dStart = clock();

        // Compute all SimHash values
        vecHash = MATRIX_LSH_SIM_HASH * vecQuery; // of size K*L x 1

        // Compute hash values
        // Compute uint64_t decimal value
        for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
        {
            iHashValue = 0;
            iMaxK = min(64 * (l + 1), PARAM_LSH_HASH_FUNCTIONS);

            for (k = 64 * l; k < iMaxK; ++k) // We use 64 bits
            {
                if (vecHash(k) > 0)
                   iHashValue += pow(2, k % 64);
            }

            vecI64Hash[l] = iHashValue;
        }

        // Estimate inner products
        for (n = 0; n < PARAM_DATA_N; ++n)
        {
            iHamDist = 0;
            for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
                iHamDist += __builtin_popcountll(vecI64Hash[l] ^ VECTOR_LSH_CODES[n * PARAM_INTERNAL_LSH_NUM_DECIMAL + l]); // ^ XOR

            vecCounter[n] = PI * iHamDist / PARAM_LSH_HASH_FUNCTIONS;
            vecCounter[n] = hackCos(vecCounter[n]) * VECTOR_LSH_PARTITION_NORM[n];
        }

        dHashTime += clock() - dStart;

        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_TOP_B);

        dLookupTime += clock() - dStart;

        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "simHash_Code_TopK_NoPost_" + int2str(q) + ".txt");
        */

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
            saveQueue(minQueTopK, "simHash_Code_TopK_Post_" + int2str(q) + ".txt");
        }
    }

    // Print time complexity of each step
    printf("Hash evaluation time is %f \n", getCPUTime(dHashTime));
    printf("Top B time is %f \n", getCPUTime(dLookupTime));
    printf("Top K time is %f \n", getCPUTime(dTopKTime));

    printf("SimHash: Code time is %f \n", getCPUTime(clock() - dStart0));
}

>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

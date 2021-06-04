#include "SimpleLSH.h"
#include "Utilities.h"
#include "Header.h"

/**
Compute hash codes as uint64_t for every points.

Output:
- VECTOR_LSH_CODES: vector of size N

**/
void build_SimpleLSH_Code()
{
    // We use PARAM_LSH_HASH_TABLES as a number of uint64_t values for binary presentation.
    cout << "Number of decimal values: " << PARAM_INTERNAL_LSH_NUM_DECIMAL << endl;
    cout << "Number of hash functions: " << PARAM_LSH_HASH_FUNCTIONS << endl;

    // Init hash functions
    simHashGenerator();

    // Each point has *PARAM_LSH_NUM_DECIMAL* uint_64t hash values
    VECTOR_LSH_CODES = I64Vector(PARAM_DATA_N * PARAM_INTERNAL_LSH_NUM_DECIMAL);

    int n, k, l, iMaxK;
    uint64_t iHashValue;

    VectorXd vecRow;
    VectorXd vecRes(PARAM_LSH_HASH_FUNCTIONS);

    // Precompute the maximum norm 2 to scale down X such that |X|< 1.
    double dMaxNorm = 0.0;
    double dTemp = 0.0;
    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        dTemp = MATRIX_X.row(n).norm();
        if (dMaxNorm < dTemp)
            dMaxNorm = dTemp;
    }

    cout << "Max norm is: " << dMaxNorm << endl;

    // Hack to make sure sqrt() not return Nan
    dMaxNorm = dMaxNorm + EPSILON;

    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        // Scale vector X
        vecRow = MATRIX_X.row(n) / dMaxNorm;
        dTemp = vecRow.squaredNorm(); // need to keep it since when increase dimension by 1, weird value is added

//        if (vecRow.norm() > 1.0)
//        {
//            cout << "There is an error of norm. The norm of " << n << " is: " << vecCol.norm() << endl;
//        }


        //cout << endl << "Before adding extra 1 dimensions " << endl;
        //printVector(vecRow);

        // Add extra 1 dimensions
        vecRow.conservativeResize(vecRow.rows() + 1, NoChange);

//        if (dTemp > 1.0)
//        {
//            cout << endl << "There is an error of squaredNorm " << n << " with norm: " << vecRow.norm() << endl;
//            //printVector(vecRow);
//        }

        vecRow(PARAM_DATA_D) = sqrt(1 - dTemp); // Note that dTemp is |x| / maxNorm^2 since dTemp is computed after scaling

        // cout << endl << "After adding extra 1 dimensions " << endl;

//        if (vecRow.norm() > 1.0)
//        {
//            cout << endl << "There is an error after adding extra dimension of " << n << " with norm: " << vecRow.norm() << endl;
//            //printVector(vecRow);
//        }


        //printVector(vecRow);

        // Compute all SimHash values
        vecRes = MATRIX_LSH_SIM_HASH * vecRow; // of size K*L x 1

        // Compute uint64_t decimal value
        for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
        {
            iHashValue = 0;
            iMaxK = min(64 * (l + 1), PARAM_LSH_HASH_FUNCTIONS);
            for (k = 64 * l; k < iMaxK; ++k) // We use 64 bits
            {
                if (vecRes(k) > 0)
                   iHashValue += pow(2, k % 64);
            }

            VECTOR_LSH_CODES[n * PARAM_INTERNAL_LSH_NUM_DECIMAL + l] = iHashValue;
        }

    }
}

/**
Query using Simple-LSH

Input:
- MATRIX_LSH_SIM_HASH: for simhash
- VECTOR_LSH_UNIVERSAL_HASH: for universal hashing

**/
void simpleLSH_Code_TopK()
{
    double dStart0 = clock();
    double dStart, dHashTime = 0, dTopKTime = 0, dLookupTime = 0;

    int q, k, n, l, iMaxK;

    uint64_t iHashValue;

    VectorXd vecQuery;
    VectorXd vecHash(PARAM_LSH_HASH_FUNCTIONS);

    // Each query points has *PARAM_LSH_NUM_DECIMAL* decimal hash values
    I64Vector vecI64Hash(PARAM_INTERNAL_LSH_NUM_DECIMAL);

    IVector vecTopB, vecBucket;
    vector<int> vecCounter(PARAM_DATA_N);

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        // Need to normalize and add extra dimension to use SimHash
        vecQuery = MATRIX_Q.col(q).normalized();
        //printVector(vecQuery);
//        if (vecQuery.norm() > 1.0)
//            cout << "There is an error! Norm of query " << q << " is: " << vecQuery.norm() << endl;

        // Add extra 1 dimension
        vecQuery.conservativeResize(vecQuery.rows() + 1, NoChange);
        vecQuery(PARAM_DATA_D) = 0.0;

        fill(vecCounter.begin(), vecCounter.end(), 0);
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
            for (l = 0; l < PARAM_INTERNAL_LSH_NUM_DECIMAL; ++l)
                vecCounter[n] += __builtin_popcountll(vecI64Hash[l] ^ VECTOR_LSH_CODES[n * PARAM_INTERNAL_LSH_NUM_DECIMAL + l]); // ^ XOR

            vecCounter[n] = PARAM_LSH_HASH_FUNCTIONS - vecCounter[n]; // ^ XOR
        }


        dHashTime += clock() - dStart;

        // Find topB
        dStart = clock();

        vecTopB.clear();
        vecTopB = extract_SortedTopK_Histogram(vecCounter, PARAM_MIPS_TOP_B);

        dLookupTime += clock() - dStart;

        // Store no post-process result
        /*
        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveVector(vecTopB, "simpleLSH_Code_TopK_NoPost_" + int2str(q) + ".txt");
        */
        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------

        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(MATRIX_Q.col(q), vecTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;
        // printf("Computing TopK: Time is %f \n", getCPUTime(dTopKTime));

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            saveQueue(minQueTopK, "simpleLSH_Code_TopK_Post_" + int2str(q) + ".txt");
        }
    }


    // Print time complexity of each step
    printf("Hash evaluation time is %f \n", getCPUTime(dHashTime));
    printf("Top B time is %f \n", getCPUTime(dLookupTime));
    printf("Top K time is %f \n", getCPUTime(dTopKTime));

    printf("Simple LSH: Code time is %f \n", getCPUTime(clock() - dStart0));
}

/**
Init LSH info (simHash and universalHash) and construct hash tables
We need to scale the point X by the maximum norm 2 such that norm 2 of X is smaller than 1.

Output:
- vector of size L x hashMap
**/
void build_SimpleLSH_Table()
{
    cout << "Number of hash tables: " << PARAM_LSH_HASH_TABLES << endl;
    cout << "Number of hash functions: " << PARAM_LSH_HASH_FUNCTIONS << endl;

    int l, n, k, iIdx;
    uint64_t iHashValue;

    VectorXd vecRow;
    VectorXd vecRes(PARAM_LSH_HASH_TABLES * PARAM_LSH_HASH_FUNCTIONS);

    // Init hash functions
    simHashGenerator();

    VECTOR_LSH_TABLES = vector<unordered_map<uint64_t, IVector>>(PARAM_LSH_HASH_TABLES);

    // Reserve space for hash table
    for (l = 0; l < PARAM_LSH_HASH_TABLES; ++l)
        VECTOR_LSH_TABLES[l].reserve(10 * PARAM_DATA_N);


    // Precompute the maximum norm 2 to scale down X such that |X|< 1.
    double dMaxNorm = 0.0;
    double dTemp = 0.0;
    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        dTemp = MATRIX_X.row(n).norm();
        if (dMaxNorm < dTemp)
            dMaxNorm = dTemp;
    }

    cout << "Max norm is: " << dMaxNorm << endl;

    // Hack to make sure sqrt() not return Nan
    dMaxNorm = dMaxNorm + EPSILON;

    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        // Scale vector X
        vecRow = MATRIX_X.row(n) / dMaxNorm;
        dTemp = vecRow.squaredNorm(); // need to keep it since when increase dimension by 1, weird value is added

//        if (vecRow.norm() > 1.0)
//        {
//            cout << "There is an error of norm. The norm of " << n << " is: " << vecCol.norm() << endl;
//        }


        //cout << endl << "Before adding extra 1 dimensions " << endl;
        //printVector(vecRow);

        // Add extra 1 dimensions
        vecRow.conservativeResize(vecRow.rows() + 1, NoChange);

//        if (dTemp > 1.0)
//        {
//            cout << endl << "There is an error of squaredNorm " << n << " with norm: " << vecRow.norm() << endl;
//            //printVector(vecRow);
//        }

        vecRow(PARAM_DATA_D) = sqrt(1 - dTemp); // Note that dTemp is |x| / maxNorm^2 since dTemp is computed after scaling

        // cout << endl << "After adding extra 1 dimensions " << endl;

//        if (vecRow.norm() > 1.0)
//        {
//            cout << endl << "There is an error after adding extra dimension of " << n << " with norm: " << vecRow.norm() << endl;
//            //printVector(vecRow);
//        }


        //printVector(vecRow);

        // Compute all SimHash values
        vecRes = MATRIX_LSH_SIM_HASH * vecRow; // of size 1 x K*L

        for (l = 0; l < PARAM_LSH_HASH_TABLES; ++l)
        {
            iHashValue = 0;
            for (k = 0; k < PARAM_LSH_HASH_FUNCTIONS; ++k)
            {
                iIdx = l * PARAM_LSH_HASH_FUNCTIONS + k;

//                if (l == 0)
//                    cout << vecRes(iIdx) << " * " << VECTOR_LSH_UNIVERSAL_HASH[iIdx] << endl;

                // If simHash = 1, then add universal_hash
                // else simHash = 0, do nothing
                if (vecRes(iIdx) > 0)
                   iHashValue += VECTOR_LSH_UNIVERSAL_HASH[iIdx];
            }

            // Since universal_hash has all non-negative integer, no need to get abs()
            // iHashValue = iHashValue % PRIME;

            // if there is not exist the hash value, create new bucket and add new point
            // if there exists the hash value, insert new point
            VECTOR_LSH_TABLES[l][iHashValue].push_back(n);
        }
    }


    // Print number of buckets and bucket size for each hash table to see the skewness
//    for (l = 0; l < PARAM_LSH_HASH_TABLES; ++l)
//    {
//        // Number of buckets
//        cout << "Number of buckets: " << VECTOR_LSH_TABLES[l].size()  << endl << endl;
//
//        // printMap(PARAM_LSH_TABLES[l]);
//
//        // Each bucket size
//        for (auto& it: VECTOR_LSH_TABLES[l])
//        {
//            if (it.second.size() > 1000)
//                cout << it.second.size() << endl;
//        }
//    }

}

/**
Query using Simple-LSH

Input:
- MATRIX_LSH_SIM_HASH: for simhash
- VECTOR_LSH_UNIVERSAL_HASH: for universal hashing

**/
void simpleLSH_Table_TopK()
{
    double dStart0 = clock();
    double dStart, dHashTime = 0, dTopKTime = 0, dLookupTime = 0, dAvgCand = 0.0;

    int q, l, k, n;
    int iIdx, iBucketSize;

    uint64_t iHashValue;

    VectorXd vecQuery, vecNewQuery;
    VectorXd vecHash(PARAM_LSH_HASH_FUNCTIONS * PARAM_LSH_HASH_TABLES);

    IVector vecBucket;
    unordered_set<int> setTopB;

    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        // Need to normalize and add extra dimension to use SimHash
        vecQuery = MATRIX_Q.col(q);

        //printVector(vecQuery);
        vecNewQuery = vecQuery.normalized();
//        if (vecNewQuery.norm() > 1.0)
//            cout << "There is an error! Norm of query " << q << " is: " << vecNewQuery.norm() << endl;

        // Add extra 1 dimension
        vecNewQuery.conservativeResize(vecNewQuery.rows() + 1, NoChange);
        vecNewQuery(PARAM_DATA_D) = 0.0;

        dStart = clock();

        // Compute all SimHash values
        vecHash = MATRIX_LSH_SIM_HASH * vecNewQuery; // of size 1 x K*L

        dHashTime += clock() - dStart;

        iBucketSize = 0;
        setTopB.clear();

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

            // iHashValue = iHashValue % PRIME;

            dHashTime += clock() - dStart;

            dStart = clock();

            // Empty bucket or existing points in the bucket
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
            saveSet(setTopB, "simpleLSH_Table_TopK_NoPost_" + int2str(q) + ".txt");
        }
        */

        //----------------------------------
        // Dequeue and compute dot product then return topK
        //----------------------------------
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, setTopB, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        // Print out or save
        //printQueue(queTopK);
        if (PARAM_INTERNAL_SAVE_OUTPUT)
        {
            printf("Number of inner product computation in LSH %d \n", (int)setTopB.size());
            saveQueue(minQueTopK, "simpleLSH_Table_TopK_Post_" + int2str(q) + ".txt");
        }

    }


    // Print time complexity of each step
    printf("Average number of candidates is %f \n", dAvgCand / PARAM_QUERY_Q);
    printf("Hash evaluation time is %f \n", getCPUTime(dHashTime));
    printf("Lookup time is %f \n", getCPUTime(dLookupTime));
    printf("Top K time is %f \n", getCPUTime(dTopKTime));

    printf("Simple LSH: Table time is %f \n", getCPUTime(clock() - dStart0));
}



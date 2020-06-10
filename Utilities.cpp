#include "Utilities.h"
#include "Header.h"

#include <vector>
#include <queue>
#include <random>
#include <fstream> // fscanf, fopen, ofstream
#include <sstream>
#include <algorithm> // set_intersect(), lower_bound()
#include <unordered_map>
#include <unordered_set>


/**
Print a vector
**/
template<typename T>
void printVector(const vector<T> &vecPrint)
{
	cout << "Vector is: ";
	for (size_t i = 0; i < vecPrint.size(); ++i)
		cout << vecPrint[i] << " ";

	cout << endl;
}

/**
Print a vector
**/
void printVector(const VectorXd &vecPrint)
{
	cout << "Vector is: ";
	for (int i = 0; i < vecPrint.size(); ++i)
		cout << vecPrint(i) << " ";

    cout << endl;
}

/**
Print vector of IDPair
**/
void printVector(const vector<IDPair> & vecPrint)
{
	cout << "Vector is: ";
	for (size_t i = 0; i < vecPrint.size(); ++i)
		cout << "{ " << vecPrint[i].m_iIndex << " " << vecPrint[i].m_dValue << " }, ";

	cout << endl;
}

/**
Print unorder_map
**/
void printMap(const unordered_map<int, IVector> & printMap)
{
	for (auto const& pair: printMap)
    {
		cout << "{" << pair.first << ": ";
        for (auto &pointIdx: pair.second)
            cout << pointIdx << " ";
        cout <<  " }\n";
	}
}

/**
Print a priority queue of IDPair(index, value)
**/
void printQueue(priority_queue<IDPair, vector<IDPair>, greater<IDPair>> quePrint)
{
    IDPair P;
	while (!quePrint.empty())
{
        P = quePrint.top();
        cout << "(" << P.m_iIndex << ", " << P.m_dValue << ", " << ")" << endl;
        quePrint.pop();
    }
    cout << endl;
}

/**
Save ranking: [index value]
**/
void saveQueue(priority_queue<IDPair, vector<IDPair>, greater<IDPair>> quePrint, string fileName)
{
    IDPair P;

    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    while (!quePrint.empty())
    {
        P = quePrint.top();
        quePrint.pop();

        myfile << P.m_iIndex;
        myfile << " ";
        myfile << P.m_dValue << endl;
    }

    myfile.close();
}

/**
Save ranking: [index value]
**/
void saveQueue(priority_queue<IIPair, vector<IIPair>, greater<IIPair>> quePrint, string fileName)
{
    IIPair P;

    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    while (!quePrint.empty())
    {
        P = quePrint.top();
        quePrint.pop();

        myfile << P.m_iIndex;
        myfile << " ";
        myfile << P.m_iValue << endl;
    }

    myfile.close();
}

/**
Save ranking: [index value]
**/
void saveVector(const IVector &vecPrint, string fileName)
{
    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    for (size_t i = 0; i < vecPrint.size(); ++i)
    {
        myfile << i;
        myfile << " ";
        myfile << vecPrint[i] << endl;
    }

    myfile.close();
}

/**
Save ranking: [index value]
**/
void saveVector(const DVector &vecPrint, string fileName)
{
    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    for (size_t i = 0; i < vecPrint.size(); ++i)
    {
        myfile << i;
        myfile << " ";
        myfile << vecPrint[i] << endl;
    }

    myfile.close();
}

/**
Save ranking from set: [index 0.0]
**/
void saveSet(unordered_set<int> setPrint, string fileName)
{
    unordered_set<int>::iterator it;

    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    for (it = setPrint.begin(); it != setPrint.end(); ++it)
    {
        myfile << *it;
        myfile << " ";
        myfile << "0.0" << endl;
    }

    myfile.close();
}

/**
Generate random variables from a normal distribution using C++ function

Input:
- p_iNumSamples: number of samples

Output:
- DVector: vector contains normal random variables
- IVector: vector contains random integers

**/
void simHashGenerator()
{
    int s;

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    normal_distribution<double> normDist(0.0, 1.0);
    uniform_int_distribution<uint32_t> unifDist(0, PRIME);

    // For SimHash
    int newD = PARAM_DATA_D + 1;
    int iKL = PARAM_LSH_HASH_TABLES * PARAM_LSH_HASH_FUNCTIONS;

    DVector vecSimHash(iKL * newD);

    for (s = 0; s < iKL * newD; ++s)
         vecSimHash[s] = normDist(generator);

    //For Universal Hash
    VECTOR_LSH_UNIVERSAL_HASH = I32Vector(iKL);

    for (s = 0; s < iKL; ++s)
        VECTOR_LSH_UNIVERSAL_HASH[s] = unifDist(generator);

    MATRIX_LSH_SIM_HASH = Map<MatrixXd>(vecSimHash.data(), iKL, newD);
}

/**
Generate random variables from a discrete distribution with options

Input:
- VectorXd::vecCol: vector of column
- p_iOption: 1 (standard C++ generator), 2 (binary search suggested by Diamond), 3 (greedy Sampling)
- p_bGetSign: If TRUE - get both samples and its sign (default = FALSE)

Output:
- IVector: vector contains point indexes (with its sign)

**/
IVector preSampling(const VectorXd &vecCol, int p_iOption, bool p_bGetSign)
{
    IVector vecSamples;

    if (p_iOption == 1)
        vecSamples = discreteGenerator(vecCol, PARAM_MIPS_PRESAMPLES_MAX, p_bGetSign);

    else if (p_iOption == 2)
        vecSamples = binaryGenerator(vecCol, PARAM_MIPS_PRESAMPLES_MAX, p_bGetSign);

    else if (p_iOption == 3)
        vecSamples = greedyGenerator(vecCol, PARAM_MIPS_PRESAMPLES_MAX, p_bGetSign);

    else
        cout << "Error! Need to set the correct generator option. " << endl << endl;

    return vecSamples;
}


/**
Generate random variables from a discrete distribution using C++ function

Input:
- VectorXd::vecCol: vector presents discrete distribution (N x 1)
- p_iNumSamples: number of samples
- p_bGetSign: If TRUE - get both samples and its sign (default = FALSE)

Output:
- IVector: vector contains point indexes (with its sign)

**/
IVector discreteGenerator(const VectorXd &vecCol, int p_iNumSamples, bool p_bGetSign)
{
    int s, iPointIdx;
    VectorXd vecAbs = vecCol.cwiseAbs();

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    discrete_distribution<int> distribution(vecAbs.data(), vecAbs.data() + PARAM_DATA_N);

    IVector vecSamples(p_iNumSamples, 0);
    for (s = 0; s < p_iNumSamples; ++s)
    {
         iPointIdx = distribution(generator);

         if (p_bGetSign)
            iPointIdx = iPointIdx * sgn(vecCol(iPointIdx));

         vecSamples[s] = iPointIdx;
        //cout << vecSamples[s]<< " ";
    }
    return vecSamples;
}

/**
Generate random variables from a CDF of discrete distribution using binary search suggested by diamond sampling

Input:
- DVector::vecCDF: vector presents CDF of discrete distribution (N x 1)
- p_iNumSamples: number of samples
- p_bGetSign: If TRUE - get both samples and its sign (default = FALSE)

Output:
- IVector: vector contains point indexes (with its sign)

**/
IVector binaryGenerator(const VectorXd &vecCol, int p_iNumSamples, bool p_bGetSign)
{
    int s, iPointIdx;
    IVector vecSamples(p_iNumSamples, 0);
    vector<double>::iterator low;

    DVector vecCDF = CDF(vecCol);
    for (s = 0; s < p_iNumSamples; ++s)
    {
        low = lower_bound(vecCDF.begin(), vecCDF.end(), (double)(rand() + 1) / (RAND_MAX + 1)); // hack to not return 0
        iPointIdx = low - vecCDF.begin() - 1;

        if (p_bGetSign)
            iPointIdx = iPointIdx * sgn(vecCol(iPointIdx));

        vecSamples[s] = iPointIdx;
    }

    return vecSamples;
}

/**
Greedy based deterministic generator for random variables from a discrete distribution

Input:
- VectorXd::vecCol: vector presents discrete distribution (N x 1) for each dimension
- p_iNumSamples: number of samples
- p_bGetSign: If TRUE - get both samples and its sign (default = FALSE)

Output:
- IVector: vector contains point indexes (with its sign)

**/
IVector greedyGenerator(const VectorXd &vecCol, int p_iNumSamples, bool p_bGetSign)
{
    int n, s, iPointIdx;
    double dValue;

    // Make sure discrete distribution is nonnegative
    VectorXd vecAbs = vecCol.cwiseAbs();
    double dSum = vecAbs.sum();

    priority_queue<IDPair, vector<IDPair>> queFreq;
    for (n = 0; n < PARAM_DATA_N; ++n)
    {
        dValue = vecAbs(n) * PARAM_DATA_N / dSum;
        queFreq.push(IDPair(n, dValue));
    }

    // Greedy approach to get samples based on order
    IVector vecSamples(p_iNumSamples, 0);
    for (s = 0; s < p_iNumSamples; ++s)
    {
        iPointIdx = queFreq.top().m_iIndex;
        dValue = queFreq.top().m_dValue;

        queFreq.pop();
        queFreq.push(IDPair(iPointIdx, dValue - 1));

        if (p_bGetSign)
            iPointIdx = iPointIdx * sgn(vecCol(iPointIdx));

        vecSamples[s] = iPointIdx;

    }

    return vecSamples;
}

/** \brief Return top K from counting histogram
 *
 * \param
 *
 - vecCounter: counting histogram of 1 x N
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - vector<int> contains top K point indexes with largest values
 *
 */
IVector extract_SortedTopK_Histogram(const IVector &vecCounter, int p_iTopK)
{
    int n;

    priority_queue< IIPair, vector<IIPair>, greater<IIPair> > minQueTopK;

    // Insert into topK first
    // There might be the case of all values are  0
    for (n = 0; n < p_iTopK; ++n)
        minQueTopK.push(IIPair(n, vecCounter[n]));

    //  For the rest
    for (n = p_iTopK; n < PARAM_DATA_N; ++n)
    {
        if (vecCounter[n] > minQueTopK.top().m_iValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IIPair(n, vecCounter[n]));
        }
    }

    // The largest value should come first.
    IVector vecTopK(p_iTopK, 0);
    for (n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        vecTopK[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    return vecTopK;
 }

 /** \brief Return top K from counting histogram
 *
 * \param
 *
 - vecCounter: counting histogram of 1 x N
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - vector<int> contains top K point indexes with largest values
 - We need to sort based on the value of the histogram.
 *
 */
IVector extract_SortedTopK_Histogram(const DVector &vecCounter, int p_iTopK)
{
    int n;

    priority_queue< IDPair, vector<IDPair>, greater<IDPair> > minQueTopK;

    // Insert into topK first
    // There might be the case of all values are  0
    for (n = 0; n < p_iTopK; ++n)
        minQueTopK.push(IDPair(n, vecCounter[n]));

    //  For the rest
    for (n = p_iTopK; n < PARAM_DATA_N; ++n)
    {
        if (vecCounter[n] > minQueTopK.top().m_dValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IDPair(n, vecCounter[n]));
        }
    }

    // The largest value should come first.
    IVector vecTopK(p_iTopK, 0);
    for (n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        vecTopK[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    return vecTopK;
 }

 /** \brief Return top K from counting histogram
 *
 * \param
 *
 - vecCounter: counting histogram of 1 x N
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - vector<int> contains top K point indexes with largest values
 - We need to sort based on the value of the histogram.
 *
 */
IVector extract_SortedTopK_Histogram(const unordered_map<int, int> &mapCounter, int p_iTopK)
{
    int n;

    priority_queue< IDPair, vector<IIPair>, greater<IIPair> > minQueTopK;

    // iterate hashMap
    for (const auto& kv : mapCounter)
    {
        if (int(minQueTopK.size()) < p_iTopK)
            minQueTopK.push(IIPair(kv.first, kv.second));
        else if (kv.second > minQueTopK.top().m_iValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IIPair(kv.first, kv.second));
        }
    }

    // The largest value should come first.
    IVector vecTopK(p_iTopK, 0);
    for (n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        vecTopK[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    return vecTopK;
 }

 /** \brief Return top K from counting histogram
 *
 * \param
 *
 - vecCounter: counting histogram of 1 x N
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - vector<int> contains top K point indexes with largest values
 - We need to sort based on the value of the histogram.
 *
 */
IVector extract_SortedTopK_Histogram(const unordered_map<int, double> &mapCounter, int p_iTopK)
{
    int n;

    priority_queue< IDPair, vector<IDPair>, greater<IDPair> > minQueTopK;

    // iterate hashMap
    for (const auto& kv : mapCounter)
    {
        if (int(minQueTopK.size()) < p_iTopK)
            minQueTopK.push(IDPair(kv.first, kv.second));
        else if (kv.second > minQueTopK.top().m_dValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IDPair(kv.first, kv.second));
        }
    }

    // The largest value should come first.
    IVector vecTopK(p_iTopK, 0);
    for (n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        vecTopK[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    return vecTopK;
 }

 /** \brief Return top K from the vector top B for postprocessing step
 *
 * \param
 *
 - VectorXd::vecQuery: vector query
 - IVector vectopB: vector of Top B indexes
 - p_iTopK: top K MIPS
 *
 * \return
 *
 - priority queue<IDPair> contains point indexes and its dot product value
 - Note that minQue will be clear outside this function
 *
 */
void extract_TopK_MIPS(const VectorXd &vecQuery, const IVector &vecTopB, int p_iTopK,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &minQueTopK)
{
    double dValue;
    int n, iPointIdx;

    for (n = 0; n < (int)vecTopB.size(); ++n)
    {
        // Get point Idx
        iPointIdx = vecTopB[n];

        // Compute dot product
        dValue = MATRIX_X.row(iPointIdx) * vecQuery;

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IDPair(iPointIdx, dValue));
        else
        {
            // Insert into minQueue
            if (dValue > minQueTopK.top().m_dValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IDPair(iPointIdx, dValue));
            }
        }
    }
 }

 /** \brief Return top K from the unordered_set top B for postprocessing step
 *
 * \param
 *
 - VectorXd::vecQuery: vector query
 - IVector vectopB: vector of Top B indexes
 - p_iTopK: top K MIPS
 *
 * \return
 *
 - priority queue<IDPair> contains point indexes and its dot product value
 *
 */
void extract_TopK_MIPS(const VectorXd &vecQuery, const unordered_set<int> &setTopB, int p_iTopK,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &minQueTopK)
{
    double dValue;

    for (auto iPointIdx: setTopB)
    {
        //cout << pointIdx << endl;
        dValue = MATRIX_X.row(iPointIdx) * vecQuery;

        if (int(minQueTopK.size()) < p_iTopK)
            minQueTopK.push(IDPair(iPointIdx, dValue));
        else
        {
            if (dValue > minQueTopK.top().m_dValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IDPair(iPointIdx, dValue));
            }
        }
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
void computeWeight(const VectorXd &vecQuery, DVector &vecWeight)
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

 /** \brief Compute the vector weight (after shifting) contains fraction of samples for each dimension
 *
 * \param
 *
 - vecQuery: vector query of size 1 x D
 - C_POS_SUM, C_NEG_SUM: norm 1 of each column (1 x D)
 *
 * \return
 *
 - DVector: vector of normalized weight
 - p_dSum: norm 1 of vector weight
 *
 */
void computeShiftWeight(const VectorXd &vecQuery, DVector &vecWeight)
{
    double dValue = 0, dSum = 0;
    int d;

    for (d = 0; d < PARAM_DATA_D; ++d)
    {
        dValue = vecQuery(d);
        if (dValue >= 0.0)
            vecWeight[d] = POS_COL_NORM_1[d] * dValue;
        else
            vecWeight[d] = -NEG_COL_NORM_1[d] * dValue;

        dSum += vecWeight[d];
    }

    // Normalize weight
    for (d = 0; d < PARAM_DATA_D; ++d)
        vecWeight[d] = vecWeight[d] / dSum;
}

 /** \brief Get top K samples from pre-samples data
 *
 * \param
 *
 - d: dimension
 - p_iNumSamples: number of samples;
 - p_iSign: sign of query value at the dimension d
 *
 * \return
 *
 - vector<int> vecSamples (Note that vecSamples will be updated many times with this function)
 *
 */
void getDeterSamples(IVector &vecSamples, int d, int p_iNumSamples, int p_iSign)
{
    // Get samples from pre-sampling set
    vecSamples = IVector(p_iNumSamples, 0);

    if (p_iSign >= 0)
        copy(POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX,
             POS_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX + p_iNumSamples, vecSamples.begin());
    else
        copy(NEG_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX,
             NEG_PRESAMPLES.begin() + d * PARAM_MIPS_PRESAMPLES_MAX + p_iNumSamples, vecSamples.begin());
 }

 /**
 Compute shifting dot products
 **/
void computeShiftingDotProduct(const IVector &vecPoints, int q)
{
    int iPointIdx, d;
    double dValue, dDotProduct;
    VectorXd tempPoint;
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueue;

    VectorXd vecQuery = MATRIX_Q.col(q);

    for (int n = 0; n < (int)vecPoints.size(); ++n)
    {
        iPointIdx = vecPoints[n];

        dDotProduct = 0.0;
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
           dValue = vecQuery(d);
           if (dValue >= 0)
               dDotProduct += (MATRIX_X.col(iPointIdx)(d) - VECTOR_COL_MIN(d)) * dValue;
           else
               dDotProduct += (MATRIX_X.col(iPointIdx)(d) - VECTOR_COL_MAX(d)) * dValue;
        }

        minQueue.push(IDPair(iPointIdx, dDotProduct));
     }

    saveQueue(minQueue, "shiftWedgeTopKDotProduct_" + int2str(q) + ".txt");
 }

 /**
Generate cuumulative vectors with sum = 1

Input:
- VectorXd::vecCol: vector presents discrete distribution (N x 1)

Output:
- IVector: vector contains point indexes (with its sign)

**/
DVector CDF(const VectorXd &vecCol)
{
    VectorXd vecAbs = vecCol.cwiseAbs();
    double dValue = vecAbs.sum();

    int iNumElements = vecAbs.rows();
    DVector vecCDF(iNumElements, 0.0);

    for (int d = 0; d < iNumElements - 1; ++d)
        vecCDF[d + 1] = vecCDF[d] + vecAbs(d) / dValue;

    return vecCDF;
}


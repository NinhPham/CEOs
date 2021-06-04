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
void printVector(const vector<int> &vecPrint)
{
	cout << "Vector is: ";
	for (size_t i = 0; i < vecPrint.size(); ++i)
		cout << vecPrint[i] << " ";

	cout << endl;
}

/**
Print a vector
**/
void printVector(const vector<double> &vecPrint)
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
Save ranking: [index value]
**/
void saveVector(const VectorXd &vecPrint, string fileName)
{
    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    for (int i = 0; i < vecPrint.size(); ++i)
    {
        myfile << i;
        myfile << " ";
        myfile << vecPrint(i) << endl;
    }

    myfile.close();
}

/**
Save ranking: [index value]
**/
void saveVector(const VectorXi &vecPrint, string fileName)
{
    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    for (int i = 0; i < vecPrint.size(); ++i)
    {
        myfile << i;
        myfile << " ";
        myfile << vecPrint(i) << endl;
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
Save unorder_map
**/
void saveMap(unordered_map<int, double> mapPrint, string fileName)
{
    unordered_map<int, double>::iterator it;

    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    for (it = mapPrint.begin(); it != mapPrint.end(); ++it)
    {
        myfile << it->first;
        myfile << " ";
        myfile << it->second << endl;
    }
    myfile.close();
}

/**
Save unorder_map
**/
void saveMap(unordered_map<int, int> mapPrint, string fileName)
{
    unordered_map<int, int>::iterator it;

    ofstream myfile;
    myfile.open(fileName, std::ios::out);

    // Save
    for (it = mapPrint.begin(); it != mapPrint.end(); ++it)
    {
        myfile << it->first;
        myfile << " ";
        myfile << it->second << endl;
    }
    myfile.close();
}

/**
Generate N(0, 1) from a normal distribution using C++ function
Generate [Prime] from a uniform distribution using C++ function

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
Generate random N(0, 1) from a normal distribution using C++ function

Input:
- p_iNumRows x p_iNumCols: Row x Col

Output:
- DVector: vector contains normal random variables

**/
void gaussGenerator(int p_iNumRows, int p_iNumCols)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    normal_distribution<double> normDist(0.0, 1.0);

    int iSize = p_iNumRows * p_iNumCols;
    DVector vecNormal(iSize);

    for (int s = 0; s < iSize; ++s)
         vecNormal[s] = normDist(generator);

    MATRIX_NORMAL_DISTRIBUTION = Map<MatrixXd>(vecNormal.data(), p_iNumRows, p_iNumCols);
}

/**
Generate random {+1, -1} from a uniform distribution using C++ function

Input:
- p_iNumSamples: number of samples

Output:
- IVector: HD1, HD2, HD3 contains random {+1, -1}

**/
void HD3Generator(int p_iSize)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    uniform_int_distribution<uint32_t> unifDist(0, 1);

    HD1 = IVector(p_iSize);
    HD2 = IVector(p_iSize);
    HD3 = IVector(p_iSize);

    for (int s = 0; s < p_iSize; ++s)
    {
        HD1[s] = 2 * unifDist(generator) - 1;
        HD2[s] = 2 * unifDist(generator) - 1;
        HD3[s] = 2 * unifDist(generator) - 1;
    }
}

 /** \brief Return topK min index of a VectorXd
 *
 * \param
 *
 - VectorXd::vecCounter: of D x 1
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - VectorXi contains top K point indexes with largest values
 *
 */
VectorXi extract_TopK_MinIdx(const VectorXd &vecCounter, int p_iTopK)
{
    int n;
    double dTemp;

    VectorXi vecMinIdx(p_iTopK);
    priority_queue<IDPair> maxQueTopK; // take the max out, keep the min in queue

    //if (p_iTopK >= vecCounter.size())
        //cout << "Error! TopK mush be smaller than the vector size!" << endl;

    // Insert into topK first
    // There might be the case of all values are  0
    for (n = 0; n < p_iTopK; ++n)
    {
        maxQueTopK.push(IDPair(n, vecCounter(n)));
    }

    for (n = p_iTopK; n < vecCounter.size(); ++n)
    {
        dTemp = vecCounter(n);

        // For the min idx
        if (dTemp < maxQueTopK.top().m_dValue)
        {
            maxQueTopK.pop();
            maxQueTopK.push(IDPair(n, dTemp));
        }
    }

    // The largest value should come first.
    for (n = p_iTopK - 1; n >= 0; --n)
    {
        // Get min idx
        vecMinIdx(n) = maxQueTopK.top().m_iIndex;
        maxQueTopK.pop();
    }

    return vecMinIdx;
 }

 /** \brief Return topK max index of a VectorXd
 *
 * \param
 *
 - VectorXd::vecCounter: of D x 1
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - vector<int> contains top K point indexes with largest values
 *
 */
VectorXi extract_TopK_MaxIdx(const VectorXd &vecCounter,  int p_iTopK)
{
    int n;
    double dTemp;
    VectorXi vecMaxIdx(p_iTopK);

    priority_queue< IDPair, vector<IDPair>, greater<IDPair> > minQueTopK; // take the min out, keep the max in queue

    // Insert into topK first
    // There might be the case of all values are  0
    for (n = 0; n < p_iTopK; ++n)
        minQueTopK.push(IDPair(n, vecCounter(n)));

    //  For the rest
    for (n = p_iTopK; n < vecCounter.size(); ++n)
    {
        dTemp = vecCounter(n);

        // For the max idx
        if (dTemp > minQueTopK.top().m_dValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IDPair(n, dTemp));
        }
    }

    // The smallest value should come first.
    for (n = p_iTopK - 1; n >= 0; --n)
    {
        // Get max idx
        vecMaxIdx(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    return vecMaxIdx;
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
 - VectorXd::vecCounter: counting histogram of 1 x N
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - vector<int> contains top K point indexes with largest values
 - We need to sort based on the value of the histogram.
 *
 */
IVector extract_SortedTopK_Histogram(const VectorXd &vecCounter, int p_iTopK)
{
    int n;

    priority_queue< IDPair, vector<IDPair>, greater<IDPair> > minQueTopK;

    // Insert into topK first
    // There might be the case of all values are 0
    for (n = 0; n < p_iTopK; ++n)
        minQueTopK.push(IDPair(n, vecCounter(n)));

    //  For the rest
    for (n = p_iTopK; n < PARAM_DATA_N; ++n)
    {
        if (vecCounter[n] > minQueTopK.top().m_dValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IDPair(n, vecCounter(n)));
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
//        dValue = 0;
//        for (d = 0; d < PARAM_DATA_D; ++d)
//            dValue += MATRIX_X(iPointIdx, d) * vecQuery(d);

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
void extract_TopK_MIPS(const VectorXd &vecQuery, const Ref<VectorXi> vecTopB, int p_iTopK,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &minQueTopK)
{
    double dValue;
    int n, iPointIdx;

    for (n = 0; n < vecTopB.size(); ++n)
    {
        // Get point Idx
        iPointIdx = vecTopB(n);

        // Compute dot product
//        dValue = 0;
//        for (d = 0; d < PARAM_DATA_D; ++d)
//            dValue += MATRIX_X(iPointIdx, d) * vecQuery(d);

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
void extract_TopK_MIPS_Projected_X(const VectorXd &vecProjectedQuery, const IVector &vecTopB, int p_iTopK,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &minQueTopK)
{
    double dValue;
    int n, iPointIdx;

    for (n = 0; n < (int)vecTopB.size(); ++n)
    {
        // Get point Idx
        iPointIdx = vecTopB[n];

        // Compute dot product
        dValue = PROJECTED_X.row(iPointIdx) * vecProjectedQuery; // (N x upD ) * (upD x 1)

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




/**
    Convert a = a + b, b = a - b
**/
void inline wht_bfly (double& a, double& b)
{
    double tmp = a;
    a += b;
    b = tmp - b;
}

/**
    Fast in-place Walsh-Hadamard Transform (http://www.musicdsp.org/showone.php?id=18)
    also see (http://stackoverflow.com/questions/22733444/fast-sequency-ordered-walsh-hadamard-transform/22752430#22752430)
    - Note that the running time is exactly NlogN
**/
void FWHT (Ref<VectorXd> data, const IVector &p_vecHD)
{
    //printVector(data);

    int n = (int)data.size();
    int nlog2 = log2(n);

    int l, m;
    for (int i = 0; i < nlog2; ++i)
    {
        l = 1 << (i + 1);
        for (int j = 0; j < n; j += l)
        {
            m = 1 << i;
            for (int k = 0; k < m; ++k)
            {
                //cout << data (j + k) << endl;
                data (j + k) = data (j + k) * p_vecHD[j + k];
                //cout << data (j + k) << endl;

                //cout << data (j + k + m) << endl;
                data (j + k + m) = data (j + k + m) * p_vecHD[j + k + m];
                //cout << data (j + k + m) << endl;

                wht_bfly (data (j + k), data (j + k + m));
                //cout << data (j + k) << endl;
                //cout << data (j + k + m) << endl;

            }

        }
    }

    //printVector(data);
}



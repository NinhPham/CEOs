<<<<<<< HEAD
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

void printVector(const vector<int> & vecPrint)
{
	cout << "Vector is: ";
	for (const auto &element: vecPrint)
		cout << element << " ";

	cout << endl;
}

void printVector(const vector<IFPair> & vecPrint)
{
	cout << "Vector is: ";
	for (const auto &element: vecPrint)
		cout << "{ " << element.m_iIndex << " " << element.m_fValue << " }, ";

	cout << endl;
}

/**
Generate random N(0, 1) from a normal distribution using C++ function

Input:
- p_iNumRows x p_iNumCols: Row x Col

Output:
- MATRIX_G: vector contains normal random variables

**/
void gaussGenerator(int p_iNumRows, int p_iNumCols)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    normal_distribution<float> normDist(0.0, 1.0);

    MATRIX_G = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    // Always iterate col first, then row later due to the col-wise storage
    for (int c = 0; c < p_iNumCols; ++c)
        for (int r = 0; r < p_iNumRows; ++r)
            MATRIX_G(r, c) = normDist(generator);
}

/**
Generate random {+1, -1} from a uniform distribution using C++ function

Input:
- p_iNumSamples: number of samples

Output:
- MatrixXi: HD3 contains 3 random {+1, -1}

**/
void HD3Generator(int p_iSize)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    uniform_int_distribution<uint32_t> unifDist(0, 1);

    // We might use only one rotation
    HD3 = MatrixXi::Zero(p_iSize, 3);

    for (int s = 0; s < 3; ++s)
        for (int r = 0; r < p_iSize; ++r)
            HD3(r, s) = 2 * unifDist(generator) - 1;
}

 /** \brief Return topK min & max index of a VectorXd
 *
 * \param
 *
 - VectorXf::vecCounter: of D x 1
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - VectorXi:: min & max contains top K point indexes
 *
 */
void extract_TopK_MinMax_Idx(const Ref<VectorXf> &vecCounter, int p_iTopK, Ref<VectorXi> vecMinIdx, Ref<VectorXi> vecMaxIdx)
{
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK; // take the min out, keep the max in queue
    priority_queue< IFPair, vector<IFPair> > maxQueTopK;

    for (int n = 0; n < (int)vecCounter.size(); ++n)
    {
        float fTemp = vecCounter(n);

        // Get min out
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(n, fTemp));
        else if (fTemp > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(n, fTemp));
        }

        // Get max out
        if ((int)maxQueTopK.size() < p_iTopK)
            maxQueTopK.push(IFPair(n, fTemp));
        else if (fTemp < maxQueTopK.top().m_fValue)
        {
            maxQueTopK.pop();
            maxQueTopK.push(IFPair(n, fTemp));
        }
    }

    // The smallest/largerst value should come first.
    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get max idx
        vecMaxIdx(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();

        vecMinIdx(n) = maxQueTopK.top().m_iIndex;
        maxQueTopK.pop();
    }
 }

 /** \brief Compute topB and then B inner product to return topK
 *
 * \param
 *
 - vecCounter: Inner product estimation histogram of N x 1
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - VectorXi::p_vecTopK contains top K point indexes with largest values
 *
 */
void extract_TopB_TopK_Histogram(const Ref<VectorXf> & p_vecCounter, const Ref<VectorXf> &p_vecQuery,
                                 int p_iTopB, int p_iTopK, Ref<VectorXi> p_vecTopK)
{
    // Find topB
    assert((int)p_vecCounter.size() >= p_iTopB);

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    for (int n = 0; n < (int)p_vecCounter.size(); ++n)
    {
        float fTemp = p_vecCounter(n);

        if ((int)minQueTopK.size() < p_iTopB)
            minQueTopK.push(IFPair(n, fTemp));
        else if (fTemp > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(n, fTemp));
        }
    }

    // The largest value should come first.
    IVector vecTopB(p_iTopB);
    for (int n = p_iTopB - 1; n >= 0; --n)
    {
        // Get point index
        vecTopB[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    // Might need to sort it so that it would be cache-efficient when computing dot product
    // If B is small then might not have any effect
    // sort(vecTopB.begin(), vecTopB.end());

    // Find top-K
    //for (int n = 0; n < (int)vecTopB.size(); ++n)
    for (const auto & iPointIdx: vecTopB)
    {
        // Get point Idx
        // int iPointIdx = vecTopB[n];
        float fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(iPointIdx, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fValue));
            }
        }
    }


    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}

/** \brief Compute topB and then B inner product to return topK
 *
 * \param
 *
 - mapCounter: key = pointIdx, value = partialEst of Inner Product
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - VectorXi::p_vecTopK contains top K point indexes with largest values
 *
 */
void extract_TopB_TopK_Histogram(const unordered_map<int, float> &mapCounter, const Ref<VectorXf> &p_vecQuery,
                                 int p_iTopB, int p_iTopK, Ref<VectorXi> p_vecTopK)
{
    // Find topB
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    // cout << "Number of unique candidates: " << mapCounter.size() << endl;
    assert((int)mapCounter.size() >= p_iTopB);

    for (const auto& kv : mapCounter) // access via kv.first, kv.second
    {
        if ((int)minQueTopK.size() < p_iTopB)
            minQueTopK.push(IFPair(kv.first, kv.second));
        else if (kv.second > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(kv.first, kv.second));
        }
    }



    // The largest value should come first.
    IVector vecTopB(p_iTopB);
    for (int n = p_iTopB - 1; n >= 0; --n)
    {
        // Get point index
        vecTopB[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    // Might need to sort it so that it would be cache-efficient when computing dot product
    // If B is small then might not have any effect
    // sort(vecTopB.begin(), vecTopB.end());

    // Find top-K
    //for (int n = 0; n < (int)vecTopB.size(); ++n)
    for (const auto& iPointIdx: vecTopB)
    {
        // Get point Idx
        //int iPointIdx = vecTopB[n];
        float fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(iPointIdx, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fValue));
            }
        }
    }



    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}


 /** \brief Return top K from VectorXi top B for postprocessing step
 *
 * \param
 *
 - VectorXf::vecQuery: vector query
 - VectorXi vectopB: vector of Top B indexes
 - p_iTopK: top K MIPS
 *
 * \return
 *
 - VectorXi::p_vecTopK contains top-K point indexes
 *
 */
void extract_TopK_MIPS(const Ref<VectorXf> &p_vecQuery, const Ref<VectorXi>& p_vecTopB, int p_iTopK,
                 Ref<VectorXi> p_vecTopK)
{
    // incase we do not have enough candidates
    assert((int)p_vecTopB.size() >= p_iTopK);

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;


    //for (int n = 0; n < (int)p_vecTopB.size(); ++n)
    for (const auto& iPointIdx: p_vecTopB)
    {
        // Get point Idx
        //int iPointIdx = p_vecTopB(n);
        float fValue = 0.0;

        // This code is used for CEOs_TA; otherwise, we do not this condition
        if (PARAM_INTERNAL_NOT_STORE_MATRIX_X && PARAM_CEOs_NUM_ROTATIONS)
            // Now: p_vecQuery is the projected query and hence we use Project X of N x Dup
            // It will be slower than the standard case due to col-wise Project_X and D_up > D
            fValue = PROJECTED_X.row(iPointIdx) * p_vecQuery;
        else
            fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(iPointIdx, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fValue));
            }
        }
    }


    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}

/** \brief Return top K from unorder_set top B for postprocessing step
 *
 * \param
 *
 - VectorXf::vecQuery: vector query
 - VectorXi vectopB: vector of Top B indexes
 - p_iTopK: top K MIPS
 *
 * \return
 *
 - VectorXi::p_vecTopK contains top-K point indexes
 *
 */
void extract_TopK_MIPS(const Ref<VectorXf> &p_vecQuery, const unordered_set<int>& p_setTopB, int p_iTopK,
                 Ref<VectorXi> p_vecTopK)
{
    // incase we do not have enough candidates : LSH-Table
    //assert((int)p_setTopB.size() >= p_iTopK);
    // hack for the case of LSH table
    if (p_setTopB.empty())
        return;

    if ((int)p_setTopB.size() <= p_iTopK)
    {
        int n = 0;
        for (const auto& iPointIdx: p_setTopB)
        {
            p_vecTopK(n) = iPointIdx;
            n++;
        }

        // for the rest: accept the point 0 as the result
        return;
    }

    assert((int)p_setTopB.size() >= p_iTopK);

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    for (const auto& iPointIdx: p_setTopB)
    {
        float fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(iPointIdx, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fValue));
            }
        }
    }

    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}

/** \brief Return top K index from a vector
 *
 * \param
 *
 - VectorXf::vecQuery:
 - p_iTopK: top K MIPS
 *
 * \return
 *
 - VectorXi::vecTopK of top-k Index of largest value
 *
 */
void extract_max_TopK(const Ref<VectorXf> &p_vecQuery, int p_iTopK, Ref<VectorXi> p_vecTopK)
{
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    for (int n = 0; n < p_vecQuery.size(); ++n)
    {
        float fValue = p_vecQuery(n);

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(n, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(n, fValue));
            }
        }
    }

    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}

/**
    Convert a = a + b, b = a - b
**/
void inline wht_bfly (float& a, float& b)
{
    float tmp = a;
    a += b;
    b = tmp - b;
}

/**
    Fast in-place Walsh-Hadamard Transform (http://www.musicdsp.org/showone.php?id=18)
    also see (http://stackoverflow.com/questions/22733444/fast-sequency-ordered-walsh-hadamard-transform/22752430#22752430)
    - Note that the running time is exactly NlogN
**/
void FWHT (Ref<VectorXf> data)
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
                data (j + k) = data (j + k);
                //cout << data (j + k) << endl;

                //cout << data (j + k + m) << endl;
                data (j + k + m) = data (j + k + m);
                //cout << data (j + k + m) << endl;

                wht_bfly (data (j + k), data (j + k + m));
                //cout << data (j + k) << endl;
                //cout << data (j + k + m) << endl;

            }

        }
    }

    //printVector(data);
}

/**
Input:
(col-wise) matrix p_matKNN of size K x Q

Output: Q x K
- Each row is for each query
**/
void outputFile(const Ref<const MatrixXi> & p_matKNN, string p_sOutputFile)
{
	cout << "Outputing File..." << endl;
	ofstream myfile(p_sOutputFile);

	//cout << p_matKNN << endl;

	for (int j = 0; j < p_matKNN.cols(); ++j)
	{
        //cout << "Print col: " << i << endl;
		for (int i = 0; i < p_matKNN.rows(); ++i)
		{
            myfile << p_matKNN(i, j) << ' ';

		}
		myfile << '\n';
	}

	myfile.close();
	cout << "Done" << endl;
}
=======
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


>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

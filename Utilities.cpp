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

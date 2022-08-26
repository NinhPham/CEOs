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
void extract_TopB_TopK_Histogram(const Ref<VectorXf> & p_vecCounter, const Ref<VectorXf> &p_vecQuery, Ref<VectorXi> p_vecTopK)
{
    // Find topB
//    assert((int)p_vecCounter.size() >= p_iTopB);

    // Use to find topB first, then use to find topK later
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    for (int n = 0; n < (int)p_vecCounter.size(); ++n)
    {
        float fTemp = p_vecCounter(n);

        if ((int)minQueTopK.size() < PARAM_MIPS_TOP_B)
            minQueTopK.push(IFPair(n, fTemp));
        else if (fTemp > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(n, fTemp));
        }
    }

    // The largest value should come first.
    IVector vecTopB(PARAM_MIPS_TOP_B);
    for (int n = PARAM_MIPS_TOP_B - 1; n >= 0; --n)
    {
        // Get point index
        vecTopB[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    if (PARAM_MIPS_TOP_B == PARAM_MIPS_TOP_K)
    {
        p_vecTopK = Map<VectorXi>(vecTopB.data(), PARAM_MIPS_TOP_B);
        return;
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
        if ((int)minQueTopK.size() < PARAM_MIPS_TOP_K)
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

    for (int n = PARAM_MIPS_TOP_K - 1; n >= 0; --n)
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
void extract_TopB_TopK_Histogram(const unordered_map<int, float> &mapCounter, const Ref<VectorXf> &p_vecQuery, Ref<VectorXi> p_vecTopK)
{
    // Find topB first, then find topK later
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    // cout << "Number of unique candidates: " << mapCounter.size() << endl;
    assert((int)mapCounter.size() >= PARAM_MIPS_TOP_B);

    for (const auto& kv : mapCounter) // access via kv.first, kv.second
    {
        if ((int)minQueTopK.size() < PARAM_MIPS_TOP_B)
            minQueTopK.push(IFPair(kv.first, kv.second));
        else if (kv.second > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(kv.first, kv.second));
        }
    }


    // The largest value should come first.
    IVector vecTopB(PARAM_MIPS_TOP_B);
    for (int n = PARAM_MIPS_TOP_B - 1; n >= 0; --n)
    {
        // Get point index
        vecTopB[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    if (PARAM_MIPS_TOP_B == PARAM_MIPS_TOP_K)
    {
        p_vecTopK = Map<VectorXi>(vecTopB.data(), PARAM_MIPS_TOP_B);
        return;
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
        if ((int)minQueTopK.size() < PARAM_MIPS_TOP_K)
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

    for (int n = PARAM_MIPS_TOP_K - 1; n >= 0; --n)
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
void extract_TopK_MIPS(const Ref<VectorXf> &p_vecQuery, const Ref<VectorXi>& p_vecTopB, Ref<VectorXi> p_vecTopK)
{
    // incase we do not have enough candidates
//    assert((int)p_vecTopB.size() >= PARAM_MIPS_TOP_K);

    if (PARAM_MIPS_TOP_B == PARAM_MIPS_TOP_K)
    {
        p_vecTopK = p_vecTopB;
        return;
    }

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    //for (int n = 0; n < (int)p_vecTopB.size(); ++n)
    for (const auto& iPointIdx: p_vecTopB)
    {
        // Get point Idx
        //int iPointIdx = p_vecTopB(n);
        float fValue = 0.0;

        // This code is used for CEOs_TA; otherwise, we do not this condition
//        if (PARAM_INTERNAL_NOT_STORE_MATRIX_X)
//            // Now: p_vecQuery is the projected query and hence we use Project X of N x Dup
//            // It will be slower than the standard case due to col-wise Project_X and D_up > D
//            fValue = PROJECTED_X.row(iPointIdx) * p_vecQuery;
//        else

        fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < PARAM_MIPS_TOP_K)
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


    for (int n = PARAM_MIPS_TOP_K - 1; n >= 0; --n)
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
void extract_TopK_MIPS(const Ref<VectorXf> &p_vecQuery, const unordered_set<int>& p_setTopB, Ref<VectorXi> p_vecTopK)
{
    // incase we do not have enough candidates : LSH-Table
//    assert((int)p_setTopB.size() >= PARAM_MIPS_TOP_K);

    if ((int)p_setTopB.size() <= PARAM_MIPS_TOP_K)
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

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    for (const auto& iPointIdx: p_setTopB)
    {
        float fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < PARAM_MIPS_TOP_K)
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

    for (int n = PARAM_MIPS_TOP_K - 1; n >= 0; --n)
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
void extract_max_TopB(const Ref<VectorXf> &p_vecEst, Ref<VectorXi> p_vecTopB)
{
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopB;

    for (int n = 0; n < p_vecEst.size(); ++n)
    {
        float fValue = p_vecEst(n);

        // Insert into minQueue
        if ((int)minQueTopB.size() < PARAM_MIPS_TOP_B)
            minQueTopB.push(IFPair(n, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopB.top().m_fValue)
            {
                minQueTopB.pop();
                minQueTopB.push(IFPair(n, fValue));
            }
        }
    }

    for (int n = PARAM_MIPS_TOP_B - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopB(n) = minQueTopB.top().m_iIndex;
        minQueTopB.pop();
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

void outputFile(const Ref<const VectorXf> & p_vecEst, string p_sOutputFile)
{
	cout << "Outputing File..." << endl;
	ofstream myfile(p_sOutputFile);

	//cout << p_matKNN << endl;

	for (int j = 0; j < p_vecEst.rows(); ++j)
	{
        //cout << "Print col: " << i << endl;
		myfile << p_vecEst(j) << '\n';
	}

	myfile.close();
	cout << "Done" << endl;
}

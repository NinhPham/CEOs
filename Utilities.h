#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "Header.h"

#include "sys/types.h" // for getting RAM
#include "sys/sysinfo.h" // for getting RAM

// Inline functions
/**
Convert an integer to string
**/
inline string int2str(int x)
{
    stringstream ss;
    ss << x;
    return ss.str();
}

/**
Fast cosine approximation with Taylor series:
https://gist.github.com/geraldyeo/988116
**/
inline float hackCos(float x)
{
    //0.405284735 =-4/(pi^2)
    //1.27323954  = 4/pi
    //6.28318531 = 2pi

    float cos;

    // make sure [-pi, pi]
    if (x < -3.14159265)
        x += 6.28318531;
    else if (x > 3.14159265)
        x -= 6.28318531;

    x += 1.57079632; // cos(x) = sin(x + pi/2)

    if (x > 3.14159265)
        x -= 6.28318531;

    // Uper bound it by 4x/pi +- 4x^2/pi^2 (depend on the sign of x)
    if (x < 0)
    {
        cos = 1.27323954 * x + 0.405284735 * x * x;

        if (cos < 0)
            cos = .225 * (cos * (-cos) - cos) + cos;
        else
            cos = .225 * (cos * cos - cos) + cos;
    }
    else
    {
        cos = 1.27323954 * x - 0.405284735 * x * x;

        if (cos < 0)
            cos = .225 * (cos *-cos - cos) + cos;
        else
            cos = .225 * (cos * cos - cos) + cos;
    }

    return cos;
}

/**
Get sign
**/
inline int sgn(float x)
{
    if (x >= 0)
        return 1;
    else
        return -1;

}

inline void getRAM()
{
    // https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
    struct sysinfo memInfo;
    sysinfo (&memInfo);
    long long totalPhysMem = memInfo.totalram - memInfo.freeram;
    //Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;
    cout << "Amount of RAM: " << (float)totalPhysMem / (1L << 30) << " GB." << endl;
}

inline void bitHD3Generator(int p_iNumBit)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    bitHD3 = boost::dynamic_bitset<> (p_iNumBit);

    // Loop col first since we use col-wise
    for (int d = 0; d < p_iNumBit; ++d)
    {
        bitHD3[d] = unifDist(generator) & 1;
    }
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
inline void extract_S0_MinMax_Idx(const Ref<VectorXf> &vecProject, IVector &vecMinIdx, IVector &vecMaxIdx)
{
//    vector<int> idx(PARAM_DATA_N);
//    iota(idx.begin(), idx.end(), 0); //Initializing
//    sort(idx.begin(), idx.end(), [&](int i,int j){return vecEstimate(i)< vecEstimate(j);} );
//
//    vecMinIdx.assign(idx.begin(), idx.begin() + s0);
//    vecMaxIdx.assign(idx.end() - s0, idx.end());

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK; // take the min out, keep the max in queue
    priority_queue< IFPair, vector<IFPair> > maxQueTopK;

    for (int n = 0; n < PARAM_CEOs_D_UP; ++n)
    {
        float fTemp = vecProject(n);

        // Get min out
        if ((int)minQueTopK.size() < PARAM_CEOs_S0)
            minQueTopK.push(IFPair(n, fTemp));
        else if (fTemp > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(n, fTemp));
        }

        // Get max out
        if ((int)maxQueTopK.size() < PARAM_CEOs_S0)
            maxQueTopK.push(IFPair(n, fTemp));
        else if (fTemp < maxQueTopK.top().m_fValue)
        {
            maxQueTopK.pop();
            maxQueTopK.push(IFPair(n, fTemp));
        }
    }

    // The smallest/largerst value should come first.
    for (int n = PARAM_CEOs_S0 - 1; n >= 0; --n)
    {
        // Get max idx
        vecMaxIdx[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();

        vecMinIdx[n] = maxQueTopK.top().m_iIndex;
        maxQueTopK.pop();
    }
 }

 /** \brief Apply random rotating via FWHT and return top-S0 min & max index in vector<int>
 *
 * \param
 *
 - VectorXf::vecCounter: of D-UP x 1
 *
 *
 * \return
 *
 - vector<int>:: min & max contains top S0 point indexes
 *
 */
inline void rotateAndGet_S0_MinMax(const Ref<const VectorXf> & vecQuery, IVector &vecMinIdx, IVector &vecMaxIdx)
{
    VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
    vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;

    for (int r = 0; r < PARAM_CEOs_NUM_ROTATIONS; ++r)
    {
        // Component-wise multiplication with a random sign
        for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
        {
            vecProjectedQuery(d) *= (2 * (int)bitHD3[r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
        }

        // Multiple with Hadamard matrix by calling FWHT transform
        fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
    }

    // Resize vector, only get D-UP positions
    vecProjectedQuery.resize(PARAM_CEOs_D_UP);

    // Extract min and max - might be slower with priority queue if D-UP is large (256 or 512)
    // You can call: extract_S0_MinMax_Idx(vecProjectedQuery, vecMinIdx, vecMaxIdx);
    // or
    // using the following simple code

//    vector<int> idx(PARAM_CEOs_D_UP);
//    iota(idx.begin(), idx.end(), 0); //Initializing
//    sort(idx.begin(), idx.end(), [&](int i,int j){return vecProjectedQuery(i)< vecProjectedQuery(j);} );
//
//    vecMinIdx.assign(idx.begin(), idx.begin() + PARAM_CEOs_S0);
//    vecMaxIdx.assign(idx.end() - PARAM_CEOs_S0, idx.end());

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK; // take the min out, keep the max in queue
    priority_queue< IFPair, vector<IFPair> > maxQueTopK;

    for (int n = 0; n < PARAM_CEOs_D_UP; ++n)
    {
        float fTemp = vecProjectedQuery(n);

        // Get min out
        if ((int)minQueTopK.size() < PARAM_CEOs_S0)
            minQueTopK.push(IFPair(n, fTemp));
        else if (fTemp > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(n, fTemp));
        }

        // Get max out
        if ((int)maxQueTopK.size() < PARAM_CEOs_S0)
            maxQueTopK.push(IFPair(n, fTemp));
        else if (fTemp < maxQueTopK.top().m_fValue)
        {
            maxQueTopK.pop();
            maxQueTopK.push(IFPair(n, fTemp));
        }
    }

    // The smallest/largerst value should come first.
    for (int n = PARAM_CEOs_S0 - 1; n >= 0; --n)
    {
        // Get max idx
        vecMaxIdx[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();

        vecMinIdx[n] = maxQueTopK.top().m_iIndex;
        maxQueTopK.pop();
    }
}

inline void rotateAndGetMax(const Ref<const VectorXf> & vecQuery, int &maxIndex)
{
    VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
    vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;

    for (int r = 0; r < PARAM_CEOs_NUM_ROTATIONS; ++r)
    {
        // Component-wise multiplication with a random sign
        for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
        {
            vecProjectedQuery(d) *= (2 * (int)bitHD3[r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
        }

        // Multiple with Hadamard matrix by calling FWHT transform
        fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
    }

    // Resize vector, only get D-UP positions
    vecProjectedQuery.resize(PARAM_CEOs_D_UP);

    // Compute max and min
    vecProjectedQuery.maxCoeff(&maxIndex);
}

inline void rotateAndGetMinMax(const Ref<const VectorXf> & vecQuery, int &minIndex, int &maxIndex)
{
    VectorXf vecProjectedQuery = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
    vecProjectedQuery.segment(0, PARAM_DATA_D) = vecQuery;

    for (int r = 0; r < PARAM_CEOs_NUM_ROTATIONS; ++r)
    {
        // Component-wise multiplication with a random sign
        for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
        {
            vecProjectedQuery(d) *= (2 * (int)bitHD3[r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
        }

        // Multiple with Hadamard matrix by calling FWHT transform
        fht_float(vecProjectedQuery.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
    }

    // Resize vector, only get D-UP positions
    vecProjectedQuery.resize(PARAM_CEOs_D_UP);

    // Compute max and min
    vecProjectedQuery.minCoeff(&minIndex);
    vecProjectedQuery.maxCoeff(&maxIndex);
}

// Printing
void printVector(const vector<int> & );
void printVector(const vector<IFPair> &);

// Output
void outputFile(const Ref<const MatrixXi> & , string );
void outputFile(const Ref<const VectorXf> & , string );

// extract topB & topK using distance
void extract_TopB_TopK_Histogram(const Ref<VectorXf>&, const Ref<VectorXf>&, Ref<VectorXi>); // CEOs-Est and coCEOs vector
void extract_TopB_TopK_Histogram(const unordered_map<int, float> &, const Ref<VectorXf>&, Ref<VectorXi>); // coCEOs map

// compute distance and extract top-K
void extract_TopK_MIPS(const Ref<VectorXf>&, const Ref<VectorXi>&, Ref<VectorXi>); // Query of 1CEOs and 2CEOs
void extract_TopK_MIPS(const Ref<VectorXf> &, const unordered_set<int>& , Ref<VectorXi> ); // for LSH
void extract_max_TopB(const Ref<VectorXf>&, Ref<VectorXi>); // for building index of 1CEOs, 2CEOs

// Fast Hadamard transform - But not use anymore
void inline wht_bfly (float& , float& );
void FWHT (Ref<VectorXf>);

#endif // UTILITIES_H_INCLUDED

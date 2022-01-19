<<<<<<< HEAD
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

// Printing
void printVector(const vector<int> & );
void printVector(const vector<IFPair> &);

// Output
void outputFile(const Ref<const MatrixXi> & , string );

// Random source
void gaussGenerator(int, int);
void HD3Generator(int);

// extract topK min&max index of query
void extract_TopK_MinMax_Idx(const Ref<VectorXf>&, int, Ref<VectorXi>, Ref<VectorXi>);

// extract topB&topK
void extract_TopB_TopK_Histogram(const Ref<VectorXf>&, const Ref<VectorXf>&, int, int, Ref<VectorXi>);
void extract_TopB_TopK_Histogram(const unordered_map<int, float> &, const Ref<VectorXf>&, int, int, Ref<VectorXi>);

// compute distance and extract top-K
void extract_TopK_MIPS(const Ref<VectorXf>&, const Ref<VectorXi>&, int, Ref<VectorXi>);
void extract_TopK_MIPS(const Ref<VectorXf> &, const unordered_set<int>& , int ,  Ref<VectorXi> ); // for LSH
void extract_max_TopK(const Ref<VectorXf>&, int, Ref<VectorXi>);

// Fast Hadamard transform
void inline wht_bfly (float& , float& );
void FWHT (Ref<VectorXf>);

#endif // UTILITIES_H_INCLUDED
=======
#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "Header.h"

#include <iostream> // cin, cout
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <time.h> // for time(0) to generate different random number
#include <random>

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
Convert CPU Time from clock() to ms
**/
inline float getCPUTime(double dTime)
{
    return (float)(dTime / CLOCKS_PER_SEC);
}

/**
Convert Wall Time from clock() to ms
**/
inline double getWallTime(timespec tStart, timespec tStop)
{
    return (double)((tStop.tv_sec + tStop.tv_nsec * 1e-6) - (double)(tStart.tv_sec + tStart.tv_nsec * 1e-6));
}

/**
Fast cosine approximation with Taylor series:
https://gist.github.com/geraldyeo/988116
**/
inline double hackCos(double x)
{
    //0.405284735 =-4/(pi^2)
    //1.27323954  = 4/pi
    //6.28318531 = 2pi

    double dRes;

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
        dRes = 1.27323954 * x + 0.405284735 * x * x;
    else
        dRes = 1.27323954 * x - 0.405284735 * x * x;

    if (dRes < 0)
        dRes = .225 * (dRes * (-dRes) - dRes) + dRes;
    else
        dRes = .225 * (dRes * dRes - dRes) + dRes;

    return dRes;
}

/**
Get sign
**/
inline int sgn(double x)
{
    if (x >= 0) return 1;
    else return -1;
    // return 0;
}

// Print
void printVector(const vector<double> &);
void printVector(const vector<int> &);

void printVector(const vector<IDPair> &);
void printVector(const VectorXd &);

void printMap(const unordered_map<int, IVector> &);

// print priority queue
void printQueue(priority_queue<IDPair, vector<IDPair>, greater<IDPair>>, string );

// save
void saveQueue(priority_queue<IDPair, vector<IDPair>, greater<IDPair>>, string);

void saveVector(const IVector &, string);
void saveVector(const DVector &, string);
void saveVector(const VectorXd &, string);
void saveVector(const VectorXi &, string);

void saveSet(unordered_set<int> , string);
void saveMap(unordered_map<int, double> , string);
void saveMap(unordered_map<int, int> , string);

void simHashGenerator();
void gaussGenerator(int, int);
void HD3Generator(int);

// extract topK min&max index of query
VectorXi extract_TopK_MinIdx(const VectorXd &, int);
VectorXi extract_TopK_MaxIdx(const VectorXd &, int);

// extract topB from a vector<IDPair>
IVector extract_SortedTopK_Histogram(const IVector &, int);
IVector extract_SortedTopK_Histogram(const DVector &, int);
IVector extract_SortedTopK_Histogram(const VectorXd &, int);

IVector extract_SortedTopK_Histogram(const unordered_map<int, int> &, int);
IVector extract_SortedTopK_Histogram(const unordered_map<int, double> &, int);

void extract_TopK_MIPS(const VectorXd &, const IVector &, int ,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &);
void extract_TopK_MIPS(const VectorXd &, const Ref<VectorXi>, int ,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &);

void extract_TopK_MIPS(const VectorXd &, const unordered_set<int> &, int ,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &);
void extract_TopK_MIPS_Projected_X(const VectorXd &, const IVector &, int ,
                 priority_queue< IDPair, vector<IDPair>, greater<IDPair> > &);

// Fast Hadamard transform
void inline wht_bfly (double& , double& );
void FWHT (Ref<VectorXd>, const IVector &);

#endif // UTILITIES_H_INCLUDED
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

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

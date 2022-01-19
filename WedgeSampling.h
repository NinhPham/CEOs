#ifndef WEDGESAMPLING_H_INCLUDED
#define WEDGESAMPLING_H_INCLUDED

#include "Header.h"

 // pre-processing functions
void dimensionSort(bool = true);

void wedge_ColWeight(const Ref<VectorXf> &, Ref<VectorXf>);
void dWedge_Vector_TopK();
void dWedge_Map_TopK();

void greedy_TopK();

#endif // WEDGESAMPLING_H_INCLUDED

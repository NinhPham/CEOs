<<<<<<< HEAD

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
=======

#ifndef WEDGESAMPLING_H_INCLUDED
#define WEDGESAMPLING_H_INCLUDED

#include "Header.h"

// utilities functions

void wedge_ColWeight(const VectorXd &, DVector &);

 // pre-processing functions
void dimensionSort(bool = true);

void dWedge_Vector_TopK();
void dWedge_Map_TopK();








#endif // WEDGESAMPLING_H_INCLUDED
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

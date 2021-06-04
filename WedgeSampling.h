
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

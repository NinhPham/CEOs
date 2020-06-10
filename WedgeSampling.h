
#ifndef WEDGESAMPLING_H_INCLUDED
#define WEDGESAMPLING_H_INCLUDED

#include "Header.h"

// utilities functions


 // pre-processing functions
void dimensionSort(bool = true);
void dimensionShiftSort();
void dimensionPosSort();

void wedge_PreSampling();
void shift_Wedge_PreSampling();

// query functions
void wedge_Vector_TopK();
void wedge_Map_TopK();

void dWedge_Vector_TopK();
void dWedge_Map_TopK();

void shift_Wedge_Vector_TopK();
void shift_Wedge_Map_TopK();
// void shiftWedgeFastTopK_2Hist(); // not use

void shift_dWedge_Vector_TopK();
void shift_dWedge_Map_TopK();

void pos_dWedge_Vector_TopK();
void pos_dWedge_Map_TopK();







#endif // WEDGESAMPLING_H_INCLUDED

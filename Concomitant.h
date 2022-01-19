
#ifndef CONCOMITANT_H_INCLUDED
#define CONCOMITANT_H_INCLUDED

#include "Header.h"

// sCEOs
void rotateData();
void sCEOs_Est_TopK();

// sCEOs with TA
void build_sCEOs_TA_Index();
void sCEOs_TA_TopK();

// co-CEOs
void build_coCEOs_Index();
void coCEOs_Map_TopK();
void coCEOs_Vector_TopK();

// 1 CEOs
void build_1CEOs_Index();
void maxCEOs_TopK();

// 2 CEOs
void build_2CEOs_Index();
void minmaxCEOs_TopK();

// Conventional RP
void GaussianRP();
void RP_Est_TopK();


#endif

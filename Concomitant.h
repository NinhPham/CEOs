
#ifndef CONCOMITANT_H_INCLUDED
#define CONCOMITANT_H_INCLUDED

#include "Header.h"

// s CEOs
void GaussianRP_Data();
void GaussianRP_Query(Ref<VectorXd>, const VectorXd &);
void sCEOs_Est_TopK();

// s CEOs with TA
void build_sCEOs_TA();
void sCEOs_TA_TopK();

// 1 CEOs
void build_1CEOs_Search();
void build_1CEOs_Hash();
void maxCEOs_Hash();
void maxCEOs_Search();

// 2 CEOs
void build_2CEOs_Search();
void minmaxCEOs_Search();

// co-CEOs
void build_coCEOs_Search();
void coCEOs_Map_Search();
void coCEOs_Vector_Search();

// RP
void GaussianRP();
void RP_Est_TopK();


#endif

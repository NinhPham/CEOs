<<<<<<< HEAD

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
=======

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
>>>>>>> 5f4f1dd09f15a6d3fb2323762601e5ea7f359134

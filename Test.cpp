#include "Utilities.h"
#include "Header.h"
#include "Test.h"

/**
Check a positive matrix
**/
bool isPosMatrix(const MatrixXd &matrixT)
{
    int d, n;
    for (d = 0; d < matrixT.cols(); ++d)
    {
        for (n = 0; n < matrixT.rows(); ++n)
        {
            if (matrixT(n, d) < 0.0)
                return false;
        }
    }

    return true;
}





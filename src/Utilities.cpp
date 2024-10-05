#include "Utilities.h"
#include "Header.h"

#include <fstream> // fscanf, fopen, ofstream

/**
Input:
(col-wise) matrix p_matKNN of size K x Q

Output: Q x K
- Each row is for each query
**/
void outputFile(const Ref<const MatrixXi> & p_matKNN, const string& p_sOutputFile)
{
//	cout << "Outputing File..." << endl;
	ofstream myfile(p_sOutputFile);

	//cout << p_matKNN << endl;

	for (int j = 0; j < p_matKNN.cols(); ++j)
	{
        //cout << "Print col: " << i << endl;
		for (int i = 0; i < p_matKNN.rows(); ++i)
		{
            myfile << p_matKNN(i, j) << ' ';

		}
		myfile << '\n';
	}

	myfile.close();
//	cout << "Done" << endl;
}

/**
 *
 * @param dataset
 * @param numPoints
 * @param numDim
 * @param MATRIX_X
 */
void loadtxtData(const string & dataset, int numPoints, int numDim, MatrixXf & MATRIX_X)
{
    FILE *f = fopen(dataset.c_str(), "r");
    if (!f) {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }

    // Important: If use a temporary vector to store data, then it doubles the memory
    MATRIX_X = MatrixXf::Zero(numDim, numPoints); // col-wise

    // Each line is a vector of d dimensions
    for (int n = 0; n < numPoints; ++n) {
        for (int d = 0; d < numDim; ++d) {
            fscanf(f, "%f", &MATRIX_X(d, n));
        }
    }

    cout << "Finish reading data" << endl;
}

/*
 * @param nargs:
 * @param args:
 * @return: Parsing parameter for FalconnPP++
 */
void readIndexParam(int nargs, char** args, IndexParam& iParam)
{
    if (nargs < 6)
        exit(1);

    // NumPoints n
    bool bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_points") == 0)
        {
            iParam.n_points = atoi(args[i + 1]);
            cout << "Number of rows/points of X: " << iParam.n_points << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Number of rows/points is missing !" << endl;
        exit(1);
    }

    // Dimension
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_features") == 0)
        {
            iParam.n_features = atoi(args[i + 1]);
            cout << "Number of columns/dimensions: " << iParam.n_features << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        cerr << "Error: Number of columns/dimensions is missing !" << endl;
        exit(1);
    }


    // numProjections
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_proj") == 0)
        {
            iParam.n_proj = atoi(args[i + 1]);
            cout << "Number of projections: " << iParam.n_proj << endl;
            bSuccess = true;
            break;

        }
    }
    if (!bSuccess)
    {
        int iTemp = ceil(log2(1.0 * iParam.n_features));
        iParam.n_proj = max(256, 1 << iTemp); // default is 256 since default of exponent is 1
        cout << "Number of projections: " << iParam.n_proj << endl;
    }

    // numRepeats
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_repeats") == 0)
        {
            iParam.n_repeats = atoi(args[i + 1]);
            cout << "Exponent (power): " << iParam.n_repeats << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.n_repeats = 1;
        cout << "Default exponent: " << iParam.n_repeats << endl;
    }

    // indexBucketSize
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--iTopPoints") == 0)
        {
            iParam.indexBucketSize = atoi(args[i + 1]);
            cout << "Top-points closest/furthest to the random vector: " << iParam.indexBucketSize << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.indexBucketSize = 1000;
        cout << "Default top-points: " << iParam.indexBucketSize << endl;
    }

    // n_threads
    iParam.n_threads = -1;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_threads") == 0)
        {
            iParam.n_threads = atoi(args[i + 1]);
            cout << "Number of threads: " << iParam.n_threads << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.n_threads = -1;
        cout << "Use all threads: " << iParam.n_threads << endl;
    }

    // centering
    iParam.centering = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--centering") == 0)
        {
            iParam.centering = true;
            cout << "We center the data. " << endl;
            break;
        }
    }

    // n_threads
    iParam.seed = -1;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--random_seed") == 0)
        {
            iParam.seed = atoi(args[i + 1]);
            cout << "Random seed: " << iParam.seed << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        cout << "Use a random seed !" << endl;
    }
}

/*
 * @param nargs:
 * @param args:
 * @return: Parsing parameter for FalconnPP++
 */
void readQueryParam(int nargs, char** args, QueryParam & qParam)
{
    if (nargs < 4)
        exit(1);

    // NumPoints n
    bool bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_queries") == 0)
        {
            qParam.n_queries = atoi(args[i + 1]);
            cout << "Number of rows/points of Q: " << qParam.n_queries << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Number of queries is missing !" << endl;
        exit(1);
    }

    // Qery dimension = Point dimensions

    // Top-K: We need this param for indexing as we might NOT filter out points on the sparse buckets.
    // This param should be equal to top-K.
    // Otherwise, PARAM_BUCKET_TOP_K = 50 might suffice for top-K = {1, ..., 100})
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_neighbors") == 0)
        {
            qParam.n_neighbors = atoi(args[i + 1]);
            cout << "Top-K query: " << qParam.n_neighbors << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_neighbors = 1;
        cout << "Default top-K: " << qParam.n_neighbors << endl;
    }

    // Top closest random vector
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--probedVectors") == 0)
        {
            qParam.n_probedVectors = atoi(args[i + 1]);
            cout << "Number of closest/furthest random vectors: " << qParam.n_probedVectors << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_probedVectors = 10;
        cout << "Default number of closest/furthest random vectors: " << qParam.n_probedVectors << endl;
    }

    // Top closest random vector
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--probedPoints") == 0)
        {
            qParam.n_probedPoints = atoi(args[i + 1]);
            cout << "Number of probed points for each vectors: " << qParam.n_probedPoints << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_probedPoints = 10;
        cout << "Default number of probed points for each vectors: " << qParam.n_probedPoints << endl;
    }

    // Candidate size
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_cand") == 0)
        {
            qParam.n_cand = atoi(args[i + 1]);
            cout << "Number of candidates: " << qParam.n_cand << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_cand = qParam.n_neighbors;
        cout << "Default number of candidates: " << qParam.n_cand << endl;
    }

    // verbose
    qParam.verbose = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--verbose") == 0)
        {
            qParam.verbose = true;
            cout << "Verbose mode: " << qParam.verbose << endl;
            break;
        }
    }

    // Use CEOs to estimate inner product
//    PARAM_QUERY_DOT_ESTIMATE = false;
//    for (int i = 1; i < nargs; i++)
//    {
//        if (strcmp(args[i], "--useEst") == 0)
//        {
//            PARAM_QUERY_DOT_ESTIMATE = true;
//            cout << "Use CEOs to estimate inner product." << endl;
//            break;
//        }
//    }

}
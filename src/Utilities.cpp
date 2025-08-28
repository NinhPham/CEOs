#include "Utilities.h"
#include "Header.h"

#include <fstream> // fscanf, fopen, ofstream
#include <fstream> // fscanf, fopen, ofstream
#include <sstream>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * Generate random bit for FHT
 *
 * @param p_iNumBit
 * @param bitHD
 * @param random_seed
 * return bitHD that contains fhtDim * n_rotate (default of n_rotate = 3)
 */
void bitHD3Generator(int p_iNumBit, int random_seed, boost::dynamic_bitset<> & bitHD)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    if (random_seed >= 0)
        seed = random_seed;

    // std::random_device rd;  // Seed source
    // std::mt19937 gen(seed); // Mersenne Twister engine seeded with rd()
    default_random_engine generator(seed);

    uniform_int_distribution<uint32_t> unifDist(0, 1);

    bitHD = boost::dynamic_bitset<> (p_iNumBit);

    // Loop col first since we use col-wise
    for (int d = 0; d < p_iNumBit; ++d)
    {
        bitHD[d] = unifDist(generator) & 1;
    }

}

/**
 * Generate 2 vectors of random sign, each for one layer.
 * We use boost::bitset for saving space
 * @param p_iNumBit = L * 3 * Length (3 rotation, 2 layers, each with L tables)
 */
void bitHD3Generator2(int p_iNumBit, int random_seed, boost::dynamic_bitset<> & bitHD1, boost::dynamic_bitset<> & bitHD2)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    if (random_seed > -1) // then use the assigned seed
        seed = random_seed;


    // std::random_device rd;  // Seed source
    // std::mt19937 generator(seed); // Mersenne Twister engine seeded with rd()
    default_random_engine generator(seed);

    uniform_int_distribution<uint32_t> unifDist(0, 1);

    bitHD1 = boost::dynamic_bitset<> (p_iNumBit);
    bitHD2 = boost::dynamic_bitset<> (p_iNumBit);

    for (int d = 0; d < p_iNumBit; ++d)
    {
        bitHD1[d] = unifDist(generator) & 1;
        bitHD2[d] = unifDist(generator) & 1;

    }

    // for (int i = 0; i < 20; i++)
    // {
    //     cout << bitHD1[i] << endl;
    //     cout << bitHD2[i] << endl;
    // }
}


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
 * @param MATRIX_X: col-wise matrix of size (numDim x numPoints)
 */
void loadtxtData(const string & dataset, int numPoints, int numDim, RowMajorMatrixXf & MATRIX_X)
{
    FILE *f = fopen(dataset.c_str(), "r");
    if (!f) {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }

    // Important: If use a temporary vector to store data, then it doubles the memory
    MATRIX_X = RowMajorMatrixXf::Zero(numDim, numPoints); // col-wise

    // Each line is a vector of d dimensions
    for (int n = 0; n < numPoints; ++n) {
        for (int d = 0; d < numDim; ++d) {
            // fscanf(f, "%f", &MATRIX_X(d, n)); // col-major
            fscanf(f, "%f", &MATRIX_X(n, d)); // row-major
        }
    }

    cout << "Finish reading data" << endl;
}
/**
 * @param dataset
 * @param numPoints
 * @param numDim
 * @param MATRIX_X: col-wise matrix of size (numDim x numPoints)
 */
void loadbinData(const string& dataset, int numPoints, int numDim, RowMajorMatrixXf & MATRIX_X) {

    // Open file
    int fd = open(dataset.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        exit(1);
    }

    size_t filesize = sb.st_size;
    size_t total_rows = filesize / (numDim * sizeof(float));

    std::cout << "Total rows = " << total_rows << std::endl;

    // Map the file into memory
    void* mapped = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    close(fd); // fd no longer needed

    if ((size_t)numPoints > total_rows) {
        std::cerr << "Error: numPoints exceeds the number of rows in the file." << std::endl;
        munmap(mapped, filesize);
        exit(1);
    }

    // Important: If use a temporary vector to store data, it doubles the memory
    //MATRIX_X = MatrixXf::Zero(numDim, numPoints); // default is col-major
    MATRIX_X = RowMajorMatrixXf::Zero(numDim, numPoints);

    // Interpret data as float array
    float* data = reinterpret_cast<float*>(mapped);

    // Each line is a vector of d dimensions
    // This is for col-major
    // for (int n = 0; n < numPoints; ++n) {
    //     for (int d = 0; d < numDim; ++d) {
    //         MATRIX_X(d, n) = data[n * numDim + d];
    //     }
    // }
    // This is for row-major
    for (int n = 0; n < numPoints; ++n) {
        for (int d = 0; d < numDim; ++d) {
            MATRIX_X(n, d) = data[n * numDim + d];
        }
    }

    // Unmap when done
    munmap(mapped, filesize);

    cout << "Finish reading data" << endl;
    cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

    /**
    Print the first col (1 x N)
    Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
    **/
    //        cout << MATRIX_X.col(0) << endl << endl;
    //        cout << "In memory (col-major):" << endl;
    //        for (n = 0; n < 10; n++)
    //            cout << *(MATRIX_X.data() + n) << "  ";
    //        cout << endl << endl;
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
        iParam.n_proj = max(256, 1 << iTemp); // default is 256 since default of repeat is 1
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

    // Top-m points (e.g. bucket size for each random vector)
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--top_m") == 0)
        {
            iParam.top_m = atoi(args[i + 1]);
            cout << "Top_m points closest/furthest to the random vector: " << iParam.top_m << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.top_m = 1000;
        cout << "Default Top_m points: " << iParam.top_m << endl;
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
        if (strcmp(args[i], "--n_probed_vectors") == 0)
        {
            qParam.n_probed_vectors = atoi(args[i + 1]);
            cout << "Number of closest/furthest random vectors: " << qParam.n_probed_vectors << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_probed_vectors = 10;
        cout << "Default number of closest/furthest random vectors: " << qParam.n_probed_vectors << endl;
    }

    // Top-points closest random vector. This value should be smaller than top-m from indexing
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_probed_points") == 0)
        {
            qParam.n_probed_points = atoi(args[i + 1]);
            cout << "Number of probed points for each vectors: " << qParam.n_probed_points << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_probed_points = 10;
        cout << "Default number of probed points for each vectors: " << qParam.n_probed_points << endl;
    }

    // Candidate size
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_cand") == 0)
        {
            qParam.n_cand = atoi(args[i + 1]);
            cout << "Number of re-ranking candidates: " << qParam.n_cand << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_cand = qParam.n_neighbors;
        cout << "Default number of reranking candidates to compute exact distance for re-ranking: " << qParam.n_cand << endl;
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
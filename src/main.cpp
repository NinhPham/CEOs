#include <iostream>
#include <ctime> // for time(0) to generate different random number

#include "Header.h"
#include "Utilities.h"
#include "CEOs.h"
#include "streamCEOs.h"

// --numData 1183514 --n_features 200 --n_tables 10 --n_proj 256 --bucket_minSize 20, --bucket_scale 0.01
// --X "/home/npha145/Dataset/kNN/CosineKNN/Glove_X_1183514_200.txt" --n_threads 4
// --Q "/home/npha145/Dataset/kNN/CosineKNN/Glove_Q_1000_200.txt" --n_queries 1000  --qProbes 10 --n_neighbors 20 --n_threads 4

int main(int nargs, char** args) {

    IndexParam iParam;
    QueryParam qParam;

    readIndexParam(nargs, args, iParam);
    readQueryParam(nargs, args, qParam);

    // Read data
    RowMajorMatrixXf MATRIX_X, MATRIX_Q;

    // Read dataset
    string dataset = "";
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--X") == 0) {
            dataset = args[i + 1]; // convert char* to string
            break;
        }
    }
    if (dataset == "") {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }
    else
        // loadtxtData(dataset, iParam.n_points, iParam.n_features, MATRIX_X);
        loadbinData(dataset, iParam.n_points, iParam.n_features, MATRIX_X);

    // Read query set
    dataset = "";
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--Q") == 0) {
            dataset = args[i + 1]; // convert char* to string
            break;
        }
    }
    if (dataset == "") {
        cerr << "Error: Query file does not exist !" << endl;
        exit(1);
    }
    else
        // loadtxtData(dataset, qParam.n_queries, iParam.n_features, MATRIX_Q);
        loadbinData(dataset, qParam.n_queries, iParam.n_features, MATRIX_Q);

    // CEOs-Est
    CEOs ceos(iParam.n_points, iParam.n_features);
    ceos.set_CEOsParam(iParam.n_proj, iParam.n_repeats, iParam.top_m, iParam.n_threads, iParam.seed);

    ceos.build_CEOs(MATRIX_X);
    ceos.n_cand = qParam.n_cand;
    ceos.n_probed_vectors = qParam.n_probed_vectors;
    ceos.search_CEOs(MATRIX_Q, qParam.n_neighbors, qParam.verbose);


//    ceos.clear();
//    ceos.set_coCEOsParam(iParam.n_proj, iParam.n_repeats, iParam.indexBucketSize, iParam.n_threads, iParam.seed);
//    ceos.build_coCEOs(MATRIX_X);
//    ceos.n_cand = qParam.n_cand;
//    ceos.n_probedBuckets = qParam.n_probedBuckets;
//    ceos.search_coCEOs(MATRIX_Q, qParam.n_neighbors, qParam.verbose);
//
//    ceos.add_coCEOs(MATRIX_Q);
//    ceos.search_coCEOs(MATRIX_Q, qParam.n_neighbors, qParam.verbose);

    // coCEOs
    streamCEOs coceos(iParam.n_features);
    coceos.set_streamCEOsParam(iParam.n_proj, iParam.n_repeats, iParam.top_m, iParam.n_threads, iParam.seed);
    coceos.build(MATRIX_X);
    coceos.n_cand = qParam.n_cand;
    coceos.n_probed_vectors = qParam.n_probed_vectors;
    coceos.estimate_search(MATRIX_Q, qParam.n_neighbors, qParam.verbose);

    coceos.update(MATRIX_Q, 1000);
    coceos.hash_search(MATRIX_Q, qParam.n_neighbors, qParam.verbose);

    return 0;

    // MATRIX_Q.resize(0, 0); // Note: we should not resize if testing multiple qProbes
}

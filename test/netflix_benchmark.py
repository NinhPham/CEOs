import utils
import numpy as np

if __name__ == '__main__':

    path = "/shared/Dataset/ANNS/CEOs/"

    n = 17770
    d = 300
    bin_file = path + 'Netflix_X_17770_300.bin'
    X = utils.mmap_bin(bin_file, n, d)


    q = 999
    bin_file = path + "Netflix_Q_999_300.bin"
    Q = utils.mmap_bin(bin_file, q, d)

    ## Cosine
    # norms = np.linalg.norm(X, axis=1, keepdims=True)
    # norms[norms == 0] = 1
    # X /= norms # X = np.array(X, copy=True)  # makes it writable
    # norms = np.linalg.norm(Q, axis=1, keepdims=True)
    # norms[norms == 0] = 1
    # Q /= norms # Q = np.array(Q, copy=True)  # makes it writable

    #-------------------------------------------------------------------
    n_threads = 32
    # k = 100
    # exact_kNN = utils.faissBF(X, Q, k, n_threads)
    # exact_kNN = exact_kNN.astype(np.int32)
    # np.save(path + "Netflix_Cosine_k_100_indices.npy", exact_kNN)    # shape: (n, k), dtype: int32
    # exact_kNN = np.load(path + "Netflix_Cosine_k_100_indices.npy")  # shape: (n, k), dtype: int32
    exact_kNN = np.load(path + "Netflix_Dot_k_100_indices.npy")  # shape: (n, k), dtype: int32

    k = 10
    exact_kNN = exact_kNN[: , :k]

    #-------------------------------------------------------------------
    probed_vectors = 20
    n_cand = 100
    n_repeats = 2**1
    D = 2**9

    print("\nCEOs")

    utils.ceos_est(exact_kNN, X, Q, k, D, probed_vectors, n_cand, n_repeats, n_threads)

    #-------------------------------------------------------------------
    top_m = 500
    probed_vectors = 40
    n_cand = 100
    n_repeats = 2**1
    D = 2**9

    print("\ncoCEOs-est")
    utils.coceos_est(exact_kNN, X, Q, k, D, top_m, probed_vectors, n_cand, n_repeats, n_threads)

    #-------------------------------------------------------------------
    # print("\ncoCEOs-stream-est")
    # coceos_est(exact_kNN, X, Q, k, numThreads, top_point, top_r, n_cand)
    #
    # n_cand = 1000
    top_m = 20
    probed_vectors = 100
    probed_points = top_m
    n_repeats = 2**3
    D = 2**9
    print("\nCEOs-hash")
    utils.ceos_hash(exact_kNN, X, Q, k, D, top_m, probed_vectors, probed_points, n_repeats, n_threads)
    #-------------------------------------------------------------------
    # utils.streamCEOs_test(exact_kNN, X, Q, k, top_m, probed_vectors, n_cand, n_repeats, n_threads)
    print("\nStreamCEOs-est")

    top_m = 500
    probed_vectors = 40
    n_cand = 100
    n_repeats = 2**1
    D = 2**9
    utils.streamCEOs_est(X, Q, k, top_m, probed_vectors, n_cand, n_repeats, n_threads)

    print("\nStreamCEOs-hash")
    top_m = 50
    probed_vectors = 40
    n_repeats = 2**1
    D = 2**9
    utils.streamCEOs_hash(X, Q, k, top_m, probed_vectors, n_repeats, n_threads)

    #-------------------------------------------------------------------
    # print("\nFaiss-IVF")
    # utils.faissIVF(exact_kNN, X, Q, k, n_list=1000, n_probe=20, n_threads=n_threads)
    #
    # #-------------------------------------------------------------------
    # print("\nHnswlib")
    # utils.hnswMIPS(exact_kNN, X, Q, k, efSearch=100, n_threads=n_threads)







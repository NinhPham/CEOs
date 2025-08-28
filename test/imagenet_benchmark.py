import utils
import numpy as np


if __name__ == '__main__':

    path = "/shared/Dataset/ANNS/CEOs/"

    n = 2340373
    d = 150
    bin_file = path + 'Imagenet_X_2340373_150.bin'
    X = utils.mmap_bin(bin_file, n, d)

    q = 200
    bin_file = path + "Imagenet_Q_200_150.bin"
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
    # np.save(path + "Imagenet_Cosine_k_100_indices.npy", exact_kNN)    # shape: (n, k), dtype: int32
    # exact_kNN = np.load(path + "Imagenet_Cosine_k_100_indices.npy")  # shape: (n, k), dtype: int32
    exact_kNN = np.load(path + "Imagenet_Dot_k_100_indices.npy")  # shape: (n, k), dtype: int32

    k = 10
    exact_kNN = exact_kNN[: , :k]

    #-------------------------------------------------------------------
    probed_vectors = 20
    n_cand = 100
    n_repeats = 2**3
    D = 2**10
    print("\nCEOs")

    utils.ceos_est(exact_kNN, X, Q, k, D, probed_vectors, n_cand, n_repeats, n_threads)

    #-------------------------------------------------------------------
    top_m = 100
    probed_vectors = 160
    n_cand = 2000
    n_repeats = 2**3
    D = 2**10

    print("\ncoCEOs-est")
    utils.coceos_est(exact_kNN, X, Q, k, D, top_m, probed_vectors, n_cand, n_repeats, n_threads)
    #-------------------------------------------------------------------
    top_m = 50
    probed_vectors = 120
    probed_points = top_m
    n_repeats = 2**3
    D = 2**10
    print("\nCEOs-hash")
    utils.ceos_hash(exact_kNN, X, Q, k, D, top_m, probed_vectors, probed_points, n_repeats, n_threads)
    #-------------------------------------------------------------------
    # print("\nHnswLib")
    # utils.hnswMIPS(exact_kNN, X, Q, k, n_threads=n_threads)










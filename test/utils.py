import os

os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["FAISS_NUM_THREADS"] = "32"

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["FAISS_NUM_THREADS"] = "1"

import faiss
import CEOs
import numpy as np
import timeit

def mmap_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    # ('r+' = read/write; 'r' = read-only)
    # mode='c' = copy-on-write (modifications are NOT written back)
    return np.memmap(bin_path, dtype=dtype, mode='c', shape=(num_rows, num_cols))

def getAcc(exact, approx):
    n, k = np.shape(exact)
    result = 0
    for i in range(n):
        result += len(np.intersect1d(exact[i], approx[i])) / k
    return result / n

# Faiss BF
def faissBF(X, Q, k, numThreads):

    n, d = np.shape(X)
    faiss.omp_set_num_threads(numThreads)

    t1 = timeit.default_timer()
    index = faiss.IndexFlatIP(d)   # build the index
    index.add(X)                  # add vectors to the index
    t2 = timeit.default_timer()
    print('Faiss bruteforce index time: {}'.format(t2 - t1))


    t1 = timeit.default_timer()
    distances, exact_kNN = index.search(Q, k)
    t2 = timeit.default_timer()
    print('Faiss bruteforce query time: {}'.format(t2 - t1))

    # Cross-check bf of the first query
    # exactDOT = np.matmul(X, Q[0, :].transpose())  # Exact dot products
    # topK = np.argsort(-exactDOT)[:k]  # topK MIPS indexes

    return exact_kNN

def faissIVF(exact_kNN, X, Q, k=10, n_list = 100, n_probe = 10, n_threads=8):
    """
    Run label propagation clustering using Faiss + iGraph.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of nearest neighbors (default: 10)
    - metric: 'squared_l2' or 'dot_product'

    Returns:
    - labels: list of cluster labels for each point
    """

    X = X.astype(np.float32)
    n, d = X.shape

    # 1. Create FAISS index
    faiss.omp_set_num_threads(n_threads) # This is also default
    nlist = n_list  # the number of clusters
    print("nlist = ", nlist)

    t1 = timeit.default_timer()
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(X)
    index.add(X)

    t2 = timeit.default_timer()
    print('Construction time of Faiss IVF (s): {}'.format(t2 - t1))

    for i in range(1):

        index.nprobe = n_probe + i * 10
        print("nprobe = ", index.nprobe)

        t1 = timeit.default_timer()
        dist, approx_kNN = index.search(Q, k=k)
        print('\tFaiss-IVF query time: {}'.format(timeit.default_timer() - t1))
        print("\tFaiss-IVF Accuracy: ", getAcc(exact_kNN, approx_kNN))


def faiss_approx_kNN_IVFPQ(exact_kNN, X, Q, k=10, n_list = 100, n_subquantizer = 8, n_probe = 10, n_threads=8):
    """
    Run label propagation clustering using Faiss + iGraph.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of nearest neighbors (default: 10)
    - metric: 'squared_l2' or 'dot_product'

    Returns:
    - labels: list of cluster labels for each point
    """

    X = X.astype(np.float32)
    n, d = X.shape

    # 1. Create FAISS index
    faiss.omp_set_num_threads(n_threads) # This is also default
    nlist = n_list  # the number of clusters
    print("nlist = ", nlist) # number of coarse centroids (IVF)

    m = n_subquantizer  # number of PQ subquantizers
    nbits = 8  # bits per subquantizer


    t1 = timeit.default_timer()
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(X)
    index.add(X)

    t2 = timeit.default_timer()
    print('Construction time of Faiss IVFPQ: {}'.format(t2 - t1))

    for i in range(1):

        index.nprobe = n_probe + i * 10
        print("nprobe = ", index.nprobe)

        t1 = timeit.default_timer()
        dist, approx_kNN = index.search(Q, k=k)
        print('\tFaiss-IVFPQ query time: {}'.format(timeit.default_timer() - t1))
        print("\tFaiss-IVFPQ Accuracy: ", getAcc(exact_kNN, approx_kNN))
    
# HNSW
def hnswMIPS(exact_kNN, X, Q, k, efSearch=100, n_threads=8):

    n, d = np.shape(X)

    import hnswlib

    hnsw_m = 64  # The number of neighbors for HNSW. This is typically 32
    efConstruction = 16
    index = hnswlib.Index(space='ip', dim=d)
    print("m = %d, ef = %d" % (hnsw_m, efConstruction))

    index.set_num_threads(n_threads)
    t1 = timeit.default_timer()
    index.init_index(max_elements=n, ef_construction=efConstruction, M=hnsw_m)
    index.add_items(X)
    t2 = timeit.default_timer()
    print('Hnswlib index time: {}'.format(t2 - t1))

    for i in range(1):

        new_efSearch = efSearch + i * 10
        index.set_ef(new_efSearch)
        print("Hnsw efSearch: ", new_efSearch)
        t1 = timeit.default_timer()
        approx_kNN, dist = index.knn_query(Q, k=k)
        print('\tHnsw query time: {}'.format(timeit.default_timer() - t1))
        print("\tHnsw Accuracy: ", getAcc(exact_kNN, approx_kNN))

# scann
def scannMIPS(exact_kNN, X, Q, k):
    n, d = np.shape(X)

    import scann

    print('Constructing the Scann')
    t1 = timeit.default_timer()
    searcher = scann.scann_ops_pybind.builder(X, k, "dot_product").tree(
        num_leaves=5000, num_leaves_to_search=100, training_sample_size=n).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    t2 = timeit.default_timer()
    print('Scann index time (s): {}'.format(t2 - t1))

    leaves_range = 100
    for j in range(1):
        # leaves = 50 + j * 10

        leaves = leaves_range
        t1 = timeit.default_timer()
        approx_kNN, dist = searcher.search_batched(Q, leaves_to_search=leaves, pre_reorder_num_neighbors=500)
        print("\tScann querying with {0} leaves has time: {1: .2f}".format(leaves, timeit.default_timer() - t1))
        print("\tScann Accuracy: ", getAcc(exact_kNN, approx_kNN))

# CEOs
def ceos_est(exact_kNN, X, Q, k, D, probed_vectors, n_cand, n_repeats = 1, n_threads = 8, verbose=True):


    t1 = timeit.default_timer()
    n, d = np.shape(X)
    index = CEOs.CEOs(n, d)
    seed = -1
    top_m = n # not use for CEOs-Est

    index.setIndexParam(D, n_repeats, top_m, n_threads, seed)
    index.build(X)  # X must have d x n
    t2 = timeit.default_timer()

    print('CEOs index time: {}'.format(t2 - t1))
    index.n_cand = n_cand

    # index.set_threads(1)
    for i in range(1):
        index.n_probed_vectors = probed_vectors + i * 5
        print("CEOs top-proj: ", index.n_probed_vectors)
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search(Q, k, verbose)  # search
        print("\tCEOs query time: {}".format(timeit.default_timer() - t1))
        print("\tCEOs accuracy: ", getAcc(exact_kNN, approx_kNN))

# coCEOs-est
def coceos_est(exact_kNN, X, Q, k, D, top_m, probed_vectors, n_cand, n_repeats=1, n_threads=8, verbose=True):

    t1 = timeit.default_timer()
    n, d = np.shape(X)
    index = CEOs.CEOs(n, d)
    seed = -1

    index.setIndexParam(D, n_repeats, top_m, n_threads, seed)

    index.build_coCEOs_Est(X)  # X must have n x d
    t2 = timeit.default_timer()
    print('coCEOs-Est index time: {}'.format(t2 - t1))
    index.n_cand = n_cand

    # index.set_threads(1)
    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probed_vectors = probed_vectors + i * 5
        index.n_probed_points = top_m
        print("top-proj = %d, top-points = %d" % (index.n_probed_vectors, index.n_probed_points))
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search_coCEOs_Est(Q, k, verbose)  # search
        print("\tcoCEOs-Est query time: {}".format(timeit.default_timer() - t1))
        print("\tcoCEOs-Est accuracy: ", getAcc(exact_kNN, approx_kNN))

def ceos_hash(exact_kNN, X, Q, k, D, top_m, probed_vectors, probed_points, n_repeats=1, n_threads=8, verbose=True):

    t1 = timeit.default_timer()
    n, d = np.shape(X)
    index = CEOs.CEOs(n, d)
    seed = -1

    index.setIndexParam(D, n_repeats, top_m, n_threads, seed)
    index.centering = True
    index.build_CEOs_Hash(X)  # X must have d x n

    t2 = timeit.default_timer()
    print('CEOs-Hash index time (s): {}'.format(t2 - t1))

    # index.set_threads(1)
    for i in range(1):
        index.n_probed_vectors = probed_vectors + i * 5
        index.n_probed_points = probed_points
        print("n_probed_vectors = %d, n_probed_points = %d" % (index.n_probed_vectors, index.n_probed_points))
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search_CEOs_Hash(Q, k, verbose)  # search
        print("\tCEOs-Hash query time (s): {}".format(timeit.default_timer() - t1))
        print("\tCEOs-Hash accuracy: ", getAcc(exact_kNN, approx_kNN))

def streamCEOs_test(exact_kNN, X, Q, k, top_m, probed_vectors, n_cand, n_repeats=1, n_threads=8, verbose=True):

    top_m = 500
    probed_vectors = 40
    n_cand = 100
    n_repeats = 2**1
    D = 2**9

    seed = -1

    n, d = np.shape(X)
    t1 = timeit.default_timer()
    index = CEOs.streamCEOs(d)

    index.set_threads(n_threads)
    index.setIndexParam(D, n_repeats, top_m, n_threads, seed)
    index.build(X)  # X must have n x d
    t2 = timeit.default_timer()
    print('coCEOs-stream-est index time: {}'.format(t2 - t1))
    index.n_cand = n_cand

    # index.set_threads(1)
    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probed_vectors = probed_vectors + i * 5
        index.n_probed_points = top_m
        print("top-vectors = %d, top-points = %d" % (index.n_probed_vectors, index.n_probed_points))
        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.estimate_search(Q, k, True)  # search
        print('\tstreamCEOs-est query time: {}'.format(timeit.default_timer() - t1))
        print("\tstreamCEOs-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    # index.set_threads(1)
    top_m = 20
    probed_vectors = 100
    probed_points = top_m

    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probed_vectors = probed_vectors + i * 5
        index.n_probed_points = probed_points
        print("top-vectors = %d, top-points = %d" % (index.n_probed_vectors, index.n_probed_points))
        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.hash_search(Q, k, True)  # search
        print('\tstreamCEOs-hash query time: {}'.format(timeit.default_timer() - t1))
        print("\tstreamCEOs-hash Accuracy: ", getAcc(exact_kNN, approx_kNN))

def streamCEOs_est(X, Q, k, top_m, probed_vectors, n_cand, n_repeats=1, n_threads=8):

    n, d = np.shape(X)

    D = 2**10
    seed = -1
    n_size = int(n / 10)

    t1 = timeit.default_timer()
    index = CEOs.streamCEOs(d)

    index.set_threads(n_threads)
    index.setIndexParam(D, n_repeats, top_m, n_threads, seed)
    index.build(X[0 : n_size, :])
    print('coCEOs-stream index time: {}'.format(timeit.default_timer() - t1))

    exact_kNN = faissBF(X[0: n_size, :], Q, k, n_threads)

    index.n_cand = n_cand
    index.n_probed_vectors = probed_vectors
    index.n_probed_points = top_m
    print("top-proj = %d, top-points = %d" % (index.n_probed_vectors, index.n_probed_points))

    t1 = timeit.default_timer()
    approx_kNN, approx_dist = index.estimate_search(Q, k, True)  # search
    print('\tcoCEOs-stream-est query time: {}'.format(timeit.default_timer() - t1))
    print("\tcoCEOs-stream-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    startTime = timeit.default_timer()
    for i in range(9):

        startIdx = n_size * (i + 1)
        endIdx = startIdx + n_size

        print("startIdx = %d, endIdx = %d" % (startIdx, endIdx))

        new_X = X[startIdx: endIdx, :]

        # add & remove items
        t1 = timeit.default_timer()
        index.update(new_X, n_size)  # X must have d x n
        print('\tstreamCEOs-est update index time: {}'.format(timeit.default_timer() - t1))

        exact_kNN = faissBF(X[startIdx : endIdx, :], Q, k, n_threads)

        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.estimate_search(Q, k, True)  # search
        print('\tstreamCEOs-est query time: {}'.format(timeit.default_timer() - t1))
        print("\tstreamCEOs-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    print('Total time update and query time: {}'.format(timeit.default_timer() - startTime))


def streamCEOs_hash(X, Q, k, top_m, probed_vectors, n_repeats=1, n_threads=8):

    n, d = np.shape(X)

    D = 2**10
    n_size = int(n / 10)
    seed = -1

    t1 = timeit.default_timer()
    index = CEOs.streamCEOs(d)

    index.set_threads(n_threads)
    index.setIndexParam(D, n_repeats, top_m, n_threads, seed)
    index.build(X[0 : n_size, :])
    print('streamCEOs-hash index time: {}'.format(timeit.default_timer() - t1))

    index.n_probed_vectors = probed_vectors
    index.n_probed_points = top_m
    print("top-proj = %d, top-points = %d" % (index.n_probed_vectors, index.n_probed_points))

    exact_kNN = faissBF(X[0: n_size, :], Q, k, n_threads)

    t1 = timeit.default_timer()
    approx_kNN, approx_dist = index.hash_search(Q, k, True)  # search
    print('\tstreamCEOs-hash query time: {}'.format(timeit.default_timer() - t1))
    print("\tstreamCEOs-hash Accuracy: ", getAcc(exact_kNN, approx_kNN))

    startTime = timeit.default_timer()
    for i in range(9):

        startIdx = n_size * (i + 1)
        endIdx = startIdx + n_size
        new_X = X[startIdx: endIdx, :]

        # add & remove items
        t1 = timeit.default_timer()
        index.update(new_X, n_size)  # X must have d x n
        print('\tstreamCEOs-hash update index time: {}'.format(timeit.default_timer() - t1))

        exact_kNN = faissBF(new_X, Q, k, n_threads)

        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.hash_search(Q, k, True)  # search
        print('\tstreamCEOs-hash query time: {}'.format(timeit.default_timer() - t1))
        print("\tstreamCEOs-hash Accuracy: ", getAcc(exact_kNN, approx_kNN))

    print('Total time update and query time: {}'.format(timeit.default_timer() - startTime))

if __name__ == '__main__':

    path = "/shared/Dataset/ANNS/CEOs/"

    n = 17770
    d = 300
    bin_file = path + 'Netflix_X_17770_300.bin'
    X = mmap_bin(bin_file, n, d)

    q = 999
    bin_file = path + "Netflix_Q_999_300.bin"
    Q = mmap_bin(bin_file, q, d)

    # k = 100
    #-------------------------------------------------------------------
    # n_threads = 8
    # # exact_kNN = faissBF(X, Q, k, numThreads)
    # # exact_kNN = exact_kNN.astype(np.int32)
    # # np.save(path + "Netflix_k_100_indices.npy", exact_kNN)    # shape: (n, k), dtype: int32
    # exact_kNN = np.load(path + "Netflix_Dot_k_100_indices.npy")  # shape: (n, k), dtype: int32
    #
    # k = 10
    # exact_kNN = exact_kNN[: , :k]
    #
    # #-------------------------------------------------------------------
    # probed_vectors = 20
    # n_cand = 100
    # n_repeats = 2**1
    # D = 2**9
    # print("\nCEOs")
    #
    # ceos_est(exact_kNN, X, Q, k, D, probed_vectors, n_cand, n_repeats, n_threads)
    #
    # #-------------------------------------------------------------------
    # top_m = 500
    # probed_vectors = 20
    # n_cand = 100
    # n_repeats = 2**1
    # D = 2**9
    #
    # print("\ncoCEOs-est")
    # coceos_est(exact_kNN, X, Q, k, D, top_m, probed_vectors, n_cand, n_repeats, n_threads)
    #
    # # print("\ncoCEOs-stream-est")
    # # coceos_est(exact_kNN, X, Q, k, numThreads, top_point, top_r, n_cand)
    # #
    # n_cand = 500
    # probed_points = 500
    # print("\ncoCEOs-hash")
    # coceos_hash(exact_kNN, X, Q, k, D, top_m, probed_vectors, probed_points, n_cand, n_repeats, n_threads)

    # print("\nHnswLib")
    # hnswMIPS(exact_kNN, X, Q, k, numThreads)










import faiss
import CEOs
import numpy as np
import timeit

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

# HNSW
def hnswMIPS(exact_kNN, X, Q, k, numThreads):

    n, d = np.shape(X)

    import hnswlib

    hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32
    efConstruction = 16
    index = hnswlib.Index(space='ip', dim=d)
    print("m = %d, ef = %d" % (hnsw_m, efConstruction))

    index.set_num_threads(numThreads)
    t1 = timeit.default_timer()
    index.init_index(max_elements=n, ef_construction=efConstruction, M=hnsw_m)
    index.add_items(X)
    t2 = timeit.default_timer()
    print('Hnswlib index time: {}'.format(t2 - t1))

    for i in range(5):

        efSearch = 10 + i * 10
        index.set_ef(efSearch)
        print("   Hnsw efSearch: ", efSearch)
        t1 = timeit.default_timer()
        approx_kNN, dist = index.knn_query(Q, k=k)
        print('   Hnsw query time: {}'.format(timeit.default_timer() - t1))
        print("   Hnsw Accuracy: ", getAcc(exact_kNN, approx_kNN))

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
    print('Scann index time: {}'.format(t2 - t1))

    leaves_range = 50
    for j in range(5):
        leaves = 50 + j * 10

        t1 = timeit.default_timer()
        approx_kNN, dist = searcher.search_batched(Q, leaves_to_search=leaves, pre_reorder_num_neighbors=500)
        print("   Scann querying with {0} leaves has time: {1: .2f}".format(leaves, timeit.default_timer() - t1))
        print("   Scann Accuracy: ", getAcc(exact_kNN, approx_kNN))

# CEOs
def ceosMIPS(exact_kNN, X, Q, k, numThreads, top_r, n_cand):

    import CEOs
    D = 2 ** 10
    repeats = 1

    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)
    index.set_threads(numThreads)

    index.setIndexParam(D, repeats, numThreads, 1)
    index.build(X.transpose())  # X must have d x n
    t2 = timeit.default_timer()

    print('CEOs index time: {}'.format(t2 - t1))
    index.n_cand = n_cand

    index.set_threads(1)
    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probedVectors = top_r + i * 5
        print("   CEOs top-proj: ", index.n_probedVectors)
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search(Q.transpose(), k, True)  # search
        print('   CEOs query time: {}'.format(timeit.default_timer() - t1))
        print("   CEOs Accuracy: ", getAcc(exact_kNN, approx_kNN))

# coCEOs-est
def coceosMIPS_est(exact_kNN, X, Q, k, numThreads, top_m, top_r, n_cand):

    D = 2 ** 10
    repeats = 1
    iBucketSize = 5000

    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)

    index.set_threads(numThreads)
    index.set_coCEOsParam(D, repeats, iBucketSize, numThreads, 1)
    index.build_coCEOs_Est(X.transpose())  # X must have d x n
    t2 = timeit.default_timer()
    print('coCEOs-Est index time: {}'.format(t2 - t1))
    index.n_cand = n_cand

    index.set_threads(1)
    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probedVectors = top_r + i * 5
        index.n_probedPoints = top_m
        print("top-proj = %d, top-points = %d" % (index.n_probedVectors, index.n_probedPoints))
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search_coCEOs_Est(Q.transpose(), k, True)  # search
        print('   coCEOs-Est query time: {}'.format(timeit.default_timer() - t1))
        print("   coCEOs-Est Accuracy: ", getAcc(exact_kNN, approx_kNN))

def coceosMIPS_hash(exact_kNN, X, Q, k, numThreads, top_m, top_r, n_cand):

    D = 2 ** 10
    repeats = 1
    iBucketSize = 5000


    index = CEOs.CEOs(n, d)
    index.set_threads(numThreads)

    t1 = timeit.default_timer()
    index.set_coCEOsParam(D, repeats, iBucketSize, numThreads, 1)
    index.build_coCEOs_Hash(X.transpose())  # X must have d x n
    t2 = timeit.default_timer()
    print('coCEOs-Hash index time: {}'.format(t2 - t1))
    index.n_cand = n_cand

    index.set_threads(1)
    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probedVectors = top_r + i * 5
        index.n_probedPoints = top_m
        print("top-proj = %d, top-points = %d" % (index.n_probedVectors, index.n_probedPoints))
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search_coCEOs_Hash(Q.transpose(), k, True)  # search
        print('   coCEOs-Hash query time: {}'.format(timeit.default_timer() - t1))
        print("   coCEOs-Hash Accuracy: ", getAcc(exact_kNN, approx_kNN))


def coceosMIPS_stream_est(exact_kNN, X, Q, k, numThreads, top_m, top_r, n_cand):

    D = 2 ** 10
    repeats = 1
    iBucketSize = 5000

    t1 = timeit.default_timer()
    index = CEOs.coCEOs(d)

    index.set_threads(numThreads)
    index.setIndexParam(D, repeats, iBucketSize, numThreads, 1, True)
    index.build(X.transpose())  # X must have d x n
    t2 = timeit.default_timer()
    print('coCEOs-stream-est index time: {}'.format(t2 - t1))
    index.n_cand = n_cand

    index.set_threads(1)
    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probedVectors = top_r + i * 5
        index.n_probedPoints = top_m
        print("top-proj = %d, top-points = %d" % (index.n_probedVectors, index.n_probedPoints))
        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.estimate_search(Q.transpose(), k, True)  # search
        print('   coCEOs-stream-est query time: {}'.format(timeit.default_timer() - t1))
        print("   coCEOs-stream-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    index.set_threads(1)
    index.n_cand = 500
    for i in range(1):
        # nprobe is the number of cells in nlist that we will search
        # nprobe < nlist
        index.n_probedVectors = top_r + i * 5
        index.n_probedPoints = top_m
        print("top-proj = %d, top-points = %d" % (index.n_probedVectors, index.n_probedPoints))
        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.hash_search(Q.transpose(), k, True)  # search
        print('   coCEOs-stream-hash query time: {}'.format(timeit.default_timer() - t1))
        print("   coCEOs-stream-hash Accuracy: ", getAcc(exact_kNN, approx_kNN))

# Compile with anaconda

X = np.loadtxt('//home/npha145/Dataset/ANNS/Own//Netflix_X_17770_300.txt')
Q = np.loadtxt('//home/npha145/Dataset/ANNS/Own//Netflix_Q_999_300.txt')
# X = np.loadtxt('//home/npha145/Dataset/ANNS/H2LSH/Gist//_X_1000000_960.txt')
# Q = np.loadtxt('//home/npha145/Dataset/ANNS/H2LSH/Gist//_Q_1000_960.txt')

n, d = np.shape(X)
print("n = %d, d = %d" % (n, d))
q, d = np.shape(Q)
print("q = %d, d = %d" % (q, d))

k = 10

n, d = np.shape(X)
#-------------------------------------------------------------------
numThreads = 4
exact_kNN = faissBF(X, Q, k, numThreads)

top_r = 10
n_cand = 50
print("\nCEOs")
ceosMIPS(exact_kNN, X, Q, k, numThreads, top_r, n_cand)


top_point = 1000
top_r = 20
n_cand = 50

print("\ncoCEOs-est")
coceosMIPS_est(exact_kNN, X, Q, k, numThreads, top_point, top_r, n_cand)
print("\ncoCEOs-stream-est")
coceosMIPS_stream_est(exact_kNN, X, Q, k, numThreads, top_point, top_r, n_cand)

n_cand = 500
print("\ncoCEOs-hash")
coceosMIPS_hash(exact_kNN, X, Q, k, numThreads, top_point, top_r, n_cand)

print("\nHnswLib")
hnswMIPS(exact_kNN, X, Q, k, numThreads)










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
    # print('Faiss bruteforce index time: {}'.format(t2 - t1))


    t1 = timeit.default_timer()
    distances, exact_kNN = index.search(Q, k)
    t2 = timeit.default_timer()
    # print('Faiss bruteforce query time: {}'.format(t2 - t1))

    # Cross-check bf of the first query
    # exactDOT = np.matmul(X, Q[0, :].transpose())  # Exact dot products
    # topK = np.argsort(-exactDOT)[:k]  # topK MIPS indexes

    return exact_kNN

def coceosMIPS_stream_acc(X, Q, k, numThreads, top_m, top_r, n_cand):

    n, d = np.shape(X)

    D = 2**10
    numRepeats = 1
    n_size = int(n / 10)
    iTopPoints = 1000

    n_size = 10000
    index = CEOs.coCEOs(d)

    index.set_threads(numThreads)
    index.setIndexParam(D, numRepeats, iTopPoints, numThreads, 1, False)
    index.build(X[0 : n_size, :].transpose())  # X must have d x n

    index.n_cand = n_cand
    index.n_probedVectors = top_r
    index.n_probedPoints = top_m
    print("top-proj = %d, top-points = %d" % (index.n_probedVectors, index.n_probedPoints))

    exact_kNN = faissBF(X[0: n_size, :], Q, k, numThreads)
    approx_kNN, approx_dist = index.estimate_search(Q.transpose(), k, True)  # search
    print("   coCEOs-stream-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    delSize = 300
    newSize = 300
    index.update(X[n_size : n_size + newSize].transpose(), delSize)  # X must have d x n

    # Exact answer
    exact_kNN = faissBF(X[delSize: n_size + newSize - delSize, :], Q, k, numThreads)

    # Approx
    approx_kNN, approx_dist = index.estimate_search(Q.transpose(), k, True)
    print("   coCEOs-stream-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    index.n_cand = 500
    approx_kNN, approx_dist = index.hash_search(Q.transpose(), k, True)
    print("   coCEOs-stream-hash Accuracy: ", getAcc(exact_kNN, approx_kNN))

def coceosMIPS_stream_est(X, Q, k, numThreads, top_m, top_r, n_cand):

    n, d = np.shape(X)

    D = 2**10
    numRepeats = 1
    n_size = int(n / 10)
    iTopPoints = 1000


    t1 = timeit.default_timer()
    index = CEOs.coCEOs(d)

    index.set_threads(numThreads)
    index.setIndexParam(D, numRepeats, iTopPoints, numThreads, 1, False)
    index.build(X[0 : n_size, :].transpose())  # X must have d x n
    print('coCEOs-stream index time: {}'.format(timeit.default_timer() - t1))

    exact_kNN = faissBF(X[0: n_size, :], Q, k, numThreads)

    index.n_cand = n_cand
    index.n_probedVectors = top_r
    index.n_probedPoints = top_m
    print("top-proj = %d, top-points = %d" % (index.n_probedVectors, index.n_probedPoints))

    t1 = timeit.default_timer()
    approx_kNN, approx_dist = index.estimate_search(Q.transpose(), k, True)  # search
    print('   coCEOs-stream-est query time: {}'.format(timeit.default_timer() - t1))
    print("   coCEOs-stream-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    startTime = timeit.default_timer()
    for i in range(9):


        startIdx = n_size * (i + 1)
        endIdx = startIdx + n_size

        print("startIdx = %d, endIdx = %d" % (startIdx, endIdx))

        new_X = X[startIdx: endIdx, :]

        # add & remove items
        t1 = timeit.default_timer()
        index.update(new_X.transpose(), n_size)  # X must have d x n
        print('   coCEOs-stream-est update index time: {}'.format(timeit.default_timer() - t1))

        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.estimate_search(Q.transpose(), k, True)  # search
        print('   coCEOs-stream-est query time: {}'.format(timeit.default_timer() - t1))

        exact_kNN = faissBF(X[startIdx : endIdx, :], Q, k, numThreads)
        print("   coCEOs-stream-est Accuracy: ", getAcc(exact_kNN, approx_kNN))

    print('Total time update and query time: {}'.format(timeit.default_timer() - startTime))


def coceosMIPS_stream_hash(X, Q, k, numThreads, top_m, top_r, n_cand):

    n, d = np.shape(X)

    D = 2**10
    numRepeats = 1
    n_size = int(n / 10)
    iTopPoints = 1000

    t1 = timeit.default_timer()
    index = CEOs.coCEOs(d)

    index.set_threads(numThreads)
    index.setIndexParam(D, numRepeats, iTopPoints, numThreads, 1, False)
    index.build(X[0 : n_size, :].transpose())  # X must have d x n
    print('coCEOs-stream index time: {}'.format(timeit.default_timer() - t1))

    index.n_cand = n_cand
    index.n_probedVectors = top_r
    index.n_probedPoints = top_m
    print("top-proj = %d, top-points = %d" % (index.n_probedVectors, index.n_probedPoints))

    exact_kNN = faissBF(X[0: n_size, :], Q, k, numThreads)

    t1 = timeit.default_timer()
    approx_kNN, approx_dist = index.hash_search(Q.transpose(), k, True)  # search
    print('   coCEOs-stream-hash query time: {}'.format(timeit.default_timer() - t1))
    print("   coCEOs-stream-hash Accuracy: ", getAcc(exact_kNN, approx_kNN))

    startTime = timeit.default_timer()
    for i in range(9):

        startIdx = n_size * (i + 1)
        endIdx = startIdx + n_size
        new_X = X[startIdx: endIdx, :]

        # add & remove items
        t1 = timeit.default_timer()
        index.update(new_X.transpose(), n_size)  # X must have d x n
        print('   coCEOs-stream-hash update index time: {}'.format(timeit.default_timer() - t1))

        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.hash_search(Q.transpose(), k, True)  # search
        print('   coCEOs-stream-hash query time: {}'.format(timeit.default_timer() - t1))

        exact_kNN = faissBF(new_X, Q, k, numThreads)
        print("   coCEOs-stream-hash Accuracy: ", getAcc(exact_kNN, approx_kNN))

    print('Total time update and query time: {}'.format(timeit.default_timer() - startTime))

# Compile with anaconda

# X = np.loadtxt('//home/npha145/Dataset/ANNS/Own//Netflix_X_17770_300.txt')
# Q = np.loadtxt('//home/npha145/Dataset/ANNS/Own//Netflix_Q_999_300.txt')
X = np.loadtxt('//home/npha145/Dataset/ANNS/H2LSH/Gist//_X_1000000_960.txt')
Q = np.loadtxt('//home/npha145/Dataset/ANNS/H2LSH/Gist//_Q_1000_960.txt')

n, d = np.shape(X)
print("n = %d, d = %d" % (n, d))
q, d = np.shape(Q)
print("q = %d, d = %d" % (q, d))

k = 10

n, d = np.shape(X)
#-------------------------------------------------------------------
numThreads = 32
top_point = 1000
top_r = 20
n_cand = 50

print("\ncoCEOs-Test accuracy")
coceosMIPS_stream_acc(X, Q, k, numThreads, top_point, top_r, n_cand)


print("\ncoCEOs-stream-est")
coceosMIPS_stream_est(X, Q, k, numThreads, top_point, top_r, n_cand)

n_cand = 500
print("\ncoCEOs-stream-hash")
coceosMIPS_stream_hash(X, Q, k, numThreads, top_point, top_r, n_cand)











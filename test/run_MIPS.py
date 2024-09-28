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

# CEOs
def ceosMIPS(exact_kNN, X, Q, k, numThreads, top_r):

    import CEOs
    D = 2 ** 10
    repeats = 1

    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)
    index.set_threads(numThreads)

    index.setIndexParam(D, repeats, numThreads, -1)
    index.build(X.transpose())  # X must have d x n
    t2 = timeit.default_timer()

    print('CEOs index time: {}'.format(t2 - t1))
    index.n_cand = 100
    # index.set_threads(1)
    for i in range(5):
	# increase top_proj will increase the accuracy
        index.top_proj = top_r + i * 5  
        print("   CEOs top-proj: ", index.top_proj)
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search(Q.transpose(), k, True)  # search
        print('   CEOs query time: {}'.format(timeit.default_timer() - t1))
        print("   CEOs Accuracy: ", getAcc(exact_kNN, approx_kNN))
        
def coceosMIPS(exact_kNN, X, Q, k, numThreads, top_m, top_r):

    D = 2 ** 10
    repeats = 1
    # top_m = 5000

    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)

    index.set_threads(numThreads)
    index.set_coCEOsParam(D, repeats, top_m, numThreads, 1)
    index.build_coCEOs(X.transpose())  # X must have d x n
    print('coCEOs index time: {}'.format(timeit.default_timer() - t1))

    index.n_cand = 50 # increase n_cand will increase the accuracy

    # index.set_threads(1)
    for i in range(5):
        # increase top_proj will increase the accuracy
        index.top_proj = top_r + i * 5
        print("   coCEOs top-proj: ", index.top_proj)
        t1 = timeit.default_timer()
        approx_kNN, approx_Dist = index.search_coCEOs(Q.transpose(), k, True)  # search
        print('   coCEOs query time: {}'.format(timeit.default_timer() - t1))
        print("   coCEOs Accuracy: ", getAcc(exact_kNN, approx_kNN))


# CEOs with dequeue
def coceosMIPS_stream(exact_kNN, X, Q, k, numThreads, top_m, top_r):

    n, d = np.shape(X)

    D = 2 ** 10
    numRepeats = 1
    n_size = int(n / 10)

    t1 = timeit.default_timer()
    index = CEOs.coCEOs(d)

    index.set_threads(numThreads)
    index.setIndexParam(D, numRepeats, top_m, numThreads, 1)
    index.build(X[0 : n_size, :].transpose())  # X must have d x n
    print('coCEOs-stream index time: {}'.format(timeit.default_timer() - t1))

    index.n_cand = 50
    index.top_proj = top_r
    print("   coCEOs top-proj: ", index.top_proj)
    t1 = timeit.default_timer()
    approx_kNN, approx_dist = index.search(Q.transpose(), k, True)  # search
    print('   coCEOs query time: {}'.format(timeit.default_timer() - t1))
    print("   coCEOs Accuracy: ", getAcc(exact_kNN, approx_kNN))

    startTime = timeit.default_timer()
    for i in range(9):

        startIdx = n_size * (i + 1)
        endIdx = startIdx + n_size
        new_X = X[startIdx: endIdx, :]

        # add & remove items
        t1 = timeit.default_timer()
        index.add_remove(new_X.transpose(), n_size)  # X must have d x n
        print('coCEOs-stream update index time: {}'.format(timeit.default_timer() - t1))

        t1 = timeit.default_timer()
        approx_kNN, approx_dist = index.search(Q.transpose(), k, True)  # search
        print('   coCEOs query time: {}'.format(timeit.default_timer() - t1))

    print('Total time update and query time: {}'.format(timeit.default_timer() - startTime))


# Compile with anaconda

X = np.loadtxt('//home/npha145/Dataset/ANNS/Own//Netflix_X_17770_300.txt')
Q = np.loadtxt('//home/npha145/Dataset/ANNS/Own//Netflix_Q_999_300.txt')
#X = np.loadtxt('//home/npha145/Dataset/ANNS/H2LSH/Gist//_X_1000000_960.txt')
#Q = np.loadtxt('//home/npha145/Dataset/ANNS/H2LSH/Gist//_Q_1000_960.txt')

n, d = np.shape(X)
print("n = %d, d = %d" % (n, d))
q, d = np.shape(Q)
print("q = %d, d = %d" % (q, d))

k = 10

n, d = np.shape(X)
#-------------------------------------------------------------------
numThreads = 32
exact_kNN = faissBF(X, Q, k, numThreads)


print("\nCEOs")
top_r = 10
ceosMIPS(exact_kNN, X, Q, k, numThreads, top_r)

print("\ncoCEOs")
top_point = 5000
top_r = 20
coceosMIPS(exact_kNN, X, Q, k, numThreads, top_point, top_r)

print("\ncoCEOs-stream")
top_point = 5000
top_r = 20
coceosMIPS_stream(exact_kNN, X, Q, k, numThreads, top_point, top_r)










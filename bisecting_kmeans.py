
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, find
import random
from sklearn.utils import shuffle
from sklearn.metrics import calinski_harabaz_score

def csr_read(fname, ftype="csr", nidx=1):
    r""" 
        Read CSR matrix from a text file. 
        
        \param fname File name for CSR/CLU matrix
        \param ftype Input format. Acceptable formats are:
            - csr - Compressed sparse row
            - clu - Cluto format, i.e., CSR + header row with "nrows ncols nnz"
        \param nidx Indexing type in CSR file. What does numbering of feature IDs start with?
    """
    
    with open(fname) as f:
        lines = f.readlines()
    
    if ftype == "clu":
        p = lines[0].split()
        nrows = int(p[0])
        ncols = int(p[1])
        nnz = long(p[2])
        lines = lines[1:]
        assert(len(lines) == nrows)
    elif ftype == "csr":
        nrows = len(lines)
        ncols = 0 
        nnz = 0 
        for i in xrange(nrows):
            p = lines[i].split()
            if len(p) % 2 != 0:
                raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(p)))
            nnz += len(p)/2
            for j in xrange(0, len(p), 2): 
                cid = int(p[j]) - nidx
                if cid+1 > ncols:
                    ncols = cid+1
    else:
        raise ValueError("Invalid sparse matrix ftype '%s'." % ftype)
    val = np.zeros(nnz, dtype=np.float)
    ind = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.long)
    n = 0 
    for i in xrange(nrows):
        p = lines[i].split()
        for j in xrange(0, len(p), 2): 
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j+1])
            n += 1
        ptr[i+1] = n 
    
    assert(n == nnz)
    
    return csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.float)

mat = csr_read("train.dat.txt")
#print mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = float(1.0/np.sqrt(rsum))
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

csr_l2normalize(mat)
#print mat

def sim(x1, x2):
    sims = x1.dot(x2.T)
    return sims

def findCluster(mat, centroids):
    idx = list()
    simsMatrix = sim(mat, centroids)

    for i in range(simsMatrix.shape[0]):
        row = simsMatrix.getrow(i).toarray()[0].ravel()
        top_indices = row.argsort()[-1]
        #top_values = row[row.argsort()[-1]]
        #print top_indices
        idx.append(top_indices + 1)
    return idx

def findCentroids(mat, idx, k=2):
    centroids = list()
    for i in range(1,k+1):
        indi = [j for j, x in enumerate(idx) if x == i]
        members = mat[indi,:]
        if (members.shape[0] > 1):
            centroids.append(members.toarray().mean(0))
    #print centroids
    centroids_csr = csr_matrix(centroids)
    return centroids_csr

def kmeans2 (mat,matrix, id_list, k):
    print(len(id_list))
    id_list_shuffle = shuffle(id_list, random_state=0)
    print(len(id_list_shuffle))
    init_centroids_index = id_list_shuffle[:2]
    print(init_centroids_index)
    centroids = mat[[init_centroids_index[0],init_centroids_index[1]],:]
    for numb in range(50): 
        print("Iteration " + str(numb) + "\n")
        idx = findCluster(matrix,centroids)
        centroids = findCentroids(matrix,idx)
    id_list1 = []
    id_list2 = []
    for i in range(len(idx)):
        if idx[i] == 1:
            id_list1.append(id_list[i])
        elif idx[i] == 2:
            id_list2.append(id_list[i])
    cluster1 = mat[id_list1,:]
    cluster2 = mat[id_list2,:]
    return id_list1, id_list2, cluster1, cluster2, centroids[0], centroids[1]

from scipy.spatial.distance import euclidean
def bisect(mat, k):
    matrix = mat
    cluster_list = []
    id_list = []
    for i in range(mat.shape[0]):
        id_list.append(i)
    while len(cluster_list) < k-1:
        sse1 = 0
        sse2 = 0
        id_list1, id_list2, cluster1, cluster2, centroids1, centroids2 = kmeans2(mat,matrix,id_list,2)
        for row in cluster1:
            sse1 += (euclidean(row.toarray(),centroids1.toarray()))**2
        for row in cluster2:
            sse2 += (euclidean(row.toarray(),centroids2.toarray()))**2    
        if sse1 < sse2:
            cluster_list.append(id_list1)
            id_list = id_list2
            matrix = cluster2
        else:
            cluster_list.append(id_list2)
            id_list = id_list1
            matrix = cluster1
    cluster_list.append(id_list)
    return cluster_list

result = bisect(mat,7)

output = [0]* mat.shape[0]
for i in range(len(result)):
    for j in range(len(result[i])):
        output[result[i][j]] = i+1
        
print output.count(1)
print output.count(2)
print output.count(3)
print output.count(4)
print output.count(5)
print output.count(6)
print output.count(7)

print("Accuracy Score: ")
print(calinski_harabaz_score(mat.toarray(),output))

f = open("final_output.dat.txt", "w")
f.write("\n".join(map(lambda x: str(x), output)))
f.close()


        



    












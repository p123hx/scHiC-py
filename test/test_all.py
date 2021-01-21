import pytest
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..')))

from scHiCTools import scHiCs
from scHiCTools import scatter

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.')))


def test():
    y = scHiCs(['data/cell_03', 'data/cell_01', 'data/cell_02'],
               reference_genome='mm9', resolution=500000,
               max_distance=4000000, format='shortest_score',
               adjust_resolution=True, chromosomes='except Y',
               operations=['convolution'], kernel_shape=3, keep_n_strata=10,
               store_full_map=True)
    y.test_fast()
    y.plot_contacts()

    emb1 = y.learn_embedding(similarity_method='innerproduct',
                             return_distance=True,
                             embedding_method='mds',
                             aggregation='median')

    emb2 = y.learn_embedding(similarity_method='HiCRep',
                             return_distance=True,
                             embedding_method='mds',
                             aggregation='median')

    emb3 = y.learn_embedding(similarity_method='Selfish',
                             return_distance=True,
                             embedding_method='mds',
                             aggregation='median')

    emb4 = y.learn_embedding(similarity_method='innerproduct',
                             return_distance=True,
                             embedding_method='mds',
                             aggregation='mean')

    emb5 = y.learn_embedding(similarity_method='innerproduct',
                             return_distance=True,
                             embedding_method='tSNE',
                             aggregation='median')

    # emb6 = y.learn_embedding(similarity_method='innerproduct',
    #                      return_distance=True,
    #                      embedding_method='UMAP',
    #                      aggregation='median',
    #                      print_time=False)

    emb7 = y.learn_embedding(similarity_method='innerproduct',
                             return_distance=True,
                             embedding_method='phate',
                             aggregation='median',
                             k=2)

    emb8 = y.learn_embedding(similarity_method='innerproduct',
                             return_distance=True,
                             embedding_method='spectral_embedding',
                             aggregation='median',
                             print_time=False)

    label1 = y.clustering(n_clusters=2,
                          clustering_method='kmeans',
                          similarity_method='innerproduct',
                          aggregation='median',
                          n_strata=None)

    label2 = y.clustering(n_clusters=2,
                          clustering_method='spectral_clustering',
                          similarity_method='innerproduct',
                          aggregation='median',
                          n_strata=None)

    hicluster = y.scHiCluster(dim=2, cutoff=0.8, n_PCs=10, n_clusters=2)

    assert len(set(label1)) == 2
    assert len(set(label2)) == 2
    assert len(set(hicluster[1])) == 2
    assert emb1[0].shape == (3, 2)
    assert emb1[1].shape == (3, 3)
    assert emb2[0].shape == (3, 2)
    assert emb2[1].shape == (3, 3)
    assert emb3[0].shape == (3, 2)
    assert emb3[1].shape == (3, 3)
    assert emb4[0].shape == (3, 2)
    assert emb4[1].shape == (3, 3)
    assert emb5[0].shape == (3, 2)
    assert emb5[1].shape == (3, 3)
    # assert emb6[0].shape==(3,2)
    # assert emb6[1].shape==(3,3)
    assert emb7[0].shape == (3, 2)
    assert emb7[1].shape == (3, 3)
    assert emb8[0].shape == (3, 2)
    assert emb8[1].shape == (3, 3)

    plt.figure()
    plt.subplot(1, 2, 1)
    scatter(emb1[0] * 100, label=label1)
    plt.subplot(1, 2, 2)
    scatter(emb2[0])

    plt.figure()
    scatter(hicluster[0], label=hicluster[1])


def test1():
    fileLst100 = ["../../Nagano/1CDX_cells/1CDX1.1/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.185/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.281/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.38/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.46/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.117/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.202/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.294/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.377/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.465/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.108/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.182/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.263/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.352/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.68/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.154/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.237/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.312/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.392/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.468/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.101/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.186/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.283/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.381/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.464/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.12/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.203/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.295/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.382/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.466/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.11/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.183/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.264/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.353/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.72/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.155/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.24/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.313/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.393/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.47/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.102/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.187/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.284/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.383/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.466/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.121/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.204/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.296/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.383/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.467/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.111/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.185/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.265/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.354/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.73/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.156/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.241/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.314/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.394/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.472/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.103/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.191/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.285/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.384/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.468/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.122/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.205/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.297/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.384/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.468/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.112/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.186/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.266/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.355/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.74/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.157/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.242/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.315/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.396/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.473/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.104/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.192/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.286/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.385/new_adj",
                  "../../Nagano/1CDX_cells/1CDX1.47/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.123/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.206/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.3/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.385/new_adj",
                  "../../Nagano/1CDX_cells/1CDX2.47/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.113/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.187/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.267/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.356/new_adj",
                  "../../Nagano/1CDX_cells/1CDX3.75/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.158/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.243/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.316/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.397/new_adj",
                  "../../Nagano/1CDX_cells/1CDX4.474/new_adj"]
    y = scHiCs(fileLst100,
               reference_genome='mm9', resolution=500000,
               max_distance=4000000, format='shortest_score',
               adjust_resolution=True, chromosomes='except Y',
               operations=['convolution'], kernel_shape=3, keep_n_strata=10,
               store_full_map=False)
    y.test_fast()


test1()

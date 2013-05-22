# -*- coding: utf-8 -*-
""" Data class for crowdsourced training data.
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2013/05/22"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2013 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

class SparseData:
    """ Sparse data class.
    We assume that the feature vectors are sparse.
    
    :IVariables:
        d : int
            dimension of feature vectors
        N : int
            the number of data.
        J : int
            the number of workers.
        x : scipy.sparse.lil_matrix
            N \times d sparse matrix (sp.sparse.lil_matrix((N,d))). Each row corresponds to each feature vector.
        y : scipy.sparse.lil_matrix
            N \times J sparse matrix (sp.sparse.lil_matrix((N,J))). Each column corresponds to each worker's label.
            y[i,j] == 0 if worker j doesn't label data i.
            y[i,j] == 0, or 1 if worker j labels data i.
        data_user : scipy.sparse.lil_matrix
            N \times J sparse matrix (sp.sparse.lil_matrix((N,d))).
            data_user[i,j] = 1 if the j-th worker labels the i-th data.
            data_user[i,j] = 0 if the j-th worker doesn't label the i-th data.
    """
    def __init__(self, x1, y1, data_user1):
        """ Initialization
        """
        self.x = x1
        self.y = y1
        self.data_user = data_user1
    
    def dim(self):
        """ Return the dimension of the feature space (d).
        
        :RType: int
        :Returns: the dimension
        """
        return self.x.shape[1]
    
    def num_data(self):
        """ Return the number of data (N).
        
        :RType: int
        :Returns: the number of data.
        """
        return self.x.shape[0]
    
    def num_worker(self):
        """  Return the number of workers (J).
        
        :RType: int
        :Returns: the number of workers.
        """
        return self.y.shape[1]

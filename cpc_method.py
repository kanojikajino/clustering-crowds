# -*- coding: utf-8 -*-
""" Clustered Personal Classifier Method
Python implementation of the CPC method that appeared in the paper "Clustering Crowds".
"""

# metadata variables
__author__ = "Hiroshi KAJINO <hiroshi.kajino.1989@gmail.com>"
__date__ = "2013/05/22"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2013 Hiroshi Kajino all rights reserved."
__docformat__ = "restructuredtext en"

import scipy as sp
import numpy as np
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
from data import *
from assistant import *

class CPCMethod:
    """ CPC method class
    Given crowd data, estimate the personal classifiers and the target classifier.
    
    :IVariables:
        data : a "SparseData" class object
            see data.py for details.
        N : int
            the number of data
        J : int
            the number of workers excluding the target classifier.
        d : int
            dimension of the feature space.
        mu : float
            coefficient for convex hierarchical clustering regularization (in fact, in the object function, it appeas as mu/J).
        eta : float
            coefficient for regularization of w0.
        rho : float
            coefficient for the augmentation term.
        eps : float
            threshold for optimization w.r.t. all variables.
        phi : numpy.array
            (J+1)**2 \times d array. phi_{ij} = phi[J*i+j,:], a lagrangian multiplier (0 <= i < j <= J).
        W : numpy.array
            an array of length (J+1)*d. W[j*d:(j+1)*d] is a parameter of the j-th worker. W[0:d] is a parameter of the target model.
        U : numpy.array
            (J+1)**2 \times d array. U[J*i+j,:] = W[i,:] - W[j,:] (0 <= i < j <= J)
        M : scipy.sparse.csr_matrix
            (J+1)**2 \times (J+1)**2 sparse.csr_matrix. M[J*i+j, J*i+j] = coefficient for ||W[i,:] - W[j,:]|| (weight matrix for convex hierarchical clustering regularization)
        ij_matrix : scipy.sparse.csr_matrix
            (J+1)**2 \times (J+1)**2 sparse.csr_matrix. ij_matrix[k,k] = 1 if k=J*i+j, i<j
        W_gap : numpy.array
            W_gap[(self.J+1)*i+j,:] = W[i*d:(i+1)*d] - W[j*d:(j+1)*d]
    """
    def __init__(self, data, mu, eta, rho, eps):
        """ Initialization. Set several parameters and initialize variables.
        """
        self.data = data
        self.data.x = self.data.x.tocsr()
        self.data.y = self.data.y.tocsr()
        self.N = self.data.num_data()
        self.J = self.data.num_worker()
        self.d = self.data.dim()
        self.mu = mu/(self.J)
        self.eta = eta
        self.rho = rho
        self.eps = eps
        
        self.phi = np.zeros(((self.J + 1)**2, self.d))
        self.W = np.zeros((self.J + 1)*self.d)
        self.U = np.zeros(((self.J + 1)**2, self.d))
        self.grad = np.zeros((self.J + 1)*self.d)
        self.ij_matrix = sp.sparse.lil_matrix(((self.J + 1)**2, (self.J + 1)**2))
        self.adj_mat = np.zeros((self.J + 1, self.J + 1))
        self.W_gap = np.array(self.ij_matrix * np.kron(self.W.reshape((self.J + 1, self.d)) , np.ones((self.J + 1, 1)))\
                              - self.ij_matrix * np.kron(np.ones((self.J + 1, 1)) , self.W.reshape((self.J + 1, self.d))))
        self.M = sp.sparse.lil_matrix(((self.J + 1)**2, (self.J + 1)**2))
        for i in range(self.J):
            for j in 1 + i + np.arange(self.J - i):
                self.M[(self.J + 1) * i + j, (self.J + 1) * i + j] = 1.0
                self.M[(self.J + 1) * j + i , (self.J + 1) * j + i] = 1.0
                self.ij_matrix[(self.J + 1) * i + j, (self.J + 1) * i + j] = 1.0
        self.M = self.M.tocsr()
        self.ij_matrix = self.ij_matrix.tocsr()
    
    def aug_lag_wrt_W(self, W_in):
        """ Calculate the augmented lagrangian that is related to W given U, phi, and w0 are fixed.
        
        :Parameters:
            W_in : numpy.array
                the same as self.W
        
        :Returns:
            The value of the augmented lagrangian w.r.t. W.
        """
        W_mat = W_in.reshape(((self.J + 1), self.d))
        self.calc_W_gap(W_in, 1)
        return (self.data.data_user.multiply(np.logaddexp(0, self.data.x * (W_mat[1:(self.J + 1),:]).transpose())) - self.data.y.multiply(self.data.x * (W_mat[1:(self.J + 1),:]).transpose())).sum()\
               + 0.5 * self.eta * (np.linalg.norm(W_in[0:self.d])**2)\
               - (self.phi * self.W_gap).sum()\
               - self.rho * (self.U * self.W_gap).sum()\
               + 0.5 * self.rho * (self.W_gap * self.W_gap).sum()    
    
    def calc_W_gap(self, W_in, triangle):
        """ Update self.W_gap with W_in.
        
        :Parameters:
            W_in : numpy.array
                the same as self.W
            triangle : int
                if triangle == 0, then update all the elements. if triangle == 1, then update the half of the elements using ij_matrix.
        """
        if triangle == 1:
            self.W_gap = np.array(self.ij_matrix * (np.repeat(W_in.reshape((self.J + 1, self.d)), self.J + 1, axis = 0)\
                                                    - np.tile(W_in.reshape((self.J + 1, self.d)), (self.J + 1, 1))))
        if triangle == 0:
            self.W_gap = np.array( np.repeat(W_in.reshape((self.J + 1, self.d)), self.J + 1, axis = 0)\
                                   - np.tile(W_in.reshape((self.J + 1, self.d)), (self.J + 1, 1)))
    
    def aug_lag(self):
        """ Calculate the complete augmented lagrangian.
        
        :Returns:
            The value of the complete augmented lagrangian.
        """
        aug_lag = self.aug_lag_wrt_W(self.W)
        U_temp = np.array(self.ij_matrix * np.matrix(self.U))
        U_norm = sp.sqrt((U_temp * U_temp).sum(axis=1)).reshape(((self.J + 1)**2, 1))
        return aug_lag + self.mu * ((self.M * np.matrix(U_norm)).sum())\
               + (self.phi * U_temp).sum() + 0.5 * self.rho * (U_temp * U_temp).sum()

    def optimize_wrt_W(self):
        """ Optimize with respect to W using the l-bfgs-b method.
        """
        W_init = self.W
        [self.W, f_val, other] = sp.optimize.fmin_l_bfgs_b(self.aug_lag_wrt_W, W_init, fprime = self.grad_W, pgtol = 0.1)
    
    def grad_W(self,W_in):
        """ Calculate the gradient of aug_lag_wrt_W with respect to W.
        
        :Parameters:
            W_in : numpy.array
                the same as self.W
        
        :RType: numpy.array. an array of length (self.J + 1) * d.
        :Returns: The gradient of aug_lag_wrt_W.
        """
        W = W_in.reshape((self.J + 1, self.d))
        W_part = W[1:self.J + 1, 0:self.d]
        sigma_x_w = logist_sparse(self.data.x, W_part.transpose()) #(N,J)
        
        W_sum = W.sum(axis=0)
        self.grad = (self.rho * (self.J + 1)) * W_in - np.tile(self.rho * (W_sum), (self.J + 1,))
        self.grad[0:self.d] += self.eta * W_in[0:self.d]
        
        temp = (((self.data.y - self.data.data_user.multiply(sigma_x_w)).transpose()) * self.data.x)
        self.grad[self.d:self.d * (self.J + 1)] -= np.array(temp).reshape(self.d * self.J)
        
        U_plus_phi = self.phi + self.rho * self.U
        for j in range(self.J + 1):
            self.grad[self.d * j:self.d * (j + 1)] -= U_plus_phi[(self.J + 1) * j:(self.J + 1) * (j + 1)].sum(axis=0)
        
        return self.grad    

    def optimize_wrt_U(self):
        """ Optimize with respect to U.
        """
        W_mat = self.W.reshape((self.J + 1, self.d))
        self.calc_W_gap(self.W, 0)
        v = self.rho * (self.W_gap) - self.phi #v[J*i+j,:] = v_ij
        v_norm = np.sqrt((v * v).sum(axis=1)) #v_norm[J*i+j] = |v_ij|
        v_norm = v_norm.reshape((v_norm.shape[0], 1))
        
        multiplier_for_v = np.nan_to_num(np.array((1.0 - self.mu*np.array(self.M.sum(axis=1)) / v_norm) / self.rho)).reshape((self.J + 1)**2)
        
        U_lil_multiplier = sp.sparse.lil_matrix(((self.J + 1)**2, (self.J + 1)**2))
        U_lil_multiplier.setdiag(np.maximum(0, multiplier_for_v))
        self.U = np.array(U_lil_multiplier * np.matrix(v))
    
    def update_phi(self):
        """ Update phi.
        """
        W_mat = self.W.reshape((self.J + 1, self.d))
        self.calc_W_gap(self.W, 0)
        self.phi = self.phi + self.rho * (self.U - self.W_gap)
        
    def optimize(self):
        """ Optimize the objective function.
        """
        iter = 0
        convergence = 0
        func_value_old = self.aug_lag()
        func_value_new = func_value_old
        print "init_obj_func =", func_value_new
        while (iter==0) or convergence==0:
            func_value_old = func_value_new
            iter += 1
            self.optimize_wrt_W()
            self.optimize_wrt_U()
            self.update_phi()
            func_value_new = self.aug_lag()
            print "func_val =", func_value_new
            if np.abs(func_value_old - func_value_new)/func_value_new < self.eps:
                convergence = 1
                print "certificate:", np.abs(func_value_old - func_value_new)/func_value_new
            
        print "opt has finished!!!"

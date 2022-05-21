import sys
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.spatial.distance import pdist
from scipy.interpolate import griddata
import time
import sklearn
from sklearn.cluster import KMeans
import scipy
import scipy.stats as stats
import scipy.special

np.random.seed(0)
tf.set_random_seed(0)

class SGPTF:
    #self.tf_log_lengthscale: log of RBF lengthscale
    #self.tf_log_tau: log of inverse variance
    #self.tf_y: observed entries
    #self.tf_U: embeddings,
    #m: no. of frequencies
    #y: observed tensor entries
    #B: batch-size
    #lr: learning rate
    #ind: entry indices
    #nvec: dim of each mode
    #R: rank
    def __init__(self, ind, nvec, R, y, m, B, lr):
        self.m = m
        self.y = y.reshape([y.size,1])
        self.ind = ind
        self.B = B
        self.learning_rate = lr
        self.nmod =  len(nvec)
        self.R = R
        self.nvec = nvec
        self.alpha = 1.5
        #define socialbility in each mode
        self.tf_v_hat = [tf.Variable(scipy.special.logit(np.random.rand(self.nvec[k])), dtype=tf.float32) for k in range(self.nmod)]
        #generate parameterized sample for v_hat, v, and omega
        log_v = [tf.math.log_sigmoid(self.tf_v_hat[k]) for k in range(self.nmod)]
        log_v_minus = [tf.math.log_sigmoid(-self.tf_v_hat[k]) for k in range(self.nmod)]
        cum_sum = [tf.cumsum(log_v_minus[k], exclusive=True) for k in range(self.nmod)]
        log_omega = [log_v[k] + cum_sum[k] for k in range(self.nmod)]
        #define locations in each mode
        self.tf_theta_hat = [tf.Variable(scipy.special.logit(np.random.rand(self.nvec[k], self.R - 1)), dtype=tf.float32) for k in range(self.nmod)]
        #generate parameterized sample for \theta
        log_theta = [tf.math.log_sigmoid(self.tf_theta_hat[k]) for k in range(self.nmod)]
        #concate sociability and locations
        self.tf_U = [tf.concat([tf.expand_dims(log_omega[k], -1), tf.exp(log_theta[k])], 1) for k in range(self.nmod)]

        #constuct Phi
        self.d = self.nmod*self.R
        #frequencies
        self.tf_S = tf.Variable(np.random.randn(self.m, self.d)/(2*np.pi), dtype=tf.float32)
        #self.tf_S = tf.constant(np.random.randn(self.m, self.d)/(2*np.pi), dtype=tf.float32)
        #self.tf_S = tf.Variable(np.random.rand(self.m, self.d)/(2*np.pi), dtype=tf.float32)

        #weight vect.
        #self.w_mu = tf.Variable(np.zeros([2*self.m,1]), dtype=tf.float32)
        self.w_mu = tf.Variable(np.random.rand(2*self.m,1), dtype=tf.float32)
        self.w_L = tf.Variable(1.0/self.m*np.eye(2*self.m), dtype=tf.float32)
        w_Ltril = tf.matrix_band_part(self.w_L, -1, 0)

        #inverse variance
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)
        tau = tf.exp(self.tf_log_tau)
        self.N = y.size
        #A mini-batch of observed entry indices
        self.tf_sub = tf.placeholder(tf.int32, shape=[None, self.nmod])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, 1])

        tf_inputs = tf.concat([ tf.gather(self.tf_U[k], self.tf_sub[:, k])  for k in range(self.nmod) ], 1)
        Phi = tf.matmul(tf_inputs, tf.transpose(self.tf_S))*tf.constant(2*np.pi, dtype = tf.float32)
        #N by 2M
        Phi = tf.concat([tf.cos(Phi), tf.sin(Phi)],1)
        log_edge_prob = tf.concat([ tf.gather(tf.expand_dims(log_omega[k], -1), self.tf_sub[:,k]) for k in range(self.nmod) ], 1)
        #stochastic ELBO
        ELBO = -2*(np.pi**2)*tf.reduce_sum(self.tf_S*self.tf_S) \
               -0.5*self.m*(tf.trace(tf.matmul(self.w_mu, tf.transpose(self.w_mu)) + tf.matmul(w_Ltril, tf.transpose(w_Ltril)))) \
               + 0.5*tf.reduce_sum(tf.log(tf.pow(tf.diag_part(w_Ltril), 2))) \
               + (self.alpha-1.0)*tf.math.add_n([tf.reduce_sum(log_v_minus[k]) for k in range(self.nmod)])\
               + 0.5*self.N*self.tf_log_tau \
                -0.5*tau*self.N/self.B*(tf.reduce_sum(tf.pow(self.tf_y - tf.matmul(Phi, self.w_mu),2)) + tf.reduce_sum(tf.pow(tf.matmul(Phi, w_Ltril),2))) \
                + self.N/self.B*tf.reduce_sum(log_edge_prob)
        
        self.loss = -ELBO
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.minimizer = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def pred(self, test_ind):
        tf_inputs = tf.concat([tf.gather(self.tf_U[k], (test_ind[:,k]))  for k in range(self.nmod) ], 1)
        Phi = tf.matmul(tf_inputs, tf.transpose(self.tf_S))*tf.constant(2*np.pi, dtype = tf.float32)
        Phi = tf.concat([tf.cos(Phi), tf.sin(Phi)],1)
        pred_mean = tf.matmul(Phi, self.w_mu)
        return pred_mean

    def test(self, test_ind):
        pred_mean = self.pred(test_ind)
        print('tau = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess))))
        res = self.sess.run(pred_mean, {self.tf_y:self.y})
        res = res.reshape(res.size)
        return res

    def get_inputs(self, ind):
        tf_inputs = tf.concat([tf.gather(self.tf_U[k], (ind[:,k]))  for k in range(self.nmod) ], 1)
        res = self.sess.run(tf_inputs, {self.tf_y:self.y})
        return res


    def callback(self, loss):
        print('Loss:', loss)

    def save_embeddings(self,file_name):
        res = self.sess.run(self.tf_U)
        for i, r in enumerate(res):
            np.savetxt(file_name+"_%d.txt"%(i),r)

    def train(self, ind_test, y_test, nepoch = 10):
        print('start')
        print(self.N/self.B)
        for iter in range(nepoch):
            curr = 0
            s = np.random.permutation(self.N)
            #self.ind = self.ind[s]
            #self.y = self.y[s]
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                tf_dict = {self.tf_sub:self.ind[batch_ind,:], self.tf_y:self.y[batch_ind]}
                #tf_dict = {self.tf_sub:self.ind[curr:curr+self.B,:], self.tf_y:self.y[curr:curr+self.B]}
                curr = curr + self.B
                self.sess.run(self.minimizer,feed_dict = tf_dict)
            #print('epoch %d finished'%iter)
            if iter%5==0:
                y_pred = self.test(ind_test)
                mse =  np.mean( np.power(y_pred - y_test, 2) )
                y_pred = self.test(self.ind)
                mse_train =  np.mean( np.power(y_pred - self.y.flatten(), 2) )
                print ('epoch %d, train mse = %g, test mse = %g'%(iter, mse_train, mse))


        print('tau = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess))))
        print('w = ')
        print(self.w_mu.eval(session=self.sess))
        print(self.w_L.eval(session=self.sess))
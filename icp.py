import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from util import *

def find_affine_transform(x, y):
    """ 
    given two sets of points, find the affine transform which relates the two 
    point sets.

    inputs:
        x: numpy array, shape (N, 3) = measured point set
        y: numpy array, shape (N, 3) = model point set
    
    outputs:
        affine transform in terms of quaternions: q
        q[0:3] = quaternions representing rotation angles
        q[4:6] = translation
    
    """
    N = x.shape[0]

    # center point sets about their means
    y_mu = np.mean(y, axis=0).reshape(1,3)
    x_mu = np.mean(x, axis=0).reshape(1,3)
    x_c = x - x_mu
    y_c = y - y_mu

    # cross covariance matrix:
    S = 1/N * np.dot(x_c.T, y_c)
    S = np.dot(x_c.T, y_c)

    # make an anti-symmetric matrix
    A = S - S.T

    # this vector for some reason
    D = np.array([
        [A[1][2]],
        [A[2][0]],
        [A[0][1]] ])

    # symmetric matrix Q
    Q = np.zeros(shape=(4,4))
    Q[0][0] = np.trace(S)
    Q[0,1:] = D.T
    Q[1:,0] = D.T
    L = S + S.T - np.identity(3) * np.trace(S)
    Q[1:,1:] = L

    # eigenvector corresponding to maximum eigenvalue of Q is used as optimal rotation
    vals, vecs = np.linalg.eig(Q)

    # rotation angles 
    # NOTE: need to use the COLUMNS of the eigenvector matrix
    rpy = quat_to_euler(vecs.T[0]) 

    # rotation matrix defined by quaternion with largest eigenvalue
    R = quat_rot(vecs.T[0])

    # translation vector
    x_mu_rot = np.dot(R, x_mu.T.reshape(3,))
    t = (y_mu - x_mu_rot)[0]

    return rpy, t, R


def find_closest_points(x, y):
    """
    re-order points in x so that distance between points in x and y is minimized
    based on:
    https://github.com/ClayFlannigan/icp/blob/master/icp.py
    """

    # TODO: determine if this is correct...
    # normalize and 0-center
    # x_c = (x - np.mean(x, axis=0))/ np.std(x, axis=0) * np.sqrt(x.shape[0])
    # y_c = (y - np.mean(y, axis=0))/ np.std(y, axis=0) * np.sqrt(y.shape[0])
    # M = np.dot(x_c.T, y_c)
    # vals, vecs = np.linalg.eig(M)
    # print('vals:\n', vals)
    # print('vecs:\n', vecs)

    n = NearestNeighbors(n_neighbors=1)
    n.fit(x)

    distances, indices = n.kneighbors(y, return_distance=True)

    x_new = x[indices].reshape(x.shape)

    return x_new



def icp(x, y, threshold = 0.01, max_iter = 60, log_frequency = 15, ideal=False):
    """
    inputs:
        - x: numpy array (N x 3), source point cloud (new measurement)
        - y: numpy array (N x 3), destination point cloud (previous measurement)
        - threshold: convergence threshold
        - max_iter: maximum number of iterations
        - log_frequency: # of iterations between plotting source/destination pointclouds
        - ideal: if true, 

    outputs:
        - affine_est: numpy array (3 x 4), estimate of affine transform between source 
        and destination pointclouds at each iteration

        - dist: list of average pt-pt distances for each iteration
    """

    # shape stuff...
    assert y.shape == x.shape 
    assert x.shape[1] == 3

    num_iter = 0
    dist = []
    dist.append(np.inf)

    # cumulative rotations and translations
    t_total = np.zeros(shape=(3,))    
    R_total = np.identity(3)
    affine_est = np.zeros(shape=(3,4))
    affine_est[:, :3] = R_total
    affine_est[:, 3] = t_total

    # recovered pointcloud
    x_est = x
    a_est = np.zeros(shape=(3,))
    t_est = np.zeros(shape=(3,))
    ests = []

    while num_iter < max_iter:
        # re-order points via PCA to minimized distance between point clouds
        if not ideal:
            x_est = find_closest_points(x_est, y)

        # estimates of rotation (angles a), translation (distances t)
        a_est, t_est, _ = find_affine_transform(x_est, y)

        x_homog = to_homogeneous(x_est)
        x_est, affine_update = affine(x_homog, a_est, t_est, inverse=True)

        # update total translations and rotations
        t_total += -1*affine_update[:,3]                   # translation update addative
        R_total = np.dot(affine_update[:,:3].T, R_total)   # rotation update left multiply
        affine_est[:, :3] = R_total
        affine_est[:, 3] = t_total

        dist.append(np.mean((x_est - y)**2))
        ests.append(affine_est)

        # printout and plots
        num_iter += 1
        print('\niteration:', num_iter)
        print('estimated rotation:\n', a_est)
        print('estimated translation:\n', t_est)
        print('average point-to-point distance:\n', dist)
        print('affine estimate:\n', affine_est)
        if num_iter % log_frequency == 0:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(y[:,0], y[:,1], y[:,2], label='original')
            ax.scatter(x[:,0], x[:,1], x[:,2], label='transformed')
            ax.scatter(x_est[:,0], x_est[:,1], x_est[:,2], label='estimated', s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title('Iteration: %i. Avg pt-pt dist: %f' %(num_iter, dist[num_iter-1]))
            plt.legend(loc=2)
            plt.show()
    
        # check for convergence
        if np.abs(dist[num_iter-1] - dist[num_iter-2]) <  threshold:
            break

    if num_iter >= max_iter:
        print('\nmax iterations exceeded before convergence')

    if np.abs(dist[num_iter-1]-dist[num_iter-2]) < threshold:
        print('\nconverged after ' + str(num_iter) + ' iterations:', dist[num_iter-1])

    # return ests, dist
    return affine_est, dist





















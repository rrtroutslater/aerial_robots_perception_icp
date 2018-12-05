import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from util import *
import features;

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

def get_correspondence_points(x, y, criterium):
    """
    Selects the points in x where the distance is minimum for each point in y given a criterium.

    Note: Does not check if the size of x and y are the same. Consequence: if #x > #y, at least two points in x
    will have the same correspondence for sure. If #x < #y, however, no problem should arise. In principle, we
    expect x and y to have N points.

    inputs:
        - x: numpy array (N x 3), source point cloud (new measurement)
        - y: numpy array (N x 3), destination point cloud (previous measurement)
        - criterium: string, criterium for correspondence. Can be 'distance', 'feature' or 'both'
    outputs:
        - y_new: numpy array (N x 3), points in y that match the points in x (in sequence)
    """

    # Convert X and Y to list of points
    Y = [];
    X = [];
    for i in range(y.shape[0]):
        Y.append(y[i]);
    for i in range(x.shape[0]):
        X.append(x[i]);

    # Compute features
    X_feat = features.smoothnessFeatures(X);
    Y_feat = features.smoothnessFeatures(Y);
    
    # Matching
    match_list = [];
    for i in range(x.shape[0]):
        idx = -1;
        if(criterium == "distance"):
            idx,_ = features.matchByDistance(X[i], Y);
        elif(criterium == "feature"):
            idx,_ = features.matchByFeature(X_feat[i], Y_feat);
        elif(criterium == "both"):
            idx,_ = features.matchWeighted(X[i], X_feat[i], Y, Y_feat, w_d=0.5, w_f=0.5);
        else:
            raise Exception("Invalid criterium");
        if(idx == -1):
            raise Exception("An error occurred: idx was not changed");
        match_list.append(idx);

    # Correspondence cloud - size of x
    y_new = np.zeros_like(x);
    for i in range(y_new.shape[0]):
        idx = match_list[i];
        y_new[i] = y[idx];
    return y_new;


def icp(x, y, threshold = 0.01, max_iter = 60, log_frequency = 15, ideal=False, guess=None):
    """
    inputs:
        - x: numpy array (N x 3), source point cloud (new measurement)
        - y: numpy array (N x 3), destination point cloud (previous measurement)
        - threshold: float, convergence threshold
        - max_iter: integer, maximum number of iterations
        - log_frequency: integer, # of iterations between plotting source/destination pointclouds
        - ideal: boolean, if true all points are already aligned
        - guess: affine matrix used as initial guess for a transform. * Use the inverse tranformation, not the true 

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
    dist_current = np.inf

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

    # allow for an initial guess of affine transform
    if num_iter == 0 and guess is not None:
        x_homog = to_homogeneous(x_est);
        print('\n \n Applying guess');
        assert guess.shape == affine_est.shape;
        R_T = np.linalg.inv(guess[:,:3]);
        guess[:,:3] = R_T;
        affine_est = guess;
        x_est = np.dot(affine_est, x_homog.T).T;

    while num_iter < max_iter and abs(dist_current) > threshold:
        # re-order points via PCA to minimized distance between point clouds
        if not ideal:
            y_est = get_correspondence_points(x_est, y, 'both')
        else:
            y_est = y;

        # estimates of rotation (angles a), translation (distances t)
        a_est, t_est, _ = find_affine_transform(x_est, y_est)
        x_homog = to_homogeneous(x_est)

        next_x, affine_update = affine(x_homog, a_est, t_est, inverse=True)
        # update total translations and rotations
        t_total += -1*affine_update[:,3]                   # translation update addative
        R_total = np.dot(affine_update[:,:3].T, R_total)   # rotation update left multiply
        affine_est[:, :3] = R_total
        affine_est[:, 3] = t_total

        print( "\naffine est shape:", affine_est.shape)

        dist.append(np.mean((next_x - y)**2))
        ests.append(affine_est)
        dist_current = dist[num_iter]
        if(num_iter != 0):
            dist_current -= dist[num_iter-1];

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
            ax.scatter(next_x[:,0], next_x[:,1], next_x[:,2], label='estimated', s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title('Iteration: %i. Avg pt-pt dist: %f' %(num_iter, dist[num_iter-1]))
            plt.legend(loc=2)
            plt.show()

        x_est = next_x;
    
    if num_iter >= max_iter:
        print('\nmax iterations exceeded before convergence')

    if dist[num_iter-1] - dist[num_iter-2] < threshold:
        print('\nconverged after ' + str(num_iter) + ' iterations:', dist[num_iter-1])

    return affine_est, dist





















from icp import * 
from util import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


# initialize test data ----------------------------------------------- #
# initial coordinates, cube
N = 360
Y = cube_points(N, add_noise=False)

sigma = np.array([
    [0.1, 0, 0],
    [0, 0.2, 0],
    [0, 0.05, 0]
])
# Y = np.random.multivariate_normal(mean=np.array([0,0,0]), cov=sigma, size=N)

# convert to homogeneous coordinates
Y_homog = to_homogeneous(Y)

# rotation angles, translations:
# for ideal case: large angles, translations ok
# angles = np.array([0.4, 0.6, 0.5])
# translation = np.array([0.5, 0.3, 0.1])

# for non-ideal case: small angles, small translations still
# angles = np.array([0.04, 0.06, 0.05])
# translation = np.array([0.05, 0.03, 0.01])

# # affine transform
# X, affine_actual = affine(Y_homog, angles, translation, inverse=False)
# print('actual affine transform:\n', affine_actual)

# shuffle coordinates
# np.random.shuffle(X)
# -------------------------------------------------------------------- #

# icp ---------------------------------------------------------------- #
# affine_est, dist = icp(X, Y, threshold=0.0000005, max_iter=45, log_frequency=1, ideal=True)
# affine_est, dist = icp(X, Y, threshold=0.00005, max_iter=45, log_frequency=5, ideal=False)
# print('difference between actual and estimated transform:\n', affine_actual - affine_est)

# convergence plots -------------------------------------------------- #
ds = []
ests = []
angles = np.array([0.04, 0, 0])
translation = np.array([0, 0, 0])
d = []

for i in range(1, 5):
    affine_mse = []
    angles = i * angles
    X, affine_actual = affine(Y_homog, angles, translation, inverse=False)

    # guess for initial affine transformation)
    guess = np.array([
        [1, 0, 0, 0],
        [0, np.cos(i*0.04), -np.sin(i*0.04), 0],
        [0, np.sin(i*0.04), np.cos(i*0.04), 0]
    ])

    affine_est, d = icp(X, Y, threshold=0.00005, max_iter=45, log_frequency=2, 
                        ideal=False, guess=guess)

    ds.append(d)

# plt.subplot(2, 1, 2)
plt.figure()
plt.plot(np.log(ds[0][:6]), label='phi= 0.04')
plt.plot(np.log(ds[1][:6]), label='phi= 0.08')
plt.plot(np.log(ds[2][:6]), label='phi= 0.16')
plt.plot(np.log(ds[3][:6]), label='phi= 0.20')
plt.title('Average pt-pt distance')
plt.xlabel('Iteration')
plt.ylabel('ln(MSE)')
plt.legend(loc=4)   
plt.show()







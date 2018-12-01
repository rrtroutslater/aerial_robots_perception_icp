import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from modules.icp import * 
from modules.util import *

USE_GUESS=True;

def test():
    # Initial coordinates, cube
    N = 360
    Y = cube_points(N, add_noise=False)

    # Convert to homogeneous coordinates
    Y_homog = to_homogeneous(Y)

    angles = np.array([0.1 * np.pi, 0, 0])
    translation = np.array([-0.3, 0.1, 0.1])

    # Test convergence with multiple guesses
    if(not USE_GUESS):
        X, affine_actual = affine(Y_homog, angles, translation, inverse=False)
        np.random.shuffle(X);
        affine_est, d = icp(\
            X, Y, threshold=5e-8, max_iter=45, log_frequency=3, 
            ideal=False, guess=None\
        )
        print("Estimated affine matrix:");
        print(affine_est);
    else:
        # Since the guess attempts to transform the modified cloud to the original, its
        # angles and translation should be on the opposite direction
        guess_angles = -0.85 * angles;
        guess_translation = -1.25 * translation;
        X, affine_guess = affine(Y_homog, guess_angles, guess_translation, inverse=False)
        affine_est, d = icp(\
            X, Y, threshold=5e-20, max_iter=45, log_frequency=4, 
            ideal=False, guess=affine_guess\
        )

    return ;

def main():
    test();
    return ;

if __name__=="__main__":
    main();




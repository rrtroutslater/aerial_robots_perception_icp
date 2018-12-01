import numpy as np 


def to_homogeneous(A):
    """ 
    converts (Nxd) matrix to homogeneous coordinates 
    """
    N, d = A.shape
    out = np.ones(shape=(N,d+1))
    out[:,:-1] = A 

    return out


def quat_rot(q):
    """ 
    rotation matrix in terms of quaternions 
    """
    [q0, q1, q2, q3] = q.tolist()

    R = np.zeros(shape=(3,3))

    R[0][0] = q0**2 + q1**2 - q2**2 - q3**2
    R[0][1] = 2*(q1*q2 - q0*q3)
    R[0][2] = 2*(q1*q3 + q0*q2)

    R[1][0] = 2*(q1*q2 + q0*q3)
    R[1][1] = q0**2 + q2**2 - q1**2 - q3**2
    R[1][2] = 2*(q2*q3 - q0*q1)

    R[2][0] = 2*(q1*q3 - q0*q2)
    R[2][1] = 2*(q2*q3 + q0*q1)
    R[2][2] = q0**2 + q3**2 - q1**2 - q2**2

    return R


def quat_to_euler(q):
    """ 
    transform quaternion to euler angles, based on wikipedia formuas
    """
    [q0, q1, q2, q3] = q.tolist()

    phi = np.arctan2(
        (2*(q0*q1+q2*q3)),
        (1 - 2*(q1**2 + q2**2))
    )

    theta = np.arcsin(
        2*(q0*q2 + q3*q1)
    )

    psi = np.arctan2(
        (2*(q0*q3 + q1*q2)),
        (1 - 2*(q2**2 + q3**2))
    )

    return np.array([phi, theta, psi])


def affine(A, angles, translation, inverse=False):
    """ 
    affine transform
    phi, theta, psi = euler angles
    x, y, z = translations
    assumes A is in homogeneous coordinates
    """
    phi = angles[0]
    theta = angles[1]
    psi = angles[2]

    x = translation[0]
    y = translation[1]
    z = translation[2]

    N = A.shape[0]
    out = np.zeros(shape=(N,3))

    R = np.array([
        [np.cos(theta) * np.cos(psi), 
        np.cos(theta) * np.sin(psi), 
        -np.sin(theta), 
        x],
        
        [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
        np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
        np.sin(phi) * np.cos(theta), 
        y],

        [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
        np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
        np.cos(phi) * np.cos(theta), 
        z] ])

    # for inverse affine transform
    R_T = np.zeros(shape=(3,4))
    R_T[:,:3] = np.linalg.inv(R[:,:3])
    R_T[0][3] = x
    R_T[1][3] = y
    R_T[2][3] = z

    if inverse:
        out = np.dot(R_T, A.T)
        return out.T, R_T
    else:
        out = np.dot(R, A.T)
        return out.T, R

    return


def cube_points(N, add_noise=False):
    """ make a numpy cube array, assumes N divisible by 12 """
    Y = np.zeros(shape=(N,3))
    seg = np.linspace(0,1,N/12)
    Y[:int(N/12),0] = seg
    Y[int(N/12):2*int(N/12),1] = seg
    Y[2*int(N/12):3*int(N/12),2] = seg

    Y[3*int(N/12):4*int(N/12),0] = seg
    Y[3*int(N/12):4*int(N/12),1] = 1

    Y[4*int(N/12):5*int(N/12),0] = seg
    Y[4*int(N/12):5*int(N/12),2] = 1

    Y[5*int(N/12):6*int(N/12),1] = seg
    Y[5*int(N/12):6*int(N/12),2] = 1

    Y[6*int(N/12):7*int(N/12),2] = seg
    Y[6*int(N/12):7*int(N/12),1] = 1

    Y[7*int(N/12):8*int(N/12),1] = seg
    Y[7*int(N/12):8*int(N/12),0] = 1

    Y[8*int(N/12):9*int(N/12),2] = seg
    Y[8*int(N/12):9*int(N/12),0] = 1

    Y[9*int(N/12):10*int(N/12),0] = seg
    Y[9*int(N/12):10*int(N/12),1] = 1
    Y[9*int(N/12):10*int(N/12),2] = 1

    Y[10*int(N/12):11*int(N/12),1] = seg
    Y[10*int(N/12):11*int(N/12),0] = 1
    Y[10*int(N/12):11*int(N/12),2] = 1

    Y[11*int(N/12):12*int(N/12),2] = seg
    Y[11*int(N/12):12*int(N/12),0] = 1
    Y[11*int(N/12):12*int(N/12),1] = 1

    if add_noise:
        noise = np.random.randn(N, 3) / 100
        Y += noise

    return Y















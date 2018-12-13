import numpy as np;
import time
from math import *;


def smoothnessFeatures(P):
    """
    Compute loam smoothness features.

    For each point of the point cloud, computes loam smoothness features.

    Args:
        P: point cloud <list of ndarray>.

    Returns:
        F: computed features <list of float>.
    """
    F = [];
    # F = np.zeros(shape=(len(P), P[0].shape[0]))
    for i in range( len(P) ) :
        diff = 0;
        f = 0;
        for j in range ( len(P) ) :
            if(i != j):
                diff+= (P[i] - P[j]);
        if(P[i][0] == 0 and P[i][1] == 0 and P[i][2] == 0):
            f = np.inf;
        else:
            f = np.linalg.norm(diff) / ( float(len(P)) * np.linalg.norm(P[i]) );
        F.append(f);
        # F[i] = f
    return F;


def filterByFeatures(P, F, f_min):
    """
    Remove points that are not from interest.

    If feature value is large (f > f_min), then point is most likely not planar (edge points). Ignores inf points.

    Args:
        P: point cloud <list of ndarray>.
        F: list of features <list of float>.
        f_min: threshold between planar and non-planar points (float).

    Returns:
        G: reduced point cloud <list of ndarray>
    """
    G = [];
    H = [];
    for i in range(len(F)):
        if(F[i] > f_min):
            G.append(P[i]);
            H.append(F[i]);
    return G;


def matchByDistance(p, G):
    """
    Matches a point with the closest point from a point cloud. Ignores inf points.

    Args:
        p: point, numpy array shape (1,3).
        G: point cloud to be matched, numpy array shape (n,3).
    
    Returns:
        idx: the index of the closest point to p <int>.
        d: the distance between the closest point and p <float>.
    """
    d_line = np.linalg.norm(p - G, axis=1)
    idx = np.argmin(d_line)
    return idx, d_line[idx];


def matchByFeature(f, H):
    """
    Finds the index of the closest feature in H with respect to f.

    Args:
        f: feature <float>.
        H: features on another point cloud <list of float>.
    
    Returns:
        idx: index of the closest feature <int>.
        d: the distance between the closest feature and the input <float>.
    """
    idx = 0;
    d = sqrt( (f-H[idx]) ** 2 );
    for j in range(1, len(H)) :
        d_line = sqrt((f-H[j])**2);
        if( d > d_line ):
            idx = j;
            d = d_line;
    return idx,d;


def matchWeighted(p,f, G, H, w_d=0.5, w_f=0.5):
    """
    Finds the index where the weighted average of distance between p and G and the distance between f and H is the smallest

    Args:
        p: point <ndarray>.
        f: feature <float>.
        G: point cloud to be matched <list of ndarray>.
        H: features on another point cloud <list of float>.
        w_d: weight of the distance factor <float>.
        w_f: weight of the feature factor <float>.

    Returns:
        idx: index of the closest feature <int>.
        d: the distance between the closest index and the input <float>.
    """
    idx = 0;
    d_f = sqrt((f-H[idx])**2);
    d_d = np.linalg.norm(p-G[idx]);
    norm = w_d + w_f; # forces the average to be between d_d and d_f
    d = ( (w_d * d_d) + (w_f * d_f) )/norm;
    for j in range(1, len(G)):
        d_f_line = sqrt((f-H[j]) ** 2);
        d_d_line = np.linalg.norm(p-G[j]);
        d_line = ( (w_d * d_d_line) + (w_f * d_f_line) )/norm;
        if( d > d_line ):
            idx = j;
            d = d_line;
    return idx,d;

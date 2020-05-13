import numpy as np

# Euclidean distance.


def euc_dist(pt1, pt2):

    square = 0
    for i in range(0, len(pt1)):
        square = square + (pt1[i] -pt2[i])*(pt1[i] -pt2[i])

    return np.sqrt(square)
    # return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))


def norm(pt, normVal):

    if normVal == 2:
        square = 0
        for i in range(0, len(pt)):
            square = square + pt[i]*pt[i]

        return np.sqrt(square)

    if normVal == -1:
        absVal = -1

        for i in range(0, len(pt)):
            if absVal < abs(pt[i]):
                absVal = abs(pt[i])

        return absVal


def normTrajectory(trajectory, normVal):

    listVals = []

    for points in trajectory:
        listVals += [norm(points, normVal)]

    return listVals


def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, P, Q), euc_dist(P[i], Q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, P, Q), euc_dist(P[0], Q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i-1, j, P, Q), _c(ca, i-1, j-1, P, Q), _c(ca, i, j-1, P, Q)), euc_dist(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""


def frechetDist(P, Q):
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P)-1, len(Q)-1, P, Q)


def normalDist(traj1, traj2, timeSteps):

    distance = 0
    part_distance = 0
    for index in range(0, len(timeSteps)-1):
        vector1 = traj1[index]
        vector2 = traj2[index]
        part_distance = euc_dist(vector1, vector2)

        distance += part_distance*(timeSteps[index+1]-timeSteps[index])

    return distance

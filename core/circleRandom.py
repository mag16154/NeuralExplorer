# generate random points in a circle

import numpy as np
import pylab as plt

''''https://gist.github.com/makokal/6883810#file-circle_random-py'''
'''http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/'''


def generate_points_in_circle(n_samples=100, dim=2):
    num_samples = n_samples
    r_scale = 1

    points = []

    if dim == 2:
        # make a simple unit circle
        theta = np.linspace(0, 2*np.pi, num_samples)
        # a, b = r_scale * np.cos(theta), r_scale * np.sin(theta)

        # generate the points
        # theta = np.random.rand((num_samples)) * (2 * np.pi)
        r = r_scale * np.random.rand((num_samples))
        # print("radius {}".format(r))
        # print("theta {}".format(theta))
        x, y = r * np.cos(theta), r * np.sin(theta)

        for idx in range(len(x)):
            point = [x[idx], y[idx]]
            points.append(point)
    elif dim == 3:
        for idx in range(n_samples):
            u = np.random.normal(0, 1)
            v = np.random.normal(0, 1)
            w = np.random.normal(0, 1)
            r = np.random.rand(1) ** (1. / 3)
            norm = (u * u + v * v + w * w) ** (0.5)
            (x, y, z) = r * (u, v, w) / norm
            point = [x, y, z]
            points.append(point)
        # print(points)
    else:
        for idx in range(n_samples):
            u = np.random.normal(0, 1, dim)  # an array of d normally distributed random variables
            norm = np.sum(u ** 2) ** (0.5)
            r = np.random.rand(1) ** (1.0 / dim)
            x = r * u / norm
            point = []
            for d in range(dim):
                point.append(x[d])
            points.append(point)
    return points

    # print("Cosines {}".format(np.cos(theta)))
    # print("Sines {}".format(np.sin(theta)))
    # print(x, y)

    # plt.figure(figsize=(7, 6))
    # plt.plot(a, b, linestyle='-', linewidth=2, label='Circle')
    # plt.plot(x, y, marker='o', label='Samples')
    # plt.ylim([-1.5, 1.5])
    # plt.xlim([-1.5, 1.5])
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.show(block=True)

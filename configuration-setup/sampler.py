import numpy as np
import random as rand
from frechet import norm


def generateRandomState(lowerBoundArray, upperBoundArray):

    dimensions = len(upperBoundArray)
    state = []

    for dim in range(0, dimensions):
        state += [rand.uniform(lowerBoundArray[dim], upperBoundArray[dim])]

    return state


def generateRandomStates(numStates, lowerBoundArray, upperBoundArray):
    # given an array on upper and lower bound and the number of states
    # generates an array of the number of states with uniform random distribution.

    iterator = 0
    states = []

    for iterator in range(0, numStates):
        states += [generateRandomState(lowerBoundArray, upperBoundArray)]

    return states


def generateRandomVectors(dimensions):

    return np.random.rand(dimensions, dimensions)


def generateRandomCoefficients(dimensions):

    return np.random.rand(dimensions)


def generateSuperpositionSampler(lowerBoundArray, upperBoundArray):

    magnitude = 3

    dimensions = len(upperBoundArray)

    center = generateRandomState(lowerBoundArray, upperBoundArray)

    vectors = generateRandomVectors(dimensions)
    coeffs = generateRandomCoefficients(dimensions)

    # print("Center {} vectors {} coeffs {}".format(center, vectors, coeffs))
    states = [center]

    iterator = 0
    for iterator in range(0, dimensions):
        states += [center + vectors[iterator]]

    print("states: {}".format(states))
    fringe = np.zeros(dimensions)

    # fringe += center

    for iterator in range(0, dimensions):
        fringe += coeffs[iterator]*vectors[iterator]

    normFringe = norm(fringe, 2)

    # print (normFringe)

    coeffs = (2*coeffs)/normFringe

    fringe = (2*fringe)/normFringe

    fringe += center

    states += [fringe]

    # print(states)

    # print(coeffs)

    return states,vectors,coeffs

# TODO: plotter for the random states generated.

import numpy as np

from sampler import generateRandomStates, generateSuperpositionSampler
from ODESolver import generateTrajectories, plotTrajectories
from verisigSystems import DnnController, Plant
import random
from os import path
import os.path


class configuration(object):
    def __init__(self, timeStep=0.01, steps=100, samples=50, dynamics='None', dimensions=2, lowerBound=[],
                upperBound=[], embedding_dimension=2):
        self.timeStep = timeStep
        self.steps = steps
        self.samples = samples
        self.dynamics = dynamics
        self.dimensions = dimensions
        self.lowerBoundArray = lowerBound
        self.upperBoundArray = upperBound
        self.trajectories = []
        self.states = []

        # Only these are in the configuration.
        # Getting trajectories is the duty of configuration.

    def setDynamics(self, dynamics):
        self.dynamics = dynamics

    def setLowerBound(self, lowerBound):
        self.lowerBoundArray = lowerBound

    def setUpperBound(self, upperBound):
        self.upperBoundArray = upperBound

    def setSamples(self, samples):
        self.samples = samples

    def setSteps(self, steps):
        self.steps = steps

    def setTimeStep(self, timeStep):
        self.timeStep = timeStep

    def setEmbeddingDimension(self, dim):
        self.embedding_dimension = dim

    def getDynamics(self):
        return self.dynamics

    def generateTrajectories(self, samples=None, r_states=None):
        if self.dynamics is 'MC':
            dnn_cntrl_fname = '/home/manishg/Research/cps-falsification/verisig/examples/mountain_car/' + 'sig16x16.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant('MC', dnn_controller_obj, None, self.steps)
            if r_states is not None:
                trajectories = plant.getSimulations(states=r_states)
                return trajectories
            else:
                if samples is None:
                    states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
                    self.trajectories = plant.getSimulations(states=states)
                    return self.trajectories
                else:
                    states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
                    trajectories = plant.getSimulations(states=states)
                    return trajectories
        elif self.dynamics is 'Quadrotor':
            dnn_cntrl_fname = '/home/manishg/Research/cps-falsification/verisig/examples/quadrotor/' + 'tanh20x20.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant('Quadrotor', dnn_controller_obj, None, self.steps)
            if r_states is not None:
                trajectories = plant.getSimulations(states=r_states)
                return trajectories
            else:
                if samples is None:
                    states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
                    self.trajectories = plant.getSimulations(states=states)
                    return self.trajectories
                else:
                    states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
                    trajectories = plant.getSimulations(states=states)
                    return trajectories
        elif self.dynamics is 'ABS':
            dnn_cntrl_fname = '/home/manishg/Research/cps-falsification/verisig/examples/ABS/' + 'controller.yml'
            dnn_tf_fname = '/home/manishg/Research/cps-falsification/verisig/examples/ABS/' + 'transform.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            dnn_transform_obj = DnnController(dnn_tf_fname, 2)
            plant = Plant('ABS', dnn_controller_obj, dnn_transform_obj, self.steps)
            if r_states is not None:
                trajectories = plant.getSimulations(states=r_states)
                return trajectories
            else:
                if samples is None:
                    states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
                    self.trajectories = plant.getSimulations(states=states)
                    return self.trajectories
                else:
                    states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
                    trajectories = plant.getSimulations(states=states)
                    return trajectories
        elif self.dynamics is 'VehiclePlatoon' or self.dynamics is 'SpikingNeuron':
            plant = Plant(self.dynamics, None, None, self.steps)
            if r_states is not None:
                trajectories = plant.getSimulations(states=r_states)
                return trajectories
            else:
                if samples is None:
                    states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
                    self.trajectories = plant.getSimulations(states=states)
                    return self.trajectories
                else:
                    states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
                    trajectories = plant.getSimulations(states=states)
                    return trajectories

        if r_states is not None:
            r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
            trajectories = generateTrajectories(self.dynamics, r_states, r_time)
            return trajectories
        else:
            if samples is None:
                self.storeTrajectories()
                return self.trajectories
                # idx = random.randint(0, self.samples - 1)
                # print("random state: {}".format(self.states[idx]))
            else:
                r_states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
                r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
                trajectories = generateTrajectories(self.dynamics, r_states, r_time)
                return trajectories

    def showTrajectories(self, trajectories=None, xindex=0, yindex=1):
        if trajectories is None:
            plotTrajectories(self.trajectories, xindex=xindex, yindex=yindex)
        else:
            plotTrajectories(trajectories, xindex=xindex, yindex=yindex)

    def storeStatesRandomSample(self):
        self.states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)

    def storeTrajectories(self):

        if self.states == [] :
            self.storeStatesRandomSample()

        assert not (self.states == [])
        # assert self.lowerBoundArray is not [] and self.upperBoundArray is not []
        self.time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
        self.trajectories = generateTrajectories(self.dynamics, self.states, self.time)

    def dumpTrajectoriesStates(self, eval_var):
        f_name = "../models/states_"
        f_name = f_name + eval_var + "_"
        f_name = f_name + self.dynamics
        f_name = f_name + ".txt"
        if path.exists(f_name):
            os.remove(f_name)
        states_f = open(f_name, "w")
        for idx in range(len(self.states)):
            states_f.write(str(self.states[idx]))
            states_f.write("\n")
        states_f.close()


# config1 = configuration()
# time step default = 0.01, number of steps default = 100
# dimensions default = 2, number of sample default = 50

# config1.setSteps(100)
# config1.setSamples(10)

# config1.setDynamics('Vanderpol')
# config1.setLowerBound([1.0, 1.0])
# config1.setUpperBound([50.0, 50.0])

# config1.setDynamics('Brussellator')
# config1.setLowerBound([1.0, 1.0])
# config1.setUpperBound([2.0, 2.0])

# config1.setDynamics('Lorentz')
# config1.setLowerBound([1.0, 1.0, 1.0])
# config1.setUpperBound([10.0, 10.0, 10.0])

# for i in range(0, 1):

# config1.storeTrajectories()
# config1.rawAnalyze()
# print (config1.eigenElbow)
# print (config1.noProminetEigVals)

# for j in range(0, config1.noProminetEigVals):
# print (config1.sortedVectors[j])
# config1.showLogEigenValues()
# config1.showTrajs()

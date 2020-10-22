import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Differential Equation presented as rhsODE

from diffEq import rhsODE


def getTimeArray(timeStep, steps):
	time = np.linspace(0, timeStep*steps, steps+1)
	return time


def generateTrajectory(dynamics, state, time):

	return odeint(rhsODE, state, time, args=(dynamics,))


def generateTrajectories(dynamics, states, time):
	# given an array of states and the time step, this generates the set of trajectories

	trajectories = []
	
	for i in range(0, len(states)):
		traj = generateTrajectory(dynamics, states[i], time)
		trajectories += [traj]
	
	return trajectories


def plotTrajectories(trajectories, xindex=0, yindex=1, dimwise=False):

	# given trajectories, plots each of them

	if dimwise is True:
		plt.figure(1)
		plt.xlabel('x' + str(xindex))

		for i in range(0, len(trajectories)):
			traj = trajectories[i]
			plt.plot(traj[:, xindex], 'b-')

		plt.show()

		plt.figure(1)
		plt.xlabel('x' + str(yindex))

		for i in range(0, len(trajectories)):
			traj = trajectories[i]
			plt.plot(traj[:, yindex], 'b-')

		plt.show()

		# zindex = 2
		# plt.figure(1)
		# plt.xlabel('x' + str(zindex))
		#
		# for i in range(0, len(trajectories)):
		# 	traj = trajectories[i]
		# 	plt.plot(traj[:, zindex], 'b-')
		#
		# plt.show()

	else:
		plt.figure(1)
		plt.xlabel('x'+str(xindex))
		plt.ylabel('x'+str(yindex))

		for i in range(0, len(trajectories)):
			traj = trajectories[i]
			plt.plot(traj[:, xindex], traj[:, yindex], 'b-')

		plt.show()


import numpy as np
import matplotlib.pyplot as plt

def getEigVal(covMatrix):

	eigval, eigvec = np.linalg.eig(covMatrix)

	return eigval

def getSortedEigVal(covMatrix):

	eigval, eigvec = np.linalg.eig(covMatrix)
	sortedEig = sorted(eigval, reverse=True)
	return sortedEig


def getNormalizedEigVal(covMatrix):

	eigval, eigvec = np.linalg.eig(covMatrix)
	sortedEig = sorted(eigval, reverse=True)
	normalizedSorted = sortedEig/sortedEig[0]
	return normalizedSorted


def listThresholdEigVal(eigVals, threshold):

	prominentEigValues = []
	for iterator in range(0, len(eigVals)):
		if eigVals[iterator] > threshold :
			prominentEigValues += [eigVals[iterator]]

	return prominentEigValues


def listKEigVal(eigVals, k):

	return eigVals[0:k]

def visualizeEigVals(eigVals, gridding=False):

	plt.figure(1)
	plt.xlabel('index')
	plt.ylabel('eigen value')
	plt.grid(gridding)

	plt.plot(range(0, len(eigVals)), np.absolute(eigVals), 'b-')
	plt.show()

	#return eigVals


def visualizeEigValsLog(eigVals, gridding=True):

	plt.figure(1)
	plt.xlabel('index')
	plt.ylabel('eigen value')

	plt.yscale('log')
	plt.grid(True)

	plt.plot(range(0, len(eigVals)), np.absolute(eigVals), 'b-')
	plt.show()

	#return eigVals

def visualizePerturbations(perturbation, gridding=True):

	plt.figure(1)
	plt.xlabel('index')
	plt.ylabel('perturbation')

	plt.grid(True)
	#plt.grid('log')

	for elements in perturbation:
		plt.plot(range(0, len(elements)), elements, 'b-')

	plt.show()

def getEigElbow(eigVals):

	loggedEig = np.log(np.absolute(eigVals))

	parameter = 4
	threshold = 5
	for i in range(parameter,len(loggedEig)-parameter-1):
		sumsBeforei = 0
		sumsAfteri = 0
		for k in range(0,parameter):
			sumsBeforei += loggedEig[i-k-1] - loggedEig[i-k]
		for k in range(0,parameter):
			sumsAfteri += loggedEig[i+k] - loggedEig[i+k+1]

		if sumsBeforei > threshold*sumsAfteri:
			break

	return eigVals[0:i+int(0.5*parameter)]

import numpy as np


def rhsODE(state, time, dynamics):
	# return the value that is given by the following function
	# change this function to anything else you want.

	if dynamics is 'None':
		return vanderpol(state, time)
	if dynamics is 'Gravity':
		return gravity(state, time)
	if dynamics is 'Vanderpol':
		return vanderpol(state, time)
	if dynamics is 'Brussellator':
		return brussellator(state, time)
	if dynamics is 'Jetengine':
		return jetengine(state, time)
	if dynamics is 'Lorentz':
		return lorentz(state, time)
	if dynamics is 'Buckling':
		return buckling(state, time)
	if dynamics is 'Lotka':
		return lotkavolterra(state, time)
	if dynamics is 'Lacoperon':
		return lacoperon(state, time)
	if dynamics is 'Roesseler':
		return roesseler(state, time)
	if dynamics is 'Steam':
		return steam(state, time)
	if dynamics is 'SpringPendulum':
		return springpendulun(state, time)
	if dynamics is 'CoupledVanderpol':
		return coupledVanderpol(state, time)
	if dynamics is 'HybridLinearOscillator':
		return hybridLinearOscillator(state, time)
	if dynamics is 'HybridLinearOscillator1':
		return hybridLinearOscillator1(state, time)
	if dynamics is 'HybridLinearOscillator2':
		return hybridLinearOscillator2(state, time)
	if dynamics is 'SmoothHybridLinearOscillator':
		return smoothHybridLinearOscillator(state, time)
	if dynamics is 'SmoothHybridLinearOscillator1':
		return smoothHybridLinearOscillator1(state, time)
	if dynamics is 'SmoothHybridLinearOscillator2':
		return smoothHybridLinearOscillator2(state, time)
	if dynamics is 'RegularOscillator':
		return regularOscillator(state, time)
	if dynamics is 'BiologicalModel1':
		return biologicalModel_1(state, time)
	if dynamics is 'BouncingBall':
		return bouncingball(state, time)
	if dynamics is 'Rober':
		return rober(state, time)
	if dynamics is 'E5':
		return e5(state, time)
	if dynamics is 'Orego':
		return orego(state, time)
	if dynamics is 'LaubLoomis':
		return laubLoomis(state, time)
	if dynamics is 'SA_Nonlinear':
		return sa_nonlinear(state, time)
	if dynamics is 'DampedOsc':
		return dampedOscillator(state, time)
	if dynamics is 'OscParticle':
		return oscParticle(state, time)
	if dynamics is 'AdaptiveCruiseControl':
		return adaptiveCruise(state, time)


def vanderpol(state, time):

	# Vanderpol oscilltor
	# dxdt = y; dydt = mu*(1-x*x)*y -x;
	# typical value of mu = 1
	# state = [x,y]

	x = state[0]
	y = state[1]

	mu = 1
	dxdt = y
	dydt = mu*(1-x*x)*y - x

	rhs = [dxdt, dydt]
	return rhs


def bouncingball(state, time):

	x = state[0]
	v = state[1]

	dxdt = 0
	dvdt = 0

	if x == 0 and v <= 0:
		dxdt = v
		dvdt = -9.81
	else:
		dxdt = v
		dvdt = -100*x - 4*v - 9.81

	# dxdt = 0
	# dvdt = 0
	#
	# if x == 0 and v <= 0:
	# 	v = -0.75 * v
	# 	dxdt = v
	# 	dvdt = -9.81
	# else:
	# 	dxdt = v
	# 	dvdt = -9.81

	rhs = [dxdt, dvdt]
	return rhs


def gravity(state, time):
	# gravity
	# dxdt = y; dydt = -9.8;
	# typical value of A = 1, B = 1.5
	# state = [x,y]

	x = state[0]
	y = state[1]

	dxdt = y
	dydt = -9.8 - 0.1 * y

	rhs = [dxdt, dydt]
	return rhs


def jetengine(state, time):
	# jet-engine dynamics
	# dxdt = -1*y - 1.5*x*x - 0.5*x*x*x - 0.5; dydt = 3*x - y;
	# state = [x,y]

	x = state[0]
	y = state[1]

	dxdt = -1 * y - 1.5 * x * x - 0.5 * x * x * x - 0.5
	dydt = 3 * x - y

	rhs = [dxdt, dydt]
	return rhs


def brussellator(state, time):
	# Brussellator
	# dxdt = A + x^2y -Bx - x; dydt = Bx - x^2y;
	# typical value of A = 1, B = 1.5
	# state = [x,y]

	x = state[0]
	y = state[1]

	A = 1
	B = 1.5
	dxdt = A + x * x * y - B * x - x
	dydt = B * x - x * x * y

	rhs = [dxdt, dydt]
	return rhs


def buckling(state, time):
	# Buckling Column
	# dxdt = y; dydt = 2x - x*x*x - 0.2*y + 0.1;
	# state = [x,y]

	x = state[0]
	y = state[1]

	dxdt = y
	dydt = 2*x - x*x*x - 0.2*y + 0.1

	rhs = [dxdt, dydt]
	return rhs


def lotkavolterra(state, time):
	# Predator prey also known as Lotka-Volterra
	# dxdt = x*(alpha - beta*y); dydt = -1*y*(gamma - delta*x)
	# state = [x,y]
	# typical values of alpha = 1.5, beta = 1, gamma = 3, delta = 1

	x = state[0]
	y = state[1]

	alpha = 1.5
	beta = 1
	gamma = 3
	delta = 1

	dxdt = x*(alpha - beta*y)
	dydt = -1*y*(gamma - delta*x)

	rhs = [dxdt, dydt]
	return rhs


def lacoperon(state, time):
	# Lac-operon model
	# dIidt = -0.4*Ii*Ii*((0.0003*G*G + 0.008) / (0.2*Ii*Ii + 2.00001) ) + 0.012 + (0.0000003 * (54660 - 5000.006*Ii) *
	# (0.2*Ii*Ii + 2.00001)) / (0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)
	# DGdt = -0.0006*G*G + (0.000000006*G*G + 0.00000016) / (0.2*Ii*Ii + 2.00001) +
	# (0.0015015*Ii*(0.2*Ii*Ii + 2.00001)) / (0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)
	# state = [Ii, G]

	Ii = state[0]
	G = state[1]

	dIidt = -0.4*Ii*Ii*((0.0003*G*G + 0.008) / (0.2*Ii*Ii + 2.00001)) + 0.012 + (0.0000003 * (54660 - 5000.006*Ii) *
												(0.2*Ii*Ii + 2.00001)) / (0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)
	dGdt = -0.0006*G*G + (0.000000006*G*G + 0.00000016) / (0.2*Ii*Ii + 2.00001) + (0.0015015*Ii*(0.2*Ii*Ii + 2.00001)) / \
												(0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)

	rhs = [dIidt, dGdt]
	return rhs


def roesseler(state, time):
	# Roesseler attractor dynamics
	# dxdt = -y-z; dydt = x+a*y; dzdt = b + z*(x-c)
	# typical values of a = 0.2, b = 0.2, c = 5.7
	# state = [x, y, z]

	a = 0.2
	b = 0.2
	c = 5.7

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = -y - z
	dydt = x + a*y
	dzdt = b + z*(x-c)

	rhs = [dxdt, dydt, dzdt]
	return rhs


def steam(state, time):
	# Steam Governer system
	# dxdt = y; dydt = z*z*sin(x)*cos(x) - sin(x) - epsilon*y; dzdt = alpha*(cos(x) - beta)
	# asymptotic stability when epsilon > 2*alpha*(beta^(3/2))
	# typical values of epsilon = 3, alpha = 1, beta = 1

	epsilon = 3
	alpha = 1
	beta = 1

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = y
	dydt = z*z*np.sin(x)*np.cos(x)
	dzdt = alpha*(np.cos(x) - beta)

	rhs = [dxdt, dydt, dzdt]
	return rhs


def lorentz(state, time):
	# lorenz attractor dynamics
	# dxdt = sigma*(y - x); dydt = x*(rho-z) - y; dzdt = x*y - beta*z;
	# state = [x,y,z]
	# typically sigma = 10; rho = 8.0/3.0; beta = 28;

	sigma = 10.0
	rho = 28.0
	beta = 8.0 / 3.0

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = sigma * (y - x)
	dydt = x * (rho - z) - y
	dzdt = x * y - beta * z

	rhs = [dxdt, dydt, dzdt]
	return rhs


def springpendulun(state, time):
	# Spring pendulum system
	# drdt = vr; dthetadt = omega;
	# dvrdt = r*omega*omega + g*np.cos(theta) - k*(r-L)
	# domegadt = (-1.0/r)*(2*vr*omega+g*np.sin(theta))
	# state = [r, theta, vr, omega]
	# k = 2; L = 1; g = 9.8

	k = 2.0
	L = 1.0
	g = 9.8

	r = state[0]
	theta = state[1]
	vr = state[2]
	omega = state[3]

	drdt = vr
	dthetadt = omega
	dvrdt = r*omega*omega + g*np.cos(theta) - k*(r - L)
	domegadt = (-1.0/r) * (2*vr*omega + g*np.sin(theta))

	rhs = [drdt, dthetadt, dvrdt, domegadt]
	return rhs


def coupledVanderpol(state, time):
	# Coupled Vanderpol oscillator
	# dx1dt = y1; dy1dt = (1 - x1*x1)*y1 - x1 + (x2-x1)
	# dx2dt = y2; dy2dt = (1 - x2*x2)*y2 - x2 + (x1 -x2)
	# state = [x1, y1, x2, y2]

	x1 = state[0]
	y1 = state[1]
	x2 = state[2]
	y2 = state[3]

	dx1dt = y1
	dy1dt = (1 - x1*x1)*y1 - x1 + (x2 - x1)
	dx2dt = y2
	dy2dt = (1 - x2*x2)*y2 - x2 + (x1 - x2)

	rhs = [dx1dt, dy1dt, dx2dt, dy2dt]
	return rhs


def hybridLinearOscillator1(state, time):
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]

	dx1dt = 0
	dy1dt = 0

	dx1dt = y1
	dy1dt = -1*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def hybridLinearOscillator2(state, time):
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]

	dx1dt = 0
	dy1dt = 0

	dx1dt = 2*y1
	dy1dt = -2*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def hybridLinearOscillator(state, time):
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]

	dx1dt = 0
	dy1dt = 0

	if x1 <= 0:
		dx1dt = y1
		dy1dt = -1*x1
	else:
		dx1dt = 2*y1
		dy1dt = -2*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def smoothHybridLinearOscillator1(state, time):
	# smoothen the dynamics by performing a linear interpolation.
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]
	scale = 1.0

	g1 = 1.0/(1.0 + np.exp(scale*x1))
	g2 = 1.0/(1.0 + np.exp(scale*-1*x1))

	lambda1 = g1/(g1+g2)

	dx2dt = 0
	dy2dt = 0

	dx1dt = y1
	dy1dt = -1*x1

	fdx1dt = lambda1*dx1dt + (1-lambda1)*dx2dt
	fdy1dt = lambda1*dy1dt + (1-lambda1)*dy2dt

	rhs = [fdx1dt, fdy1dt]
	# print rhs

	return rhs


def smoothHybridLinearOscillator2(state, time):
	# smoothen the dynamics by performing a linear interpolation.
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]
	scale = 1.0

	g1 = 1.0/(1.0 + np.exp(scale*x1))
	g2 = 1.0/(1.0 + np.exp(scale*-1*x1))

	lambda1 = g1/(g1+g2)

	dx1dt = 0
	dy1dt = 0
	dx2dt = 2*y1
	dy2dt = -2*x1

	fdx1dt = lambda1*dx1dt + (1-lambda1)*dx2dt
	fdy1dt = lambda1*dy1dt + (1-lambda1)*dy2dt

	rhs = [fdx1dt, fdy1dt]

	# print rhs

	return rhs


def smoothHybridLinearOscillator(state, time):
	# smoothen the dynamics by performing a linear interpolation.
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]
	scale = 1.0

	g1 = 1.0/(1.0 + np.exp(scale*x1))
	g2 = 1.0/(1.0 + np.exp(scale*-1*x1))

	lambda1 = g1/(g1+g2)

	dx1dt = 0
	dy1dt = 0
	dx2dt = 0
	dy2dt = 0

	if x1 <= 0 :
		dx1dt = y1
		dy1dt = -1*x1
	if x1 > 0 :
		dx2dt = 2*y1
		dy2dt = -2*x1

	fdx1dt = lambda1*dx1dt + (1-lambda1)*dx2dt
	fdy1dt = lambda1*dy1dt + (1-lambda1)*dy2dt

	rhs = [fdx1dt, fdy1dt]

	# print rhs

	return rhs


def regularOscillator(state, time):

	x1 = state[0]
	y1 = state[1]

	dx1dt = y1
	dy1dt = -1*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def biologicalModel_1(state, time):

	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]
	x5 = state[4]
	x6 = state[5]
	x7 = state[6]

	dx1dt = -0.4*x1 + 5*x3*x4
	dx2dt = 0.4*x1 - x2
	dx3dt = x2 - 5*x3*x4
	dx4dt = 5*x5*x6 - 5*x3*x4
	dx5dt = -5*x5*x6 + 5*x3*x4
	dx6dt = 0.5*x7 - 5*x5*x6
	dx7dt = -0.5*x7 + 5*x5*x6

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt]

	return rhs


def laubLoomis(state, time):

	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]
	x5 = state[4]
	x6 = state[5]
	x7 = state[6]
	# x8 = state[7]

	dx1dt = 1.4*x3 - 0.9*x1
	dx2dt = 2.5*x5 - 1.5*x2
	dx3dt = 0.6*x7 - 0.8*x2*x3
	dx4dt = 2.0 - 1.3*x3*x4
	dx5dt = 0.7*x1 - x4*x5
	dx6dt = 0.3*x1 - 3.1*x6
	dx7dt = 1.8*x6 - 1.5*x2*x7
	# dx8dt = 1

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt]

	return rhs


'''https://github.com/schillic/HA2Stateflow'''


def rober(state, time):
	k1 = 0.04
	k2 = 30000000
	k3 = 10000

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = -k1*x + k3*y*z
	dydt = k1*x - k2*y*y - k3*y*z
	dzdt = k2*y*y

	rhs = [dxdt, dydt, dzdt]
	return rhs


def orego(state, time):
	s = 77.27
	w = 0.161
	q = 0.008375

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = s*(y-x*y+x-q*x*x)
	dydt = (1/s)*(-y-x*y+z)
	dzdt = w*(y-z)

	rhs = [dxdt, dydt, dzdt]
	return rhs


def e5(state, time):
	A = 0.00000000789
	B = 11000000
	C = 1130
	M = 1000000

	x = state[0]
	y = state[1]
	z = state[2]
	w = state[3]

	dxdt = -A*x - B*x*z
	dydt = A*x - M*C*y*z
	dzdt = A*x - B*x*z - M*C*y*z + C*w
	dwdt = B*x*z - C*w

	rhs = [dxdt, dydt, dzdt, dwdt]
	return rhs


def sa_nonlinear(state, time):
	pi = 22/7
	x = state[0]
	y = state[1]

	dxdt = x - y + 0.1*time
	dydt = y * np.cos(2 * pi * y) - x * np.sin(2 * pi * x) + 0.1*time
	rhs = [dxdt, dydt]
	return rhs


def dampedOscillator(state, time):

	x = state[0]
	y = state[1]

	dxdt = -0.1 * x + y
	dydt = -1 * x - 0.1 * y

	rhs = [dxdt, dydt]

	return rhs


def oscParticle(state, time):

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = -0.1 * x - y
	dydt = x - 0.1 * y
	dzdt = -0.15 * z

	rhs = [dxdt, dydt, dzdt]

	return rhs


def adaptiveCruise(state, time):

	s = state[0]
	v = state[1]
	a = state[2]
	vf = 20

	dsdt = -1 * v + vf
	dvdt = a
	dadt = s - 4 * v + 3 * vf - 3 * a - 10

	rhs = [dsdt, dvdt, dadt]

	return rhs
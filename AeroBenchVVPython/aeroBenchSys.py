import numpy as np
from numpy import deg2rad

from RunF16Sim import RunF16Sim
from PassFailAutomaton import AirspeedPFA, FlightLimits, FlightLimitsPFA
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController
from Autopilot import FixedSpeedAutopilot
from controlledF16 import controlledF16
from Autopilot import FixedAltitudeAutopilot
from plot import plot2d


class AeroBenchSim(object):
    def __init__(self, dimensions, timeStep, steps, f16_plant='morelli'):
        self.p_gain = 0.01
        if dimensions == 2:
            self.diff_level = 'easy'
        elif dimensions == 5:
            self.diff_level = 'medium'
        self.f16_plant = f16_plant  # 'stevens' or 'morelli'
        self.tMax = int(steps * timeStep)  # simulation time
        self.sim_step = timeStep

    def getSimulationsEasy(self, states):
        setpoint = 1220
        ctrlLimits = CtrlLimits()
        flightLimits = FlightLimits()
        llc = LowLevelController(ctrlLimits)

        ap = FixedSpeedAutopilot(setpoint, self.p_gain, llc.xequil, llc.uequil, flightLimits, ctrlLimits)

        def der_func(t, y):
            'derivative function'

            der = controlledF16(t, y, self.f16_plant, ap, llc)[0]

            rv = np.zeros((y.shape[0],))

            rv[0] = der[0]  # speed
            rv[11] = der[11]  # alt
            rv[12] = der[12]  # power lag term

            return rv

        pass_fail = AirspeedPFA(60, setpoint, 5)

        power = 0  # Power

        # Default alpha & beta
        alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
        beta = 0  # Side slip angle (rad)

        # alt = 20000  # Initial Attitude
        # Vt = 1000  # Initial Speed
        phi = 0  # (pi/2)*0.5           # Roll angle from wings level (rad)
        theta = 0  # (-pi/2)*0.8        # Pitch angle from nose level (rad)
        psi = 0  # -pi/4                # Yaw angle from North (rad)

        trajectories = []
        for state in states:
            # Build Initial Condition Vectors
            # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
            alt = state[1]
            Vt = state[0]
            # power = state[2]
            initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

            passed, times, states, modes, ps_list, Nz_list, u_list = \
                RunF16Sim(initialState, self.tMax, der_func, self.f16_plant, ap, llc, pass_fail, sim_step=self.sim_step)

            # print("Simulation Conditions Passed: {}".format(passed))

            statesArr = np.array(states)
            traj = np.take(statesArr, [0, 11], axis=1)
            trajectories += [traj]
            # filename = None  # engine_e.png
            # plot2d(filename, times, [(states, [(0, 'Vt'), (12, 'Pow')]), (u_list, [(0, 'Throttle')])])

        return trajectories

    def getSimulationsMedium(self, states):

        # Initial Conditions ###
        power = 0  # Power

        # Default alpha & beta
        # alpha = 0  # angle of attack (rad)
        beta = 0  # Side slip angle (rad)

        # alt = 500  # Initial Attitude
        # Vt = 540  # Initial Speed
        phi = 0
        # theta = alpha
        psi = 0

        trajectories = []
        for state in states:
            Vt = state[0]
            alt = state[1]
            alpha = state[2]
            theta = alpha
            pitchRate = state[3]

            if alt > 550:
                setpoint = 500
            else:
                setpoint = 550
            ctrlLimits = CtrlLimits()
            flightLimits = FlightLimits()
            llc = LowLevelController(ctrlLimits)

            ap = FixedAltitudeAutopilot(setpoint, llc.xequil, llc.uequil, flightLimits, ctrlLimits)

            def der_func(t, y):
                'derivative function for RK45'

                der = controlledF16(t, y, self.f16_plant, ap, llc)[0]

                rv = np.zeros((y.shape[0],))

                rv[0] = der[0]  # air speed (Vt)
                rv[1] = der[1]  # alpha
                rv[4] = der[4]  # pitch angle
                rv[7] = der[7]  # pitch rate
                rv[11] = der[11]  # altitude (alt)
                rv[12] = der[12]  # power lag term
                rv[13] = der[13]  # Nz integrator

                return rv

            pass_fail = FlightLimitsPFA(flightLimits)
            pass_fail.break_on_error = False

            # Build Initial Condition Vectors
            # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]

            initialState = [Vt, alpha, beta, phi, theta, psi, 0, pitchRate, 0, 0, 0, alt, power]

            print(initialState)
            passed, times, states, modes, ps_list, Nz_list, u_list = \
                RunF16Sim(initialState, self.tMax, der_func, self.f16_plant, ap, llc, pass_fail, sim_step=self.sim_step)

            print("Simulation Conditions Passed: {}".format(passed))

            statesArr = np.array(states)
            traj = np.take(statesArr, [0, 1, 4, 7, 11, 13], axis=1)
            # print(traj[0])
            # print(traj.T[5].shape)
            traj.T[5] = np.array(Nz_list)
            trajectories += [traj]

            # filename = None  # longitudinal.png
            # plot2d(filename, times, [
            #     (states, [(0, 'Vt'), (11, 'Altitude')]), (u_list, [(0, 'Throttle'), (1, 'elevator')]),
            #     (Nz_list, [(0, 'Nz')])])

        return trajectories

    def getSimulations(self, states):

        if self.diff_level is "easy":
            return self.getSimulationsEasy(states)
        elif self.diff_level == "medium":
            return self.getSimulationsMedium(states)

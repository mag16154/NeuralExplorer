import numpy as np
import yaml
import matplotlib.pyplot as plt

# For SatReLU
#  https://pythonhosted.org/neurolab/_modules/neurolab/trans.html


class DnnController(object):
    def __init__(self, dnn_fname, dimensions):
        self.dnn_file = dnn_fname
        self.weights = []
        self.activations = []
        self.offsets = []
        self.input_dimensions = dimensions
        self.n_layers = 2

    def parseDNNYML(self, dynamics):
        # print("Parsing DNN YML file.")
        yaml_f = open(self.dnn_file, 'r')
        yml_obj = yaml.load(yaml_f)
        acts = yml_obj.get('activations')
        offsets = yml_obj.get('offsets')
        weights = yml_obj.get('weights')

        if dynamics == 'ABS':
            for val in acts:
                self.activations.append(val)
            for val in offsets:
                self.offsets.append(np.array(val))
            for val in weights:
                self.weights.append(np.array(val))
            self.n_layers = len(self.activations)
            self.weights[self.n_layers - 1] = np.array([self.weights[self.n_layers - 1]])
        else:
            for item, val in acts.items():
                self.activations.append(val)
            offsets = yml_obj.get('offsets')
            for item, val in offsets.items():
                self.offsets.append(np.array(val))
            weights = yml_obj.get('weights')
            for item, val in weights.items():
                self.weights.append(np.array(val))
            self.n_layers = len(self.activations)

    def performForwardPass(self, input):
        assert self.input_dimensions == len(input)
        prev_layer_output = np.array(input)
        for idx in range(self.n_layers):
            # print(self.weights[idx].shape, prev_layer_output.shape)
            wx = np.matmul(self.weights[idx], prev_layer_output)
            # print(wx.shape)
            # print(self.offsets[idx].shape)
            # print(wx)
            wx_b = wx + self.offsets[idx]
            # print(wx_b.shape)
            # print(-wx_b)
            if self.activations[idx] == 'Sigmoid':
                current_layer_output = 1/(1 + np.exp(-wx_b))
            elif self.activations[idx] == 'Tanh':
                current_layer_output = np.tanh(wx_b)
            elif self.activations[idx] == 'ReLU':
                current_layer_output = np.maximum(wx_b, 0)
            elif self.activations[idx] == 'SatReLU':
                current_layer_output = np.copy(wx_b)
                current_layer_output[np.where(current_layer_output <= 0)] = 0
                current_layer_output[np.where(current_layer_output >= 1)] = 1
                print("In SatReLU")
                print(wx_b, current_layer_output)
            else:
                current_layer_output = wx_b
                # print("Linear")
            prev_layer_output = current_layer_output
        return prev_layer_output


class Plant(object):
    def __init__(self, model, dnn_control=None, dnn_transform=None, steps=1000):
        self.model = model
        self.steps = steps
        self.dnn_controller = dnn_control
        self.dnn_transform = dnn_transform

    def simulate_MC(self, init_state):
        traj = []
        traj.append(np.array(init_state))
        prev_state = init_state
        controller_output = self.dnn_controller.performForwardPass(prev_state)
        # print(controller_output[-1])
        prev_pos = prev_state[0]
        prev_vel = prev_state[1]
        # print(prev_pos, prev_vel)
        for step in range(1, self.steps):
            next_pos = prev_pos + prev_vel
            next_vel = prev_vel + 0.0015*controller_output[-1] - 0.0025*np.cos(3*prev_pos)
            next_state = [next_pos, next_vel]
            controller_output = self.dnn_controller.performForwardPass(next_state)
            # print(controller_output[-1])
            traj.append(next_state)
            prev_pos = next_pos
            prev_vel = next_vel
        return traj

    def getControlInputs_quad(self, controller_output):
        if controller_output[7] <= 0.0:
            u1 = 0.1
            u2 = 0.1
            u3 = 11.81
            # print("Control action 1")
        elif controller_output[6] <= 0.0 and controller_output[4] <= 0.0:
            u1 = 0.1
            u2 = 0.1
            u3 = 7.81
            # print("Control action 2")
        elif controller_output[5] <= 0.0 and controller_output[3] <= 0.0:
            u1 = 0.1
            u2 = -0.1
            u3 = 11.81
            # print("Control action 3")
        elif controller_output[4] <= 0.0 and controller_output[2] <= 0.0:
            u1 = 0.1
            u2 = -0.1
            u3 = 7.81
            # print("Control action 4")
        elif controller_output[3] <= 0.0 and controller_output[1] <= 0.0:
            u1 = -0.1
            u2 = 0.1
            u3 = 11.81
            # print("Control action 5")
        elif controller_output[2] <= 0.0 and controller_output[0] <= 0.0:
            u1 = -0.1
            u2 = 0.1
            u3 = 7.81
            # print("Control action 6")
        elif controller_output[1] <= 0.0:
            u1 = -0.1
            u2 = -0.1
            u3 = 11.81
            # print("Control action 7")
        else:
            u1 = -0.1
            u2 = -0.1
            u3 = 7.81
            # print("Control action 8")

        return u1, u2, u3

    def simulate_quad(self, init_state):
        traj = []
        traj.append(np.array(init_state))
        prev_state = init_state
        controller_output = self.dnn_controller.performForwardPass(prev_state)
        prev_pos_x = prev_state[0]
        prev_pos_y = prev_state[1]
        prev_pos_z = prev_state[2]
        prev_vel_x = prev_state[3]
        prev_vel_y = prev_state[4]
        prev_vel_z = prev_state[5]
        t = 0
        step_size = 0.01
        mode = 1

        for step in range(1, self.steps):
            if mode == 1:
                # print("Mode 1 {}".format(t))
                const1 = -0.5
                const2 = 0.5
                # if prev_pos_x <= 0.0 or prev_pos_y <= 0.0:
                if prev_pos_x <= -0.05:
                    # print("Switching to mode 2")
                    mode = 2
            elif mode == 2:
                # print("Mode 2 {}".format(t))
                const1 = -0.5
                const2 = -0.5
                # if prev_pos_y <= 0.02:
                if prev_pos_y <= -0.1:
                    # print("Switching to mode 3")
                    mode = 3
            elif mode == 3:
                # print("Mode 3 {}".format(t))
                const1 = 0.5
                const2 = -0.5
                # if prev_pos_y <= 0.0:
                if prev_pos_x >= 0.1:
                    # print("Switching to mode 4")
                    mode = 4
            elif mode == 4:
                # print("Mode 4 {}".format(t))
                const1 = 0.5
                const2 = 0.5
                if prev_pos_x >= 0.4:
                    # print("Switching to mode 5")
                    mode = 5
            elif mode == 5:
                const1 = 0.1
                const2 = -0.1
            # print(controller_output)
            u1, u2, u3 = self.getControlInputs_quad(controller_output)
            # print(u1, u2, u3)
            next_pos_x = prev_pos_x + (prev_vel_x + const1)*step_size
            next_pos_y = prev_pos_y + (prev_vel_y + const2)*step_size
            next_pos_z = prev_pos_z + prev_vel_z * step_size
            next_vel_x = prev_vel_x + (9.81 * np.tan(u1))*step_size
            next_vel_y = prev_vel_y - (9.81 * np.tan(u2))*step_size
            next_vel_z = prev_vel_z + (u3 - 9.81)*step_size

            next_state = [next_pos_x, next_pos_y, next_pos_z, next_vel_x, next_vel_y, next_vel_z]
            controller_output = self.dnn_controller.performForwardPass(next_state)
            # print(controller_output[-1])
            traj.append(next_state)
            prev_pos_x = next_pos_x
            prev_pos_y = next_pos_y
            prev_pos_z = next_pos_z
            prev_vel_x = next_vel_x
            prev_vel_y = next_vel_y
            prev_vel_z = next_vel_z
            t = t+step_size

        return traj

    def computeTTC(self, state):
        d_k = state[0]
        v_k = state[1]
        a_k = state[2]

        ttc_val = 0
        if a_k == 0:
            ttc_val = v_k/d_k
        else:
            temp = v_k*v_k + 2*a_k*d_k
            if temp >= 0:
                ttc_val = (-a_k)/(v_k - np.sqrt(temp))
            else:
                ttc_val = 0
        return ttc_val

    def simulate_abs(self, init_state):
        traj = []
        ttc = []
        acc = []
        brakes = []
        traj.append(np.array(init_state))
        prev_state = init_state

        norm_mat = np.array([[0.004, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.05]])
        vel_scale = 0.03
        output_scale = 500
        # prev_state = np.array([1.0, 1.0, 1.0])
        prev_pos = prev_state[0]
        prev_vel = prev_state[1]
        prev_acc = prev_state[2]
        controller_output = np.array([prev_acc])
        # acc.append(prev_acc)
        # ttc.append(self.computeTTC(init_state))
        # brake = self.dnn_controller.performForwardPass(np.matmul(norm_mat, np.array(prev_state)))
        # tf_input = [brake[-1], vel_scale*prev_vel]
        # controller_output = self.dnn_transform.performForwardPass(tf_input)
        # controller_output[-1] = controller_output[-1]*output_scale
        dt = 1/15
        # print(prev_pos, prev_vel)
        for step in range(0, self.steps):
            next_pos = (prev_pos - dt*prev_vel)
            next_vel = (prev_vel + dt * controller_output[-1])
            next_acc = (controller_output[-1])
            next_state = [next_pos, next_vel, next_acc]

            brake = self.dnn_controller.performForwardPass(np.dot(norm_mat, np.array(next_state)))
            tf_input = [brake[-1], vel_scale*next_vel]
            controller_output = self.dnn_transform.performForwardPass(tf_input)
            controller_output[-1] = controller_output[-1]*output_scale
            # print(next_state, init_state)
            # traj.append(np.multiply(next_state, init_state))
            traj.append(next_state)
            ttc.append(self.computeTTC(next_state))
            prev_pos = next_pos
            prev_vel = next_vel
            prev_acc = next_acc
            acc.append(prev_acc)
            brakes.append(brake)

        print(ttc)
        plt.plot(ttc)
        plt.show()
        plt.plot(acc)
        plt.show()
        plt.plot(brakes)
        plt.show()
        return traj

    def simulate_vehicle_platoon(self, init_state):
        traj = []
        traj.append(np.array(init_state))
        prev_state = init_state
        t = 0
        dt = 0.02
        mode = 1
        prev_e1 = prev_state[0]
        prev_e1p = prev_state[1]
        prev_a1 = prev_state[2]
        prev_e2 = prev_state[3]
        prev_e2p = prev_state[4]
        prev_a2 = prev_state[5]
        prev_e3 = prev_state[6]
        prev_e3p = prev_state[7]
        prev_a3 = prev_state[8]

        for step in range(1, self.steps):
            next_state = []
            if mode == 1:
                next_e1 = prev_e1 + prev_e1p*dt
                next_e1p = prev_e1p - prev_a1*dt
                next_a1 = prev_a1 + (
                        1.605*prev_e1 + 4.868*prev_e1p - 3.5754*prev_a1 - 0.8198*prev_e2 + 0.427*prev_e2p - \
                        0.045*prev_a2 - 0.1942*prev_e3 + 0.3626*prev_e3p - 0.0946*prev_a3)*dt
                next_e2 = prev_e2 + prev_e2p*dt
                next_e2p = prev_e2p + (prev_a1 - prev_a2)*dt
                next_a2 = prev_a2 + (
                        0.8718*prev_e1 + 3.814*prev_e1p - 0.0754*prev_a1 + 1.1936*prev_e2 + 3.6258*prev_e2p - \
                        3.2396*prev_a2 - 0.595*prev_e3 + 0.1294*prev_e3p - 0.0796*prev_a3)*dt
                next_e3 = prev_e3 + prev_e3p*dt
                next_e3p = prev_e3p + (prev_a2 - prev_a3)*dt
                next_a3 = prev_a3 + (
                        0.7132*prev_e1 + 3.573*prev_e1p - 0.0964*prev_a1 + 0.8472*prev_e2 + 3.2568*prev_e2p - \
                        0.0876*prev_a2 + 1.2726*prev_e3 + 3.072*prev_e3p - 3.1356*prev_a3)*dt
                t = t+dt

                next_state = [next_e1, next_e1p, next_a1, next_e2, next_e2p, next_a2, next_e3, next_e3p, next_a3]
                traj.append(next_state)

                if t >= 2:
                    t = 0
                    mode = 2
                    # print("Switching to mode 2")

            elif mode == 2:
                next_e1 = prev_e1 + prev_e1p * dt
                next_e1p = prev_e1p - prev_a1 * dt
                next_a1 = prev_a1 + (1.605 * prev_e1 + 4.868 * prev_e1p - 3.5754 * prev_a1) * dt
                next_e2 = prev_e2 + prev_e2p * dt
                next_e2p = prev_e2p + (prev_a1 - prev_a2) * dt
                next_a2 = prev_a2 + (1.1936 * prev_e2 + 3.6258 * prev_e2p - 3.2396 * prev_a2) * dt
                next_e3 = prev_e3 + prev_e3p * dt
                next_e3p = prev_e3p + (prev_a2 - prev_a3) * dt
                next_a3 = prev_a3 + (
                        0.7132 * prev_e1 + 3.573 * prev_e1p - 0.0964 * prev_a1 + 0.8472 * prev_e2 + 3.2568 * prev_e2p \
                        - 0.0876 * prev_a2 + 1.2726 * prev_e3 + 3.072 * prev_e3p - 3.1356 * prev_a3) * dt
                t = t + dt

                next_state = [next_e1, next_e1p, next_a1, next_e2, next_e2p, next_a2, next_e3, next_e3p, next_a3]
                traj.append(next_state)

                if t >= 2:
                    t = 0
                    mode = 1
                    # print("Switching to mode 1")

            prev_e1 = next_state[0]
            prev_e1p = next_state[1]
            prev_a1 = next_state[2]
            prev_e2 = next_state[3]
            prev_e2p = next_state[4]
            prev_a2 = next_state[5]
            prev_e3 = next_state[6]
            prev_e3p = next_state[7]
            prev_a3 = next_state[8]

        return traj

    def simulate_spiking_neuron(self, init_state):
        traj = []
        traj.append(np.array(init_state))
        prev_state = init_state
        t = 0
        dt = 0.02
        a = 0.02
        b = 0.2
        c = -65
        d = 8
        I = 40
        prev_v = prev_state[0]
        prev_u = prev_state[1]
        for step in range(1, self.steps):
            next_v = prev_v + (0.04*prev_v*prev_v + 5*prev_v + 140 - prev_u + I)*dt
            next_u = prev_u + a*(b*prev_v - prev_u)*dt
            t = t + dt
            next_state = [next_v, next_u]
            traj.append(next_state)
            if next_v >= 30:
                prev_v = c
                prev_u = next_u + d
                # print("Switching mode")
            else:
                prev_v = next_state[0]
                prev_u = next_state[1]

        # plt.figure(1)
        # temp = np.array(traj)
        # plt.plot(temp[:, 0])
        # plt.show()
        # plt.figure(1)
        # plt.plot(temp[:, 1])
        # plt.show()
        return traj

    def getSimulations(self, states):
        if self.dnn_controller is not None:
            self.dnn_controller.parseDNNYML(self.model)
        if self.model is 'ABS':
            self.dnn_transform.parseDNNYML(self.model)
        # print("Generating Trajectories")
        trajectories = []

        if self.model is 'MC':
            for state in states:
                state[1] = 0.0
                # print("State {}".format(state))
                traj = self.simulate_MC(state)
                trajectories += [np.array(traj)]
            return trajectories
        elif self.model is 'Quadrotor':
            for state in states:
                state[2] = 0.0
                state[3] = 0.0
                state[4] = 0.0
                state[5] = 0.0
                # print("State {}".format(state))
                traj = self.simulate_quad(state)
                trajectories += [np.array(traj)]
            return trajectories
        elif self.model is 'ABS':
            print(self.steps)
            for state in states:
                # state[2] = 0.0
                print("State {}".format(state))
                traj = self.simulate_abs(state)
                trajectories += [np.array(traj)]
            return trajectories
        elif self.model is 'VehiclePlatoon':
            # print(self.steps)
            for state in states:
                # print("State {}".format(state))
                traj = self.simulate_vehicle_platoon(state)
                trajectories += [np.array(traj)]
            return trajectories
        elif self.model is 'SpikingNeuron':
            # print(self.steps)
            for state in states:
                # print("State {}".format(state))
                traj = self.simulate_spiking_neuron(state)
                trajectories += [np.array(traj)]
            return trajectories

        # print(trajectories)

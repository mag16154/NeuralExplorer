# import sys
# sys.path.append('../configuration-setup/')

from keras.models import load_model
import numpy as np
from frechet import norm
from learningModule import DataConfiguration
from circleRandom import generate_points_in_circle
from mpl_toolkits import mplot3d
import random as rand
import os.path
from os import path
import time
from sampler import generateRandomStates
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


class Evaluation(object):
    def __init__(self, dynamics):
        self.data_object = None
        self.debug_print = False
        self.dynamics = dynamics
        self.iter_count = 10
        self.n_states = 5
        self.usafelowerBoundArray = []
        self.usafeupperBoundArray = []
        self.staliro_run = False
        self.eval_dir = '../../eval/'

    def getDataObject(self):
        assert self.data_object is not None
        return self.data_object

    def setDataObject(self, d_obj_f_name=None):

        if d_obj_f_name is None:
            d_obj_f_name = self.eval_dir + 'dconfigs/d_object_'+self.dynamics+'.txt'

        if path.exists(d_obj_f_name):
            d_obj_f = open(d_obj_f_name, 'r')
            lines = d_obj_f.readlines()
            line_idx = 0
            dimensions = int(lines[line_idx])
            line_idx += 1
            steps = int(lines[line_idx][:-1])
            line_idx += 1
            samples = int(lines[line_idx][:-1])
            line_idx += 1
            timeStep = float(lines[line_idx][:-1])
            line_idx += 1
            lowerBoundArray = []
            for idx in range(dimensions):
                token = lines[line_idx][:-1]
                line_idx += 1
                lowerBoundArray.append(float(token))
            upperBoundArray = []
            for idx in range(dimensions):
                token = lines[line_idx][:-1]
                line_idx += 1
                upperBoundArray.append(float(token))

            d_obj_f.close()

            self.data_object = DataConfiguration(dynamics=self.dynamics, dimensions=dimensions)
            self.data_object.setSteps(steps)
            self.data_object.setSamples(samples)
            self.data_object.setTimeStep(timeStep)
            self.data_object.setLowerBound(lowerBoundArray)
            self.data_object.setUpperBound(upperBoundArray)

            # print(steps, self.data_object.steps)
            # print(dimensions, self.data_object.dimensions)
            # print(samples, self.data_object.samples)
            # print(timeStep, self.data_object.timeStep)
            # print(lowerBoundArray, self.data_object.lowerBoundArray)
            # print(upperBoundArray, self.data_object.upperBoundArray)

        return self.data_object

    def setUnsafeSet(self, lowerBound, upperBound):
        self.usafelowerBoundArray = lowerBound
        self.usafeupperBoundArray = upperBound
        self.staliro_run = True

    def setIterCount(self, iter_count):
        self.iter_count = iter_count

    def setNStates(self, n_states):
        self.n_states = n_states

    def generateRandomUnsafeStates(self, samples):
        states = generateRandomStates(samples, self.usafelowerBoundArray, self.usafeupperBoundArray)
        return states

    def check_for_bounds(self, state):
        # print("Checking for bounds for the state {}".format(state))
        for dim in range(self.data_object.dimensions):
            l_bound = self.data_object.lowerBoundArray[dim]
            u_bound = self.data_object.upperBoundArray[dim]
            if state[dim] < l_bound:
                # print("******* Updated {} to {}".format(state[dim], l_bound + 0.000001))
                state[dim] = l_bound + 0.000001
            elif state[dim] > u_bound:
                # print("******* Updated {} to {}".format(state[dim], u_bound - 0.000001))
                # x_val[dim] = 2 * u_bound - x_val[dim]
                state[dim] = u_bound - 0.000001
        return state

    def evalModel(self, input=None, eval_var='v', model=None):
        output = None
        if eval_var is 'vp':
            x_v_t_pair = list(input[0])
            x_v_t_pair = x_v_t_pair + list(input[1])
            x_v_t_pair = x_v_t_pair + list(input[2])
            x_v_t_pair = x_v_t_pair + [input[4]]
            x_v_t_pair = np.asarray([x_v_t_pair], dtype=np.float64)
            predicted_vp = model.predict(x_v_t_pair)
            predicted_vp = predicted_vp.flatten()
            output = predicted_vp
            # print(predicted_vp)

        elif eval_var is 'v':
            xp_vp_t_pair = list(input[0])
            xp_vp_t_pair = xp_vp_t_pair + list(input[1])
            xp_vp_t_pair = xp_vp_t_pair + list(input[3])
            xp_vp_t_pair = xp_vp_t_pair + [input[4]]
            xp_vp_t_pair = np.asarray([xp_vp_t_pair], dtype=np.float64)
            predicted_v = model.predict(xp_vp_t_pair)
            predicted_v = predicted_v.flatten()
            output = predicted_v
            # print(predicted_v)

        return output

    def reachDest_fwd(self, src=None, dests=None, time_step=None, threshold=None):
        start = time.time()
        model_f_name = self.eval_dir + '/models/model_v_2_vp_'
        model_f_name = model_f_name + self.data_object.dynamics
        model_f_name = model_f_name + '.h5'
        if path.exists(model_f_name):
            model_vp = load_model(model_f_name, compile=False)
        else:
            end = time.time()
            print("Time taken: {}".format(end - start))
            print("Model file does not exists for the benchmark {}".format(self.data_object.dynamics))
            return

        random_i_states = []
        predicted_dists = []
        actual_dists = []
        error_profiles = []
        projected_vps = []
        actual_vps = []

        for dest in dests:
            error_profile = []
            curr_ref_traj = self.data_object.generateTrajectories(samples=1)[0]

            if time_step is None:
                d_time_step = 0
                min_dist = norm(curr_ref_traj[0] - dest, 2)
                for idx in range(1, len(curr_ref_traj)):
                    curr_dist = norm(curr_ref_traj[idx] - dest, 2)
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        d_time_step = idx
            else:
                min_dist = norm(curr_ref_traj[time_step] - dest, 2)
                d_time_step = time_step
                for idx in range(time_step-10, time_step+10, 1):
                    curr_dist = norm(curr_ref_traj[idx] - dest, 2)
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        d_time_step = idx

            ref_state = curr_ref_traj[0]
            print("Time step: {} ref_state {} min_dist {} \n".format(d_time_step, curr_ref_traj[0], min_dist))
            # if min_dist > 1:
            #     scaling_factor = 4*min_dist
            # else:
            #     scaling_factor = 2*min_dist
            random_i_states.append(ref_state)
            actual_dists.append(min_dist)
            predicted_dists.append(min_dist)
            iteration = 1
            while min_dist > threshold and iteration < self.iter_count:
                vecs_in_unit_circle = generate_points_in_circle(n_samples=100, dim=self.data_object.dimensions)

                # print(vecs_in_unit_circle)
                vp_dist = None
                vp_dist_vec = None
                for vec in vecs_in_unit_circle:
                    v_val = np.array(vec)
                    vp_val = v_val
                    x_val = ref_state
                    xp_val = curr_ref_traj[d_time_step]
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, d_time_step)
                    predicted_vp_scaled = self.evalModel(input=data_point, eval_var='vp', model=model_vp)
                    predicted_dest = xp_val + predicted_vp_scaled
                    predicted_dist = norm(dest - predicted_dest, 2)
                    i_state = [self.check_for_bounds(ref_state + vec)]
                    actual_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
                    actual_dist = norm(dest - actual_traj[d_time_step], 2)
                    random_i_states.append(i_state[0])
                    predicted_dists.append(predicted_dist)
                    actual_dists.append(actual_dist)
                    error_profile.append(actual_traj[d_time_step] - predicted_dest)
                    actual_vps.append(actual_traj[d_time_step] - curr_ref_traj[d_time_step])
                    projected_vps.append(predicted_vp_scaled)
                    if vp_dist is None or vp_dist > actual_dist:
                        vp_dist = actual_dist
                        vp_dist_vec = vec
                        # random_i_states.append(i_state[0])
                        # predicted_dists.append(predicted_dist)
                        # actual_dists.append(actual_dist)

                # print(vp_dist)
                new_ref_state = [self.check_for_bounds(ref_state + vp_dist_vec)]
                new_traj = self.data_object.generateTrajectories(r_states=new_ref_state)[0]
                curr_dist = norm(dest - new_traj[d_time_step], 2)
                if min_dist > curr_dist:
                    min_dist = curr_dist
                    ref_state = new_ref_state[0]
                    #  print("Orig Destination {} actual {} ref state {} min_dist {}".format(dest, new_traj[d_time_step], ref_state, min_dist))
                iteration = iteration + 1

            error_profiles.append(error_profile)

        predicted_point_cloud = []
        actual_point_cloud = []
        for idx in range(10):
            predicted_point_list = []
            predicted_point_cloud.append(predicted_point_list)
            actual_point_list = []
            actual_point_cloud.append(actual_point_list)

        for idx in range(len(random_i_states)):
            state = random_i_states[idx]
            pred_dist = predicted_dists[idx]
            if pred_dist >= 1:
                predicted_point_cloud[9].append(state)
            else:
                val_idx = int(np.floor(pred_dist * 10))
                predicted_point_cloud[val_idx].append(state)

            actual_dist = actual_dists[idx]
            if actual_dist >= 1:
                actual_point_cloud[9].append(state)
            else:
                val_idx = int(np.floor(actual_dist * 10))
                actual_point_cloud[val_idx].append(state)

        # for idx in range(len(predicted_point_cloud)):
        #     print(len(predicted_point_cloud[idx]))
        #     print(len(actual_point_cloud[idx]))

        i_x_min = self.data_object.lowerBoundArray[0]
        i_x_max = self.data_object.upperBoundArray[0]
        i_y_min = self.data_object.lowerBoundArray[1]
        i_y_max = self.data_object.upperBoundArray[1]

        i_verts = [
            (i_x_min, i_y_min),  # left, bottom
            (i_x_max, i_y_min),  # left, top
            (i_x_max, i_y_max),  # right, top
            (i_x_min, i_y_max),  # right, bottom
            (i_x_min, i_y_min),  # ignored
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 10)]
        # colors = ['red', 'black', 'blue', 'brown', 'green']

        dist_dict = {
            "0": "0-0.1",
            "1": "0.1-0.2",
            "2": "0.2-0.3",
            "3": "0.3-0.4",
            "4": "0.4-0.5",
            "5": "0.5-0.6",
            "6": "0.6-0.7",
            "7": "0.7-0.8",
            "8": "0.8-0.9",
            "9": ">=0.9"
        }

        fig, ax1 = plt.subplots()

        i_path = Path(i_verts, codes)

        i_patch = patches.PathPatch(i_path, facecolor='none', lw=2)

        ax1.add_patch(i_patch)

        for idx in range(len(predicted_point_cloud)):
            first_point = True
            for point in predicted_point_cloud[idx]:
                if first_point is True:
                    ax1.scatter(point[0], point[1], color=colors[idx], label=dist_dict.get(str(idx)))
                else:
                    ax1.scatter(point[0], point[1], color=colors[idx])
                first_point = False

        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title('Predicted distance profiles')
        plt.legend()
        plt.show()

        fig, ax2 = plt.subplots()

        i_path = Path(i_verts, codes)

        i_patch = patches.PathPatch(i_path, facecolor='none', lw=2)

        ax2.add_patch(i_patch)

        for idx in range(len(actual_point_cloud)):
            first_point = True
            for point in actual_point_cloud[idx]:
                if first_point is True:
                    ax2.scatter(point[0], point[1], color=colors[idx], label=dist_dict.get(str(idx)))
                else:
                    ax2.scatter(point[0], point[1], color=colors[idx])
                first_point = False

        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_title('Actual distance profiles')
        plt.legend()
        plt.show()

        fig, ax3 = plt.subplots()
        for error_profile in error_profiles:
            for idx in range(len(error_profile)):
                point = error_profile[idx]
                ax3.scatter(point[0], point[1])

        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_title('Error distance profiles')
        plt.show()

        fig, ax4 = plt.subplots()
        for point in projected_vps:
            ax4.scatter(point[0], point[1], color='red')
        ax4.set_xlabel('x1')
        ax4.set_ylabel('x2')
        ax4.set_title("Projected displacement")
        plt.show()

        fig, ax5 = plt.subplots()
        for point in actual_vps:
            ax5.scatter(point[0], point[1], color='green')
        ax5.set_xlabel('x1')
        ax5.set_ylabel('x2')
        ax5.set_title("Actual displacement")
        plt.show()

    def point_cloud_fwd_naive(self, src=None, dest=None):
        start = time.time()
        model_f_name = './models/model_v_2_vp_'
        model_f_name = model_f_name + self.data_object.dynamics
        model_f_name = model_f_name + '.h5'
        if path.exists(model_f_name):
            model_vp = load_model(model_f_name, compile=False)
        else:
            end = time.time()
            print("Time taken: {}".format(end - start))
            print("Model file does not exists for the benchmark {}".format(self.data_object.dynamics))
            return

        random_i_states = []
        predicted_dists = []

        for idx in range(1):
            curr_ref_traj = self.data_object.generateTrajectories(samples=1)[0]

            d_time_step = 0
            min_dist = norm(curr_ref_traj[0] - dest, 2)
            for idx in range(1, len(curr_ref_traj)):
                curr_dist = norm(curr_ref_traj[idx] - dest, 2)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    d_time_step = idx
            print("Time step: {}\n".format(d_time_step))

            ref_state = curr_ref_traj[0]
            new_states = generateRandomStates(2000, self.data_object.lowerBoundArray, self.data_object.upperBoundArray)
            for state in new_states:
                v_val = state - ref_state
                vp_val = v_val
                x_val = ref_state
                xp_val = curr_ref_traj[d_time_step]
                time_step = d_time_step
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, time_step)
                predicted_vp_scaled = self.evalModel(input=data_point, eval_var='vp', model=model_vp)
                predicted_dest = xp_val + predicted_vp_scaled
                predicted_dist = norm(dest - predicted_dest, 2)
                random_i_states.append(state)
                predicted_dists.append(predicted_dist)

        point_cloud = []
        for idx in range(10):
            point_list = []
            point_cloud.append(point_list)

        for idx in range(len(random_i_states)):
            state = random_i_states[idx]
            dist = predicted_dists[idx]
            if dist >= 1:
                point_cloud[9].append(state)
            else:
                val_idx = int(np.floor(dist * 10))
                point_cloud[val_idx].append(state)

        for idx in range(len(point_cloud)):
            print(len(point_cloud[idx]))

        i_x_min = self.data_object.lowerBoundArray[0]
        i_x_max = self.data_object.upperBoundArray[0]
        i_y_min = self.data_object.lowerBoundArray[1]
        i_y_max = self.data_object.upperBoundArray[1]

        i_verts = [
            (i_x_min, i_y_min),  # left, bottom
            (i_x_max, i_y_min),  # left, top
            (i_x_max, i_y_max),  # right, top
            (i_x_min, i_y_max),  # right, bottom
            (i_x_min, i_y_min),  # ignored
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 10)]
        # colors = ['red', 'black', 'blue', 'brown', 'green']

        fig, ax = plt.subplots()

        i_path = Path(i_verts, codes)

        i_patch = patches.PathPatch(i_path, facecolor='none', lw=2)

        ax.add_patch(i_patch)

        for idx in range(len(point_cloud)):
            first_point = True
            for point in point_cloud[idx]:
                if first_point is True:
                    ax.scatter(point[0], point[1], color=colors[idx], label=str(idx))
                else:
                    ax.scatter(point[0], point[1], color=colors[idx])
                first_point = False

        plt.legend()
        plt.show()

        end = time.time()
        print("Time taken: {}".format(end - start))

    def compute_reachableSet(self, time_bound=100, plot_point_cloud=False):
        start = time.time()
        model_f_name = self.eval_dir + 'models/model_v_2_vp_'
        model_f_name = model_f_name + self.data_object.dynamics
        model_f_name = model_f_name + '.h5'
        if path.exists(model_f_name):
            model_vp = load_model(model_f_name, compile=False)
        else:
            end = time.time()
            print("Time taken: {}".format(end - start))
            print("Model file does not exists for the benchmark {}".format(self.data_object.dynamics))
            return

        curr_ref_traj = self.data_object.generateTrajectories(samples=1)[0]
        vecs_in_unit_circle = generate_points_in_circle(n_samples=20, dim=self.data_object.dimensions)

        ref_state = curr_ref_traj[0]
        predicted = []
        simulated = []
        projected_vps = []
        actual_vps = []
        for vec in vecs_in_unit_circle:
            projected_vp = []
            actual_vp = []
            v_val = np.array(vec)
            vp_val = v_val
            x_val = ref_state
            i_state = [self.check_for_bounds(x_val + vec)]
            actual_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
            actual_traj = actual_traj[0:time_bound]
            predicted_traj = [ref_state+v_val]
            projected_vp.append(v_val)
            actual_vp.append(actual_traj[0]-curr_ref_traj[0])
            for idx in range(1, time_bound):
                time_step = idx
                xp_val = curr_ref_traj[time_step]
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, time_step)
                predicted_vp_scaled = self.evalModel(input=data_point, eval_var='vp', model=model_vp)
                predicted_dest = xp_val + predicted_vp_scaled
                predicted_traj.append(predicted_dest)
                projected_vp.append(predicted_vp_scaled)
                actual_vp.append(actual_traj[idx]-curr_ref_traj[idx])

            predicted_traj = np.array(predicted_traj)
            predicted.append(predicted_traj)
            simulated.append(actual_traj)
            projected_vps.append(projected_vp)
            actual_vps.append(actual_vp)

        if plot_point_cloud is True:

            i_x_min = self.data_object.lowerBoundArray[0]
            i_x_max = self.data_object.upperBoundArray[0]
            i_y_min = self.data_object.lowerBoundArray[1]
            i_y_max = self.data_object.upperBoundArray[1]

            i_verts = [
                (i_x_min, i_y_min),  # left, bottom
                (i_x_max, i_y_min),  # left, top
                (i_x_max, i_y_max),  # right, top
                (i_x_min, i_y_max),  # right, bottom
                (i_x_min, i_y_min),  # ignored
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]

            fig, ax = plt.subplots()
            i_path = Path(i_verts, codes)

            i_patch = patches.PathPatch(i_path, facecolor='none', lw=1)
            ax.add_patch(i_patch)
            colors = ['red', 'black', 'blue', 'brown', 'green']

            ref_points = [curr_ref_traj[0]]

            for idx in range(15):
                ref_point = generateRandomStates(1, self.data_object.lowerBoundArray, self.data_object.upperBoundArray)
                ref_points.append(ref_point[0])
            # print(ref_points)
            for idx in range(len(ref_points)):
                ref_point = ref_points[idx]
                ax.scatter(ref_point[0], ref_point[1], color='red')
                for vec in vecs_in_unit_circle:
                    neighbor = self.check_for_bounds(ref_point + 0.25*np.array(vec))
                    ax.scatter(neighbor[0], neighbor[1], color='green')

            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title("Sampled initial states")
            plt.show()

        plt.figure(1)
        for idx in range(len(predicted)):
            predicted_traj = predicted[idx]
            plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], color='red')
        plt.title('Predicted trajectories')
        plt.show()

        plt.figure(2)
        for idx in range(len(predicted)):
            simulated_traj = simulated[idx]
            plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], color='green')
        plt.title('Actual trajectories')
        plt.show()

        fig, ax2 = plt.subplots()
        for idx in range(len(projected_vps)):
            projected_vp = projected_vps[idx]
            for point in projected_vp:
                ax2.scatter(point[0], point[1], color='red')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_title("Projected displacement")
        plt.show()

        fig, ax3 = plt.subplots()
        for idx in range(len(actual_vps)):
            actual_vp = actual_vps[idx]
            for point in actual_vp:
                ax3.scatter(point[0], point[1], color='green')
        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_title("Actual displacement")
        plt.show()

    def reachabilityInv(self, time_bound=100):
        start = time.time()
        model_f_name = self.eval_dir + 'models/model_vp_2_v_'
        model_f_name = model_f_name + self.data_object.dynamics
        model_f_name = model_f_name + '.h5'
        if path.exists(model_f_name):
            model_v = load_model(model_f_name, compile=False)
        else:
            end = time.time()
            print("Time taken: {}".format(end - start))
            print("Model file does not exists for the benchmark {}".format(self.data_object.dynamics))
            return

        curr_ref_traj = self.data_object.generateTrajectories(samples=1)[0]
        vecs_in_unit_circle = generate_points_in_circle(n_samples=10, dim=self.data_object.dimensions)
        projected_vps = []
        actual_vps = []
        simulated_trajs = []
        predicted_trajs = []
        error_vecs_magnitude = []

        # scaling_factor = 0.1 For LaubLoomis and Quadrotor 0.01 for MC, 0.1 for Vanderpol, 1.5 for Steam

        if self.data_object.dynamics == 'Steam':
            scaling_factor = 0.6
        elif self.data_object.dynamics == 'LaubLoomis' or self.data_object.dynamics == 'Quadrotor':
            scaling_factor = 0.05
        elif self.data_object.dynamics == 'MC':
            scaling_factor = 0.01
        else:
            scaling_factor = 0.1

        print(scaling_factor)
        ref_state = curr_ref_traj[0]
        for vec in vecs_in_unit_circle:
            predicted = []
            simulated = []
            i_states = []

            vp_val = scaling_factor*np.array(vec)
            x_val = ref_state
            v_val = vp_val
            projected_vp = []
            actual_vp = []
            error_vec_magnitude = []
            for t_step in range(time_bound, 0, -1):
                xp_val = curr_ref_traj[t_step]
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_step)
                predicted_v_scaled = self.evalModel(input=data_point, eval_var='v', model=model_v)
                i_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                i_states.append(i_state)
                predicted.append(xp_val+vp_val)
                actual_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
                simulated.append(actual_traj[t_step])
                actual_vp.append(vp_val)
                projected_vp.append(actual_traj[t_step] - curr_ref_traj[t_step])
                error_vec_magnitude.append(norm(vp_val - (actual_traj[t_step] - curr_ref_traj[t_step]), 2))

            simulated.reverse()
            simulated = np.array(simulated)
            predicted.reverse()
            predicted = np.array(predicted)
            simulated_trajs.append(simulated)
            predicted_trajs.append(predicted)
            projected_vp.reverse()
            projected_vps.append(projected_vp)
            error_vec_magnitude.reverse()
            error_vecs_magnitude.append(error_vec_magnitude)
            actual_vp.reverse()
            actual_vps.append(actual_vp)
            # print(error_vec_magnitude)

        x_index = 0
        y_index = 1

        if self.dynamics == 'MC':
            x_index = 1
            y_index = 0

        plt.figure(1)
        for idx in range(len(predicted_trajs)):
            predicted = predicted_trajs[idx]
            simulated = simulated_trajs[idx]
            if idx == 0:
                plt.plot(simulated[:, x_index], simulated[:, y_index], color='green', label='Actual trajectory')
                plt.plot(predicted[:, x_index], predicted[:, y_index], color='red', label='Derived trajectory')
            else:
                plt.plot(simulated[:, x_index], simulated[:, y_index], color='green')
                plt.plot(predicted[:, x_index], predicted[:, y_index], color='red')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("Actual vs derived trajectories")
        plt.legend()
        plt.show()

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 10)]

        fig, ax1 = plt.subplots()
        for idx in range(4):
            projected_vp = projected_vps[idx]
            for point in projected_vp:
                ax1.scatter(point[x_index], point[y_index], color=colors[2*idx])
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title("Derived displacement")
        plt.show()

        fig, ax2 = plt.subplots()
        for idx in range(4):
            actual_vp = actual_vps[idx]
            for point in actual_vp:
                ax2.scatter(point[x_index], point[y_index], color=colors[2*idx])
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_title("Actual displacement")
        plt.show()

        # plt.figure(1)
        # for idx in range(len(error_vecs_magnitude)):
        #     error_vec_magnitude = error_vecs_magnitude[idx]
        #     plt.plot(error_vec_magnitude)
        # plt.xlabel('time')
        # plt.ylabel('l2-norm')
        # plt.title("l2-norm of the difference in displacements")
        # plt.show()

    def reachDest_wo_f(self, ref_traj=None, dest=None, d_time_step=0, threshold=0.1, model_v=None):

        # print("*** Implementing Second algorithm w/ factorization ***")

        new_traj = ref_traj
        scale = 15 / 16

        dist = norm(dest - new_traj[d_time_step], 2)

        if d_time_step is 0:
            for idx in range(len(new_traj)):
                curr_dist = norm(dest - new_traj[idx], 2)
                if curr_dist < dist:
                    dist = curr_dist
                    d_time_step = idx
            print("Setting the d_time_step to {}".format(d_time_step))

        iter_states = [new_traj[0]]

        if self.debug_print is True:
            print("x {}, xp {}, dist {}, time {}".format(new_traj[0], new_traj[d_time_step], dist, d_time_step))

            plt.figure(1)
            plt.xlabel('x' + str(0))
            plt.ylabel('x' + str(1))
            plt.plot(dest[0], dest[1], 'r*')
            plt.plot(new_traj[d_time_step][0], new_traj[d_time_step][1], 'g*')

            plt.plot(new_traj[:, 0], new_traj[:, 1])

            plt.show()

        x_val = new_traj[0]
        xp_val = new_traj[d_time_step]
        v_val = dest - x_val
        vp_val = dest - xp_val
        t_val = d_time_step
        min_dist = dist
        min_dist_state = x_val

        iteration = 0

        while dist > threshold and iteration < self.iter_count:
            data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
            predicted_v_scaled = self.evalModel(input=data_point, eval_var='v', model=model_v)
            predicted_v_scaled = predicted_v_scaled * scale

            if self.debug_print is True:
                print("predicted v {}, vp {}".format(predicted_v_scaled, vp_val))

            new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
            # if data_object.dynamics == 'MC':
            #     new_init_state[0][1:] = 0.0
            new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
            x_val = new_traj[0]

            # iter_states.append(x_val)

            xp_val = new_traj[d_time_step]
            v_val = dest - x_val
            vp_val = dest - xp_val
            t_val = d_time_step
            dist = norm(dest - xp_val, 2)
            if dist < min_dist:
                min_dist = dist
                min_dist_state = x_val
                iter_states.append(x_val)

            iteration = iteration + 1

            if self.debug_print is True:
                print("x {}, xp {}, dist {}, time{}".format(x_val, xp_val, dist, d_time_step))
                plt.figure(1)
                plt.xlabel('x' + str(0))
                plt.ylabel('x' + str(1))
                plt.plot(dest[0], dest[1], 'r*')
                plt.plot(xp_val[0], xp_val[1], 'g*')
                plt.plot(new_traj[:, 0], new_traj[:, 1])

                plt.show()

        min_dist_state = [self.check_for_bounds(min_dist_state)]
        min_dist_traj = self.data_object.generateTrajectories(r_states=min_dist_state)[0]
        min_dist = norm(min_dist_traj[d_time_step] - dest, 2)
        return iter_states, min_dist_state[0], min_dist

    def reachDest_w_f(self, ref_traj=None, dest=None, d_time_step=0, threshold=0.1, model_v=None, iterative=False):

        print("*** Implementing Second algorithm w/ factorization ***")
        new_traj = ref_traj[0]
        scale = 15 / 16

        dist = norm(new_traj[d_time_step] - dest, 2)

        iteration = 1

        if self.debug_print is True:
            print("x {}, xp {}, dist {}, time {}".format(new_traj[0], new_traj[d_time_step], dist, d_time_step))

            plt.figure(1)
            plt.xlabel('x' + str(0))
            plt.ylabel('x' + str(1))
            plt.plot(dest[0], dest[1], 'r*')
            plt.plot(new_traj[d_time_step][0], new_traj[d_time_step][1], 'g*')

            plt.plot(new_traj[:, 0], new_traj[:, 1])

            plt.show()

        x_val = new_traj[0]
        xp_val = new_traj[d_time_step]
        v_val = dest - x_val
        vp_val = dest - xp_val
        t_val = d_time_step
        # print("original vp {}".format(vp_val))
        init_div_factor = 2
        div_factor = init_div_factor
        vp_val = vp_val / div_factor
        # print("new vp {}".format(vp_val))
        min_dist = dist
        min_dist_state = x_val
        while dist > threshold and iteration < self.iter_count:
            data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
            predicted_v_scaled = self.evalModel(input=data_point, eval_var='v', model=model_v)
            predicted_v_scaled = predicted_v_scaled * scale
            print("predicted v {}, vp {}".format(predicted_v_scaled, vp_val))
            new_init_state = [x_val + predicted_v_scaled]
            new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
            div_factor = div_factor - 1
            x_val = new_traj[0]
            xp_val = new_traj[d_time_step]
            v_val = dest - x_val
            t_val = d_time_step
            curr_dist = norm(dest - xp_val, 2)
            print("x {}, xp {}, dist {}, time{}".format(x_val, xp_val, curr_dist, d_time_step))
            if div_factor == 0:
                div_factor = init_div_factor
                print("Updating div factor")
            vp_val = dest - xp_val
            vp_val = vp_val / div_factor
            dist = curr_dist

            if dist < min_dist:
                min_dist = dist
                min_dist_state = x_val

            iteration = iteration + 1

            if self.debug_print is True:
                plt.figure(1)
                plt.xlabel('x' + str(0))
                plt.ylabel('x' + str(1))
                plt.plot(dest[0], dest[1], 'r*')
                plt.plot(xp_val[0], xp_val[1], 'g*')
                plt.plot(new_traj[:, 0], new_traj[:, 1])

                plt.show()

        return min_dist_state, min_dist

    def reachDestTimeRange(self, src, dests, d_time_steps, threshold, model_v):

        start = d_time_steps[0]
        end = d_time_steps[len(d_time_steps)-1]
        states = []
        time_steps = []
        dists = []
        for dest in dests:
            min_dist = 100.0
            min_dist_state = None
            min_t_step = 0
            for time_step in range(start, end, 1):
                t_step_dist = 100.0
                t_step_state = None
                for idx in range(self.n_states):
                    if src is not None:
                        init_state = []
                        init_state += [src]
                        ref_traj = self.data_object.generateTrajectories(r_states=init_state)[0]
                    else:
                        ref_traj = self.data_object.generateTrajectories(samples=1)[0]
                        # self.data_object.showTrajectories(ref_traj)
                    iter_states, state, dist = self.reachDest_wo_f(ref_traj, dest, time_step, threshold, model_v)
                    if dist < t_step_dist:
                        t_step_dist = dist
                        t_step_state = state
                # print("time step {} dist {}".format(time_step, t_step_dist))
                if t_step_dist < min_dist:
                    min_dist = t_step_dist
                    min_dist_state = t_step_state
                    min_t_step = time_step
            ref_traj = self.data_object.generateTrajectories(r_states=[min_dist_state])[0]
            print("state {} time_step {} dest {} predicted dest {} dist {}".format(min_dist_state, min_t_step, dest,
                                                                                   ref_traj[min_t_step], min_dist))
            states.append(min_dist_state)
            time_steps.append(min_t_step)
            dists.append(min_dist)

        if self.data_object.dimensions == 2 or self.data_object.dimensions == 6:
            plt.figure(1)
            x_index = 0
            y_index = 1
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]
                time_step = time_steps[idx]
                plt.plot(traj[:, x_index], traj[:, y_index])
                plt.plot(traj[time_step][x_index], traj[time_step][y_index], '^', label='pred dest state '+str(idx+1))
                # print("pred init state {}, pred dest state {} distance {}".format(state, traj[time_step],
                #                                                                         dists[idx]))

            for idx in range(len(dests)):
                dest = dests[idx]
                plt.plot(dest[x_index], dest[y_index], '*', label='original dest state '+str(idx+1))

            plt.legend()
            plt.show()

            mt_sim_file_name = self.eval_dir + '/matlab_figs/mt_simulation_' + self.dynamics+'_tr.m'
            x_index = 0
            y_index = 1

            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            mt_sim_file.write('hold on\n')
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]
                mt_sim_file.write('iteration_' + str(idx) + ' = [')
                for point in traj:
                    mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                mt_sim_file.write('];\n')
                time_step = time_steps[idx]
                mt_sim_file.write('predicted_state_' + str(idx) + ' =  [')
                pred_dest = traj[time_step]
                mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                mt_sim_file.write('initial_state_' + str(idx) + ' =  [')
                init_state = traj[0]
                mt_sim_file.write('{},{}];\n'.format(init_state[x_index], init_state[y_index]))

            for idx in range(len(dests)):
                dest = dests[idx]
                mt_sim_file.write('dest_state_{} = [{},{}];\n'.format(idx, dest[x_index], dest[y_index]))
            mt_sim_file.close()

    def reachDestInv(self, src=None, dests=None, d_time_steps=None, threshold=0.1, basic=False, eval_first_step=False,
                     plot_point_cloud=False):

        if d_time_steps is None:
            d_time_steps = [10]
        assert self.data_object is not None

        if basic is True or self.staliro_run is True:
            self.n_states = 1

        if eval_first_step is True:
            self.iter_count = 1

        start = time.time()
        model_f_name = self.eval_dir + 'models/model_vp_2_v_'
        model_f_name = model_f_name + self.data_object.dynamics
        model_f_name = model_f_name + '.h5'
        if path.exists(model_f_name):
            model_v = load_model(model_f_name, compile=False)
        else:
            print("Model file does not exists for the benchmark {}".format(self.data_object.dynamics))
            return

        if len(d_time_steps) > 1:
            self.reachDestTimeRange(src, dests, d_time_steps, threshold, model_v)
            return

        d_time_step = d_time_steps[0]

        ref_states_for_dest = []
        staliro_i_states = []
        staliro_dists = []
        for dest in dests:
            if self.staliro_run is False:
                print("dest {}".format(dest))
            states = []
            dists = []
            avg_rel_dist = 0.0
            avg_dist = 0.0
            avg_rel_dist_1 = 0.0
            avg_dist_1 = 0.0

            for count in range(self.n_states):
                iter_states = []
                state = None
                dist = 10000.0
                ref_traj = None
                for idx in range(1):
                    if src is not None:
                        init_state = []
                        init_state += [src]
                        curr_ref_traj = self.data_object.generateTrajectories(r_states=init_state)[0]
                    else:
                        curr_ref_traj = self.data_object.generateTrajectories(samples=1)[0]
                    # self.data_object.showTrajectories(ref_traj)

                    # print("state: {}", curr_ref_traj[0])
                    curr_iter_states, curr_state, curr_dist = self.reachDest_wo_f(curr_ref_traj, dest, d_time_step,
                                                                                  threshold, model_v)
                    if curr_dist < dist:
                        dist = curr_dist
                        iter_states = curr_iter_states
                        state = curr_state
                        ref_traj = curr_ref_traj

                if basic is True or self.staliro_run is True:
                    for idx in range(len(iter_states)):
                        iter_state = iter_states[idx]
                        states.append(self.check_for_bounds(iter_state))
                        traj = self.data_object.generateTrajectories(r_states=[iter_state])[0]
                        dist = norm(dest - traj[d_time_step], 2)
                        dists.append(dist)

                else:

                    if eval_first_step is False:
                        original_distance = norm(dest - ref_traj[d_time_step], 2)
                        ref_states_for_dest.append(ref_traj[0])
                        avg_rel_dist += (dist/original_distance)
                        avg_dist += dist
                        if len(iter_states) > 1:
                            iter_idx = 1
                        else:
                            iter_idx = 0
                        traj_1 = self.data_object.generateTrajectories(r_states=[iter_states[iter_idx]])[0]
                        dist_1 = norm(dest - traj_1[d_time_step], 2)
                        avg_rel_dist_1 += (dist_1/original_distance)
                        if (dist_1/original_distance) > 1.0:
                            print(iter_idx, dist_1, original_distance, dist)
                        avg_dist_1 += dist_1
                        state = self.check_for_bounds(state)
                        states.append(state)
                        dists.append(dist)
                    else:
                        original_distance = norm(dest - ref_traj[d_time_step], 2)
                        avg_rel_dist += (dist / original_distance)
                        avg_dist += dist
                        dists.append(original_distance)
                        dists.append(dist)
                        states.append(ref_traj[0])
                        states.append(state)

            if basic is True:
                self.plotResultsBasic(states, d_time_step, dest, dists)  # Plotting for n_states = 1
                # self.plotResultsMatlab(states, d_time_step, dest, dists)

            elif avg_rel_dist != 1.0 and self.staliro_run is False:
                avg_rel_dist = (avg_rel_dist / self.n_states)
                avg_dist = (avg_dist / self.n_states)
                avg_rel_dist_1 = (avg_rel_dist_1 / self.n_states)
                avg_dist_1 = (avg_dist_1 / self.n_states)
                print("Avg new dist 1 {} rel dist 1 {} for n_states {}".format(avg_dist_1, avg_rel_dist_1,
                                                                               self.n_states))
                print("Avg new dist {} {} rel dist {} {} for n_states {}".format(self.iter_count, avg_dist,
                                                                                 self.iter_count, avg_rel_dist,
                                                                                 self.n_states))

            if self.staliro_run is True:
                staliro_i_states.append(states)
                staliro_dists.append(dists)

            # plt.figure(1)
            # x_index = 0
            # y_index = 1
            # plt.xlabel('x' + str(x_index))
            # plt.ylabel('x' + str(y_index))
            # test_dests = []
            # for state in ref_states_for_dest:
            #     test_traj = self.data_object.generateTrajectories(r_states=[state])[0]
            #     test_dests.append(test_traj[d_time_step])
            #
            # ref_states_for_dest = np.asarray(ref_states_for_dest)
            # test_dests = np.asarray(test_dests)
            # # print(ref_states_for_dest, ref_states_for_dest.shape)
            # plt.plot(ref_states_for_dest[:, 0], ref_states_for_dest[:, 1], 'g*')
            # # plt.plot(test_dests[1:100, 0], test_dests[1:100, 1], 'bo')
            # plt.plot(dest[x_index], dest[y_index], 'r^')
            # plt.show()

        if self.staliro_run is True:
            best_i_state = None
            best_dist = None
            best_dest = None
            best_ref_state = None
            best_idx = 0
            for idx in range(len(staliro_dists)):
                temp_dists = staliro_dists[idx]
                temp_states = staliro_i_states[idx]
                if best_dist is None or best_dist > temp_dists[len(temp_dists) - 1]:
                    best_dist = temp_dists[len(temp_dists) - 1]
                    best_i_state = temp_states[len(temp_states) - 1]
                    best_dest = dests[idx]
                    best_ref_state = temp_states[0]
                    best_idx = idx

            print(staliro_i_states[best_idx], staliro_dists[best_idx])
            if best_i_state is not None:
                best_traj = self.data_object.generateTrajectories(r_states=[best_i_state])[0]
                best_d_state = best_traj[d_time_step]
                print("Destination {} Ref state {} Pred Init state {} Pred Dest state {} Distance {}".format(best_dest,
                                                                best_ref_state, best_i_state, best_d_state, best_dist))
                if plot_point_cloud is True:
                    self.plotStaliroResults(staliro_i_states, best_i_state, d_time_step, staliro_dists)
                else:
                    self.plotStaliroResults(staliro_i_states, best_i_state, d_time_step, None)

        end = time.time()
        print("Time taken: {}".format(end - start))

    '''' Evaluation using random points in unit circle'''

    def evaluate_4_vp_random(self, d_time_steps=None):

        assert self.data_object is not None
        assert d_time_steps is not None

        model_f_name = './models/model_vp_2_v_'
        model_f_name = model_f_name + self.data_object.dynamics
        model_f_name = model_f_name + '.h5'
        model_v = None
        if path.exists(model_f_name):
            model_v = load_model(model_f_name, compile=False)
        else:
            print("Model file does not exists for the benchmark {}".format(self.data_object.dynamics))
            return

        exponents = np.linspace(0, 1, 16)
        scales = np.power(10, exponents)

        # out_f_name = './outputs/eval_vp_vals_'
        # out_f_name = out_f_name + self.data_object.dynamics
        # out_f_name = out_f_name + ".txt"
        #
        # if path.exists(out_f_name):
        #     os.remove(out_f_name)
        # vals_f = open(out_f_name, "w")

        abs_errors = []
        rel_errors = []
        for time_step in d_time_steps:
            ref_traj = self.data_object.generateTrajectories(samples=1)
            xp_val = ref_traj[0][time_step]
            x_val = ref_traj[0][0]
            points = generate_points_in_circle(n_samples=500, dim=self.data_object.dimensions)
            abs_error_t = []
            rel_error_t = []
            for scale in scales:
                abs_err = 0.0
                rel_err = 0.0
                # vals_f.write("********************************\n")
                for point in points:
                    # print(scale, point)
                    vp_val = scale*np.array(point)
                    v_val = vp_val
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, time_step)
                    predicted_v_scaled = self.evalModel(input=data_point, eval_var='v', model=model_v)
                    new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                    new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
                    predicted_vp = new_traj[time_step]-xp_val
                    vp_difference = vp_val - predicted_vp
                    if norm(vp_difference, 2) > 100:
                        print("vp {} predicted v {} predicted vp {}".format(vp_val, predicted_v_scaled, predicted_vp))
                    abs_err += norm(vp_difference, 2)
                    rel_err += norm(vp_difference, 2)/norm(vp_val, 2)

                    # vals_f.write(str(vp_val))
                    # vals_f.write(" , ")
                    # vals_f.write(str(predicted_vp))
                    # vals_f.write(" ... ")
                    # vals_f.write(str(norm(vp_difference, 2)))
                    # vals_f.write(" , ")
                    # vals_f.write(str(norm(vp_difference, 2)/norm(vp_val, 2)))
                    # vals_f.write("\n")

                abs_err = abs_err/(len(points)-1)
                rel_err = rel_err/(len(points)-1)
                abs_error_t.append([scale, abs_err])
                rel_error_t.append([scale, rel_err])
            abs_errors.append(np.array(abs_error_t))
            rel_errors.append(np.array(rel_error_t))
        print(np.array(abs_errors).shape)
        print(np.array(rel_errors).shape)
        print(scales)
        # print(abs_errors)
        # print(rel_errors)

        # vals_f.close()

        plt.figure(1)
        plt.xlabel('||v||')
        plt.ylabel('absolute error')
        for idx in range(len(d_time_steps)):
            # print(abs_errors[idx].shape)
            plt.plot(abs_errors[idx][:, 0], abs_errors[idx][:, 1], label='t = '+str(d_time_steps[idx]))
        plt.legend()
        plt.show()

        plt.figure(2)
        plt.xlabel('||v||')
        plt.ylabel('relative error')
        for idx in range(len(d_time_steps)):
            plt.plot(rel_errors[idx][:, 0], rel_errors[idx][:, 1], label=str('t = '+str(d_time_steps[idx])))
        plt.legend()
        plt.show()

        return

    def plotStaliroResults(self, staliro_i_states, best_i_state, d_time_step, staliro_dists=None):

        u_x_min = self.usafelowerBoundArray[0]
        u_x_max = self.usafeupperBoundArray[0]
        u_y_min = self.usafelowerBoundArray[1]
        u_y_max = self.usafeupperBoundArray[1]

        u_verts = [
            (u_x_min, u_y_min),  # left, bottom
            (u_x_max, u_y_min),  # left, top
            (u_x_max, u_y_max),  # right, top
            (u_x_min, u_y_max),  # right, bottom
            (u_x_min, u_y_min),  # ignored
        ]

        i_x_min = self.data_object.lowerBoundArray[0]
        i_x_max = self.data_object.upperBoundArray[0]
        i_y_min = self.data_object.lowerBoundArray[1]
        i_y_max = self.data_object.upperBoundArray[1]

        i_verts = [
            (i_x_min, i_y_min),  # left, bottom
            (i_x_max, i_y_min),  # left, top
            (i_x_max, i_y_max),  # right, top
            (i_x_min, i_y_max),  # right, bottom
            (i_x_min, i_y_min),  # ignored
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        u_path = Path(u_verts, codes)

        fig, ax = plt.subplots()

        u_patch = patches.PathPatch(u_path, facecolor='red', lw=2)
        ax.add_patch(u_patch)

        i_path = Path(i_verts, codes)

        i_patch = patches.PathPatch(i_path, facecolor='none', lw=1)
        ax.add_patch(i_patch)

        if staliro_dists is None:
            for i_states in staliro_i_states:
                for state in i_states:
                    ax.scatter(state[0], state[1], color='green')
        else:
            point_cloud = []
            for idx in range(10):
                point_list = []
                point_cloud.append(point_list)

            for idx in range(len(staliro_dists)):
                temp_dists = staliro_dists[idx]
                temp_states = staliro_i_states[idx]
                for idy in range(len(temp_dists)):
                    dist = temp_dists[idy]
                    if dist >= 1:
                        point_cloud[9].append(temp_states[idy])
                    else:
                        val_idx = int(np.floor(dist * 10))
                        point_cloud[val_idx].append(temp_states[idy])

            # for idx in range(len(point_cloud)):
            #     print(len(point_cloud[idx]))

            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, 10)]
            # colors = ['red', 'black', 'blue', 'brown', 'green']

            dist_dict = {
                "0": "0-0.1",
                "1": "0.1-0.2",
                "2": "0.2-0.3",
                "3": "0.3-0.4",
                "4": "0.4-0.5",
                "5": "0.5-0.6",
                "6": "0.6-0.7",
                "7": "0.7-0.8",
                "8": "0.8-0.9",
                "9": ">=0.9"
            }
            for idx in range(len(point_cloud)):
                first_point = True
                for point in point_cloud[idx]:
                    if first_point is True:
                        ax.scatter(point[0], point[1], color=colors[idx], label=dist_dict.get(str(idx)))
                    else:
                        ax.scatter(point[0], point[1], color=colors[idx])
                    first_point = False
            plt.legend()

        traj = self.data_object.generateTrajectories(r_states=[best_i_state])[0]
        traj = traj[0:d_time_step+50]

        ax.plot(traj[:, 0], traj[:, 1])

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        # ax.set_xlim(5, 10)
        # ax.set_ylim(5, 30)
        plt.show()

    def plotResultsMatlab(self, states, d_time_step, dest, dists):
        if self.n_states == 1:
            sim_file_name = self.eval_dir + './matlab_figs/Basic1/mt_simulation_' + self.dynamics
        else:
            sim_file_name = self.eval_dir + './matlab_figs/Basic2/mt_simulation_' + self.dynamics

        if self.data_object.dimensions == 2 or self.data_object.dimensions == 6:
            mt_sim_file_name = sim_file_name + '_basic.m'

            x_index = 0
            y_index = 1

            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            mt_sim_file.write('hold on\n')
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                else:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                    mt_sim_file.write('initial_state_'+str(idx)+' =  [')
                    init_state = traj[0]
                    mt_sim_file.write('{},{}];\n'.format(init_state[x_index], init_state[y_index]))
            mt_sim_file.write('dest_state = [{},{}];\n'.format(dest[x_index], dest[y_index]))
            mt_sim_file.close()
        elif self.data_object.dimensions == 3:
            mt_sim_file_name = sim_file_name + '_basic.m'

            x_index = 0
            y_index = 1
            z_index = 2
            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    mt_sim_file.write('iteration_' + str(idx) + ' = [')
                    for point in traj:
                        mt_sim_file.write('{},{},{};\n'.format(str(point[x_index]), str(point[y_index]), str(point[z_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_' + str(idx) + ' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index], pred_dest[z_index]))
                else:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{},{};\n'.format(str(point[x_index]), str(point[y_index]), str(point[z_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index], pred_dest[z_index]))
                    mt_sim_file.write('initial_state_'+str(idx)+' =  [')
                    init_state = traj[0]
                    mt_sim_file.write('{},{},{}];\n'.format(init_state[x_index], init_state[y_index], init_state[z_index]))

            mt_sim_file.write('dest_state = [{},{},{}];\n'.format(dest[x_index], dest[y_index], dest[z_index]))
            mt_sim_file.close()
        elif self.data_object.dimensions == 4:
            mt_sim_file_name = sim_file_name + '_basic_1.m'
            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            mt_sim_file.write('hold on\n')
            x_index = 0
            y_index = 1
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                else:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                    mt_sim_file.write('initial_state_'+str(idx)+' =  [')
                    init_state = traj[0]
                    mt_sim_file.write('{},{}];\n'.format(init_state[x_index], init_state[y_index]))
            mt_sim_file.write('dest_state = [{},{}];\n'.format(dest[x_index], dest[y_index]))
            mt_sim_file.close()
            x_index = 2
            y_index = 3
            mt_sim_file_name = sim_file_name + '_basic_2.m'
            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            mt_sim_file.write('hold on\n')
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                else:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                    mt_sim_file.write('initial_state_'+str(idx)+' =  [')
                    init_state = traj[0]
                    mt_sim_file.write('{},{}];\n'.format(init_state[x_index], init_state[y_index]))
            mt_sim_file.write('dest_state = [{},{}];\n'.format(dest[x_index], dest[y_index]))
            mt_sim_file.close()
        elif self.data_object.dimensions == 7:

            x_index = 0
            y_index = 1
            z_index = 2
            mt_sim_file_name = sim_file_name + '_basic_1.m'
            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    mt_sim_file.write('iteration_' + str(idx) + ' = [')
                    for point in traj:
                        mt_sim_file.write('{},{},{};\n'.format(str(point[x_index]), str(point[y_index]), str(point[z_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_' + str(idx) + ' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index], pred_dest[z_index]))
                else:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{},{};\n'.format(str(point[x_index]), str(point[y_index]), str(point[z_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index], pred_dest[z_index]))
                    mt_sim_file.write('initial_state_'+str(idx)+' =  [')
                    init_state = traj[0]
                    mt_sim_file.write('{},{},{}];\n'.format(init_state[x_index], init_state[y_index], init_state[z_index]))

            mt_sim_file.write('dest_state = [{},{},{}];\n'.format(dest[x_index], dest[y_index], dest[z_index]))
            mt_sim_file.close()
            x_index = 3
            y_index = 4
            mt_sim_file_name = sim_file_name + '_basic_2.m'
            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            mt_sim_file.write('hold on\n')
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                else:
                    mt_sim_file.write('iteration_' + str(idx) + ' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_' + str(idx) + ' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                    mt_sim_file.write('initial_state_' + str(idx) + ' =  [')
                    init_state = traj[0]
                    mt_sim_file.write('{},{}];\n'.format(init_state[x_index], init_state[y_index]))
            mt_sim_file.write('dest_state = [{},{}];\n'.format(dest[x_index], dest[y_index]))
            mt_sim_file.close()
            x_index = 5
            y_index = 6
            mt_sim_file_name = sim_file_name + '_basic_3.m'
            if path.exists(mt_sim_file_name):
                os.remove(mt_sim_file_name)
            mt_sim_file = open(mt_sim_file_name, 'w')
            mt_sim_file.write('h = figure(1);\n')
            mt_sim_file.write('hold on\n')
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    mt_sim_file.write('iteration_'+str(idx)+' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_'+str(idx)+' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                else:
                    mt_sim_file.write('iteration_' + str(idx) + ' = [')
                    for point in traj:
                        mt_sim_file.write('{},{};\n'.format(str(point[x_index]), str(point[y_index])))
                    mt_sim_file.write('];\n')
                    mt_sim_file.write('predicted_state_' + str(idx) + ' =  [')
                    pred_dest = traj[d_time_step]
                    mt_sim_file.write('{},{}];\n'.format(pred_dest[x_index], pred_dest[y_index]))
                    mt_sim_file.write('initial_state_' + str(idx) + ' =  [')
                    init_state = traj[0]
                    mt_sim_file.write('{},{}];\n'.format(init_state[x_index], init_state[y_index]))
            mt_sim_file.write('dest_state = [{},{}];\n'.format(dest[x_index], dest[y_index]))
            mt_sim_file.close()

    def plotResultsBasic(self, states, d_time_step, dest, dists):
        if self.data_object.dimensions == 2:
            plt.figure(1)
            x_index = 0
            y_index = 1
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    plt.plot(traj[:, x_index], traj[:, y_index], label='iteration '+str(idx))
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], '^', label='pred dest state '+str(idx))
                    print("Iteration {}: pred init state {}, pred dest state {} distance {}".format(idx, state,
                                                                                                    traj[d_time_step],
                                                                                                    dists[idx]))
                else:
                    plt.plot(traj[:, x_index], traj[:, y_index])
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], '^', label='pred dest state')
                    print("pred init state {}, pred dest state {} distance {}".format(state, traj[d_time_step],
                                                                                                    dists[idx]))

            plt.plot(dest[x_index], dest[y_index], 'r*', label='original dest state')
            plt.legend()
            plt.show()
        elif self.data_object.dimensions == 3:
            x_index = 0
            y_index = 1
            z_index = 2
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index], label='iteration ' + str(idx))
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                            label='pred dest state '+str(idx))
                    print("Iteration {}: pred init state {}, pred dest state {} distance {}".format(idx, state,
                                                                                                    traj[d_time_step],
                                                                                                    dists[idx]))
                else:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index])
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                 color='green', label='pred dest state')
                    print("pred init state {}, pred dest state {} distance {}".format(state, traj[d_time_step],
                                                                                                    dists[idx]))

            ax.scatter3D(dest[x_index], dest[y_index], dest[z_index], color='red', label='orig dest state')
            plt.legend()
            plt.show()

            # z_line = np.linspace(0, 15, 1000)
            # x_line = np.cos(z_line)
            # y_line = np.sin(z_line)
            # ax.plot3D(x_line, y_line, z_line, 'gray')

            # z_points = 15 * np.random.random(100)
            # x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
            # y_points = np.sin(z_points) + 0.1 * np.random.randn(100)
            # ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')

        elif self.data_object.dimensions == 4:
            x_index = 0
            y_index = 1
            plt.figure(1)
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    plt.plot(traj[:, x_index], traj[:, y_index], label='iteration ' + str(idx))
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], '^',
                             label='pred dest state ' + str(idx))
                    print("Iteration {}: pred init state {}, pred dest state {} distance {}".format(idx, state,
                                                                                                    traj[d_time_step],
                                                                                                    dists[idx]))
                else:
                    plt.plot(traj[:, x_index], traj[:, y_index])
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], 'g^', label='pred dest state')
                    print("pred init state {}, pred dest state {} distance {}".format(state, traj[d_time_step],
                                                                                      dists[idx]))

            plt.plot(dest[x_index], dest[y_index], 'r*', label='original dest state')
            plt.legend()
            plt.show()

            x_index = 2
            y_index = 3
            plt.figure(1)
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    plt.plot(traj[:, x_index], traj[:, y_index], label='iteration ' + str(idx))
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], '^',
                             label='pred dest state ' + str(idx))

                else:
                    plt.plot(traj[:, x_index], traj[:, y_index])
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], 'g^', label='pred dest state')

            plt.plot(dest[x_index], dest[y_index], 'r*', label='original dest state')
            plt.legend()
            plt.show()

        elif self.data_object.dimensions == 5:
            x_index = 0
            y_index = 1
            z_index = 2
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index], label='iteration ' + str(idx))
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                            label='pred dest state '+str(idx))
                    print("Iteration {}: pred init state {}, pred dest state {} distance {}".format(idx, state,
                                                                                                    traj[d_time_step],
                                                                                                    dists[idx]))
                else:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index])
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                 color='green', label='pred dest state')
                    print("pred init state {}, pred dest state {} distance {}".format(state, traj[d_time_step],
                                                                                      dists[idx]))

            ax.scatter3D(dest[x_index], dest[y_index], dest[z_index], color='red', label='orig dest state')
            plt.legend()
            plt.show()

            x_index = 3
            y_index = 4
            plt.figure(1)
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    plt.plot(traj[:, x_index], traj[:, y_index], label='iteration ' + str(idx))
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], '^',
                             label='pred dest state ' + str(idx))
                else:
                    plt.plot(traj[:, x_index], traj[:, y_index])
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], 'g^', label='pred dest state')

            plt.plot(dest[x_index], dest[y_index], 'r*', label='original dest state')
            plt.legend()
            plt.show()

        elif self.data_object.dimensions == 6:
            x_index = 0
            y_index = 1
            z_index = 2
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index], label='iteration ' + str(idx))
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                            label='pred dest state '+str(idx))
                    print("Iteration {}: pred init state {}, pred dest state {} distance {}".format(idx, state,
                                                                                                    traj[d_time_step],
                                                                                                    dists[idx]))
                else:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index])
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                 color='green', label='pred dest state')
                    print("pred init state {}, pred dest state {} distance {}".format(state, traj[d_time_step],
                                                                                      dists[idx]))

            ax.scatter3D(dest[x_index], dest[y_index], dest[z_index], color='red', label='orig dest state')
            plt.legend()
            plt.show()

            x_index = 3
            y_index = 4
            z_index = 5
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index], label='iteration ' + str(idx))
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                            label='pred dest state '+str(idx))
                else:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index])
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                 color='green', label='pred dest state')

            ax.scatter3D(dest[x_index], dest[y_index], dest[z_index], color='red', label='orig dest state')
            plt.legend()
            plt.show()

        elif self.data_object.dimensions >= 7:
            x_index = 0
            y_index = 1
            z_index = 2
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index], label='iteration ' + str(idx))
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                            label='pred dest state '+str(idx))
                    print("Iteration {}: pred init state {}, pred dest state {} distance {}".format(idx, state,
                                                                                                    traj[d_time_step],
                                                                                                    dists[idx]))
                else:
                    ax.plot3D(traj[:, x_index], traj[:, y_index], traj[:, z_index])
                    ax.scatter3D(traj[d_time_step][x_index], traj[d_time_step][y_index], traj[d_time_step][z_index],
                                 color='green', label='pred dest state')
                    print("pred init state {}, pred dest state {} distance {}".format(state, traj[d_time_step],
                                                                                      dists[idx]))

            ax.scatter3D(dest[x_index], dest[y_index], dest[z_index], color='red', label='orig dest state')
            plt.legend()
            plt.show()

            x_index = 3
            y_index = 4
            plt.figure(1)
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    plt.plot(traj[:, x_index], traj[:, y_index], label='iteration ' + str(idx))
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], '^',
                             label='pred dest state ' + str(idx))
                else:
                    plt.plot(traj[:, x_index], traj[:, y_index])
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], 'g^', label='pred dest state')

            plt.plot(dest[x_index], dest[y_index], 'r*', label='original dest state')
            plt.legend()
            plt.show()

            x_index = 5
            y_index = 6
            plt.figure(1)
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            for idx in range(len(states)):
                state = states[idx]
                traj = self.data_object.generateTrajectories(r_states=[state])[0]

                if self.n_states == 1:
                    plt.plot(traj[:, x_index], traj[:, y_index], label='iteration ' + str(idx))
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], '^',
                             label='pred dest state ' + str(idx))
                else:
                    plt.plot(traj[:, x_index], traj[:, y_index])
                    plt.plot(traj[d_time_step][x_index], traj[d_time_step][y_index], 'g^', label='pred dest state')

            plt.plot(dest[x_index], dest[y_index], 'r*', label='original dest state')
            plt.legend()
            plt.show()

    def reachDest_r(self, src=None, dest=None, d_time_step=0, threshold=0.1):
        if src is not None:
            init_state = []
            init_state += [src]
            ref_traj = self.data_object.generateTrajectories(r_states=init_state)
        else:
            ref_traj = self.data_object.generateTrajectories(sample=1)
        self.data_object.showTrajectories(ref_traj)
        new_traj = ref_traj[0]
        scale = 15/16

        model_f_name = './models/model_vp_2_v_'
        model_f_name = model_f_name + self.data_object.dynamics
        model_f_name = model_f_name + '.h5'
        model_v = None
        if path.exists(model_f_name):
            model_v = load_model(model_f_name, compile=False)
        else:
            print("Model file does not exists for the benchmark {}".format(self.data_object.dynamics))
            return

        dist = norm(new_traj[d_time_step] - dest, 2)
        if d_time_step is 0:
            for idx in range(len(new_traj)):
                curr_dist = norm(dest - new_traj[idx], 2)
                if curr_dist < dist:
                    dist = curr_dist
                    d_time_step = idx

        print("x {}, xp {}, dist {}, time {}".format(new_traj[0], new_traj[d_time_step], dist, d_time_step))

        plt.figure(1)
        plt.xlabel('x' + str(0))
        plt.ylabel('x' + str(1))
        plt.plot(dest[0], dest[1], 'r*')
        plt.plot(new_traj[d_time_step][0], new_traj[d_time_step][1], 'g*')

        plt.plot(new_traj[:, 0], new_traj[:, 1])

        plt.show()

        iter = 1
        if dist > threshold:
            output = self.findDest_r(new_traj, dest, iter, d_time_step, dist, threshold, model_v)
            print("Output is {}".format(output))

    def findDest_r(self, traj, dest, iter, d_time_step, dist, threshold, model_v, vp_val=None):
        x_val = traj[0]
        xp_val = traj[d_time_step]
        v_val = dest - x_val
        if vp_val is None:
            vp_val = dest - xp_val
        t_val = d_time_step
        data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
        predicted_v_scaled = self.evalModel(data_point, 'v', model_v)
        # predicted_v_scaled = predicted_v_scaled * 15/16
        print("v {}, vp {}".format(predicted_v_scaled, vp_val))
        new_init_state = [x_val + predicted_v_scaled]
        new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
        curr_dist = norm(dest - new_traj[d_time_step], 2)
        print("x {}, xp {}, dist {}, time{}".format(new_traj[0], new_traj[d_time_step], curr_dist, d_time_step))
        plt.figure(1)
        plt.xlabel('x' + str(0))
        plt.ylabel('x' + str(1))
        plt.plot(dest[0], dest[1], 'r*')
        plt.plot(new_traj[d_time_step][0], new_traj[d_time_step][1], 'g*')
        plt.plot(new_traj[:, 0], new_traj[:, 1])

        plt.show()
        if curr_dist < threshold:
            return 0
        elif iter >= self.iter_count:
            return -1

        if curr_dist > dist:
            curr_vp_val = dest - new_traj[d_time_step]
            curr_vp_val = vp_val / 3
            print("Current distance is greater than previous distance")
            return self.findDest_r(traj, dest, iter+1, d_time_step, dist, threshold, model_v, curr_vp_val)
        else:
            return self.findDest_r(new_traj, dest, iter+1, d_time_step, curr_dist, threshold, model_v)


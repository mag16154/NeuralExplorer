import sys
sys.path.append('../configuration-setup/')
from NNConfiguration_keras import NNConfiguration
from itertools import combinations
from configuration import configuration
import os.path
from os import path
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import losses
import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
from frechet import norm
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
import random
import re
# Custom activation function
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from rbflayer import RBFLayer, InitCentersRandom


class DataConfiguration(configuration):

    def __init__(self, timeStep=0.01, dynamics='None', dimensions=2):

        configuration.__init__(self, timeStep=timeStep, dynamics=dynamics, dimensions=dimensions)
        self.data = []
        self.dimensions = dimensions

    def dumpDataConfiguration(self):

        d_obj_f_name = './dconfigs/d_object_' + self.dynamics + '.txt'
        if path.exists(d_obj_f_name):
            os.remove(d_obj_f_name)
        d_obj_f = open(d_obj_f_name, 'w')
        d_obj_f.write(str(self.dimensions)+'\n')
        d_obj_f.write(str(self.steps)+'\n')
        d_obj_f.write(str(self.samples)+'\n')
        d_obj_f.write(str(self.timeStep)+'\n')
        for val in self.lowerBoundArray:
            d_obj_f.write(str(val))
            d_obj_f.write('\n')
        for val in self.upperBoundArray:
            d_obj_f.write(str(val))
            d_obj_f.write('\n')
        d_obj_f.close()

    def readnLoadTrajectories(self, traj_count):

        # dataObject = DataConfiguration(dynamics='MC', dimensions=2)
        # dataObject.setSteps(5338)
        # dataObject.setSamples(24)
        # dataObject.setTimeStep(0.02)
        # dataObject.readnLoadTrajectories(24)

        trajectories = []
        min_length = None
        trajs = []
        for idx in range(traj_count):
            f_name = '/home/manishg/Research/cps-falsification/verisig/examples/mountain_car/outputs/autosig_'
            f_name = f_name + str(idx)
            f_name = f_name + '.plt'
            traj = []
            if path.exists(f_name):
                traj_f = open(f_name, "r")
                lines = traj_f.readlines()
                idx = 10
                while idx < (len(lines) - 1):
                    states_0 = []
                    states_1 = []
                    for idy in range(0, 8):
                        line = lines[idx]
                        line = line[:-1]
                        tokens = line.split()
                        states_0.append(tokens[0])
                        states_1.append(tokens[1])
                        idx = idx+1
                    idx = idx + 3
                    state_0 = np.sum(np.array(states_0, dtype=float))/8
                    state_1 = np.sum(np.array(states_1, dtype=float))/8
                    traj.append(np.array([state_0, state_1]))
                if min_length is None or min_length > len(traj):
                    min_length = len(traj)
                trajs.append(traj)
            else:
                print("The file {} doesn't exists.".format(f_name))

        for traj in trajs:
            traj = traj[:min_length]
            trajectories += [np.array(traj)]
        self.trajectories = trajectories

    def createData(self, jumps=[1], dim=-1):
        assert self.lowerBoundArray is not [] and self.upperBoundArray is not []
        assert dim < self.dimensions
        self.dumpDataConfiguration()

        traj_combs = []

        end_idx = self.samples
        start_idx = 0
        traj_indices = list(range(start_idx, end_idx))
        traj_combs += list(combinations(traj_indices, 2))
        print(traj_indices)
        steps = len(self.trajectories[traj_combs[0][0]]) - 1
        xList = []
        xpList = []
        vList = []
        vpList = []
        tList = []
        for traj_pair in traj_combs:
            t_pair = list(traj_pair)
            traj_1 = self.trajectories[t_pair[0]]
            traj_2 = self.trajectories[t_pair[1]]
            x_idx = 0
            for jump in jumps:
                v_val = traj_2[x_idx] - traj_1[x_idx]
                for step in range(1, (steps - jump), jump):
                    xList.append(traj_1[x_idx])
                    vList.append(v_val)
                    t_val = step
                    xpList.append(traj_1[t_val])
                    vp_val = traj_2[t_val] - traj_1[t_val]
                    vpList.append(vp_val)
                    tList.append(t_val * self.timeStep)

            # for jump in jumps:
            #     for step in range(1, (steps - jump), 1):
            #         v_val = traj_2[step] - traj_1[step]
            #         xList.append(traj_1[step])
            #         vList.append(v_val)
            #         t_val = step + jump
            #         xpList.append(traj_1[t_val])
            #         vp_val = traj_2[t_val] - traj_1[t_val]
            #         vpList.append(vp_val)
            #         tList.append((t_val-step) * self.timeStep)

        xList = np.asarray(xList)
        xpList = np.asarray(xpList)
        vList = np.asarray(vList)
        vpList = np.asarray(vpList)
        tList = np.asarray(tList)

        self.data.append(xList.tolist())
        self.data.append(xpList.tolist())
        self.data.append(vList.tolist())
        self.data.append(vpList.tolist())
        self.data.append(tList.tolist())

    def getData(self):
        return self.data

    def getDimensions(self):
        return self.dimensions

    def getRandomDataPoints(self, num):
        data_points = []
        for val in range(num):
            idx = random.randint(0, len(self.data[0]) - 1)
            data_point = []
            data_point.append(self.data[0][idx])
            data_point.append(self.data[1][idx])
            data_point.append(self.data[2][idx])
            data_point.append(self.data[3][idx])
            data_point.append(self.data[4][idx])
            data_points.append(data_point)
        return data_points

    def createDataPoint(self, x_val, xp_val, v_val, vp_val, t_val):
        data_point = []
        data_point.append(x_val)
        data_point.append(xp_val)
        data_point.append(v_val)
        data_point.append(vp_val)
        data_point.append(t_val*self.timeStep)
        return data_point


class CreateTrainNN(NNConfiguration):

    def __init__(self, dynamics=None):
        NNConfiguration.__init__(self)
        self.dimensions = None
        self.predict_var = None
        self.dynamics = dynamics

    def createInputOutput(self, data_object, inp_vars, out_vars):
        assert data_object.getDynamics() == self.dynamics
        inp_indices = []
        out_indices = []
        data = data_object.getData()
        self.dimensions = data_object.getDimensions()

        for var in inp_vars:
            if var is 'x':
                inp_indices.append(0)
            if var is 'xp':
                inp_indices.append(1)
            if var is 'v':
                inp_indices.append(2)
            if var is 'vp':
                inp_indices.append(3)
            if var is 't':
                inp_indices.append(4)

        for var in out_vars:
            if var is 'v':
                out_indices.append(2)
                self.predict_var = var
            elif var is 'vp':
                out_indices.append(3)
                self.predict_var = var

        self.setInputSize((len(inp_indices)-1) * self.dimensions + 1)
        self.setOutputSize(self.dimensions)

        input = []
        output = []
        dataCount = len(data[0])
        for idx in range(dataCount):
            input_pair = []
            output_pair = []
            for inp in inp_indices:
                if inp is not 4:
                    input_pair = input_pair + list(data[inp][idx])
                else:
                    input_pair = input_pair + [data[inp][idx]]
            for out in out_indices:
                output_pair = data[out][idx]
            input.append(input_pair)
            output.append(output_pair)
        # print(input.shape)
        self.setInput(np.asarray(input, dtype=np.float64))
        self.setOutput(np.asarray(output, dtype=np.float64))

    def trainTestNN(self, optim='SGD', loss_fn='mae', act_fn='ReLU', layers=4, neurons=400):

        print(self.input.shape)
        x_train, x_test, y_train, y_test = train_test_split(self.input, self.output, test_size=self.test_size,
                                                            random_state=1)
        print(x_train.shape)
        print(y_train.shape)

        inputs_train = x_train
        targets_train = y_train
        inputs_test = x_test
        targets_test = y_test

        def swish(x):
            return K.sigmoid(x) * 5

        get_custom_objects().update({'custom_activation': Activation(swish)})

        if act_fn is 'Tanh':
            act = 'tanh'
        elif act_fn is 'Sigmoid':
            act = 'sigmoid'
        elif act_fn is 'Exponential':
            act = 'exponential'
        elif act_fn is 'Linear':
            act = 'linear'
        elif act_fn is 'SoftMax':
            act = 'softmax'
        elif act_fn is 'Swish':
            act = swish
        else:
            act = 'relu'
            print("\nSetting the activation function to default - ReLU.\n")

        def mre_loss(y_true, y_pred):
            # loss = K.mean(K.square(y_pred - y_true)/K.square(y_pred))
            # loss = K.mean(K.abs((y_true - y_pred) / K.clip(K.abs(y_pred), K.epsilon(), np.inf)))
            # my_const = tf.constant(0.0001, dtype=tf.float32)
            # loss = K.mean(K.square(y_pred - y_true) / (K.square(y_pred)+3))
            # loss = K.mean(K.square(y_pred - y_true) / (K.square(y_pred) + 2))
            # loss = tf.reduce_mean(tf.divide((tf.subtract(y_pred, y_true))**2, (y_pred**2 + 1e-10)))

            loss_1 = K.sum(K.mean(K.square(y_pred - y_true) / (K.square(y_pred) + 4), axis=1))
            loss_0 = K.sum(K.mean(K.square(y_pred - y_true) / (K.square(y_pred) + 4), axis=0))

            bool_idx = K.greater((loss_1 - loss_0), 0)

            # Vanderpol, Jetengine. Buckling
            loss = K.switch(bool_idx, loss_1 * 1.4 + loss_0 * 0.9, loss_1 * 0.9 + loss_0 * 1.4)

            # Spring Pendulum and rest, if not specified
            # loss = K.switch(bool_idx, loss_1 * 2.0 + loss_0 * 1.3, loss_1 * 1.3 + loss_0 * 2.0)

            return loss

        model = Sequential()

        # model.add(RBFLayer(neurons, initializer=InitCentersRandom(x_train), betas=40.0, input_shape=(7,)))
        # model.add(RBFLayer(5, 0.5))
        # model.add(Dense(2, activation=act))
        # model.add(Dense(2, activation='linear'))

        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(neurons, activation=act, input_dim=self.input_size))
        for idx in range(layers):
            model.add(Dense(neurons, activation=act))
            model.add(BatchNormalization())
        model.add(Dense(self.output_size, activation='linear'))

        if optim is 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif optim is 'RMSProp':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.9)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        if loss_fn is 'mse':
            model.compile(loss=losses.mean_squared_error, optimizer=optimizer, metrics=['accuracy'])
        elif loss_fn is 'mae':
            model.compile(loss=losses.mean_absolute_error, optimizer=optimizer, metrics=['accuracy', 'mse'])
        elif loss_fn is 'mape':
            model.compile(loss=losses.mean_absolute_percentage_error, optimizer=optimizer, metrics=['accuracy', 'mse'])
        elif loss_fn is 'mre':
            model.compile(loss=mre_loss, optimizer=optimizer, metrics=['accuracy', 'mse'])

        model.fit(inputs_train, targets_train,
                  epochs=self.epochs,
                  batch_size=self.batch_size, verbose=1)
        # score = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        # print(score)

        predicted_train = model.predict(inputs_train)
        print(predicted_train.shape)

        if self.predict_var is 'v':
            v_f_name = "./models/model_vp_2_v_"
            v_f_name = v_f_name + str(self.dynamics)
            v_f_name = v_f_name + ".h5"
            if path.exists(v_f_name):
                os.remove(v_f_name)
            model.save(v_f_name)
        elif self.predict_var is 'vp':
            vp_f_name = "./models/model_v_2_vp_"
            vp_f_name = vp_f_name + str(self.dynamics)
            vp_f_name = vp_f_name + ".h5"
            if path.exists(vp_f_name):
                os.remove(vp_f_name)
            model.save(vp_f_name)

        # mse_train = mean_squared_error(predicted_train, targets_train)

        predicted_test = model.predict(inputs_test)
        predicted_train = tf.cast(predicted_train, tf.float64)
        predicted_test = tf.cast(predicted_test, tf.float64)

        with tf.Session() as sess:
            # print(sess.run(mse_train))
            # print(sess.run(mse_test))
            predicted_train = sess.run(predicted_train)
            predicted_test = sess.run(predicted_test)

        max_se_train = 0.0
        min_se_train = 1000.0
        mse_train = 0.0
        for idx in range(len(targets_train)):
            dist_val = norm(predicted_train[idx] - targets_train[idx], 2)
            if dist_val > max_se_train:
                max_se_train = dist_val
            if dist_val < min_se_train:
                min_se_train = dist_val
            mse_train += dist_val
        mse_train = mse_train / (len(targets_train))

        min_se_test = 1000.0
        max_se_test = 0.0
        mse_test = 0.0
        for idx in range(len(targets_test)):
            dist_val = norm(predicted_test[idx] - targets_test[idx], 2)
            if dist_val > max_se_test:
                max_se_test = dist_val
            if dist_val < min_se_test:
                min_se_test = dist_val
            mse_test += dist_val
        mse_test = mse_test / (len(targets_test))

        # print("Max RMSE Train {}".format(max_se_train))
        # print("Max RMSE Test {}".format(max_se_test))
        # print("Min RMSE Train {}".format(min_se_train))
        # print("Min RMSE Test {}".format(min_se_test))
        print("Mean RMSE Train {}".format(mse_train))
        print("Mean RMSE Test {}".format(mse_test))

        max_re_train = 0.0
        mre_train = 0.0
        for idx in range(len(targets_train)):
            dist_val = norm(predicted_train[idx] - targets_train[idx], 2)
            dist_val = (dist_val / (norm(targets_train[idx], 2)))
            if dist_val > max_re_train:
                max_re_train = dist_val
            mre_train += dist_val
        mre_train = mre_train / (len(targets_train))

        max_re_test = 0.0
        mre_test = 0.0
        for idx in range(len(targets_test)):
            dist_val = norm(predicted_test[idx] - targets_test[idx], 2)
            dist_val = (dist_val / (norm(targets_test[idx], 2)))
            if dist_val > max_re_test:
                max_re_test = dist_val
            mre_test += dist_val
        mre_test = mre_test / (len(targets_test))
        # print("Max Relative Error Train {}".format(max_re_train))
        # print("Max Relative Error Test {}".format(max_re_test))
        print("Mean Relative Error Train {}".format(mre_train))
        print("Mean Relative Error Test {}".format(mre_test))

        self.visualizePerturbation(targets_train, predicted_train)
        self.visualizePerturbation(targets_test, predicted_test)

    def visualizePerturbation(self, t, p):
        # targets = t.detach().numpy()
        # predicted = tf.Session().run(p)
        targets = t
        predicted = p
        print(targets.shape)
        print(predicted.shape)
        t_shape = targets.shape
        for dim in range(self.dimensions):

            y_test_plt = []
            predicted_test_plt = []

            for idx in range(0, 999 * 2):
                y_test_plt += [targets[idx][dim]]
                predicted_test_plt += [predicted[idx][dim]]

            plt.figure()
            plt.plot(y_test_plt)
            plt.plot(predicted_test_plt)
            plt.show()

        if self.predict_var is 'v':
            f_name = './outputs/v_vals_'
        else:
            f_name = './outputs/vp_vals_'
        f_name = f_name + self.dynamics
        f_name = f_name + ".txt"
        if path.exists(f_name):
            os.remove(f_name)
        vals_f = open(f_name, "w")

        for idx in range(0, t_shape[0]-1):
            vals_f.write(str(targets[idx]))
            vals_f.write(" , ")
            vals_f.write(str(predicted[idx]))
            vals_f.write(" ... ")
            t_norm = norm(targets[idx], 2)
            vals_f.write(str(t_norm))
            vals_f.write(" , ")
            p_norm = norm(predicted[idx], 2)
            vals_f.write(str(p_norm))
            vals_f.write("\n")

        vals_f.close()

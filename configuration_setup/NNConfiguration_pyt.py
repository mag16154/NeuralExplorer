import matplotlib.pyplot as plt
from configuration import configuration
from frechet import normTrajectory, norm
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import numpy as np


class RadialBasisFunction(nn.Module):
    def __init__(self, gamma=None, centers=None):
        self.centers = centers
        # initialize alpha
        if gamma is None:
            self.gamma = torch.nn.parameter(torch.tensor(0.0))  # create a tensor out of alpha
        else:
            self.gamma = torch.nn.parameter(torch.tensor(gamma))  # create a tensor out of alpha

        self.gamma.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
            print(x.shape)
            return x
            # return torch.exp(-self.gamma * np.linalg.norm(center - data_point) ** 2)


class NNConfiguration(configuration):

    def __init__(self, timeStep=0.01, steps=100, samples=50, dynamics='None', dimensions=2, lowerBound=[], upperBound=[]):

        configuration.__init__(self, timeStep, steps, samples, dynamics, dimensions, lowerBound, upperBound)
        self.relativeError = []
        self.mseError = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.input_size = None
        self.output_size = None
        self.dim_train = -1
        self.num_epochs = 300
        self.learning_rate = 0.01

    def setInputSize(self, input_size):
        self.input_size = input_size

    def setOutputSize(self, output_size):
        self.output_size = output_size

    def setXtrain(self, x_train):
        self.x_train = x_train

    def setYtrain(self, y_train):
        self.y_train = y_train

    def setXtest(self, x_test):
        self.x_test = x_test

    def setYtest(self, y_test):
        self.y_test = y_test

    def setdimTrain(self, dim):
        self.dim_train = dim

    def generateTrajectories(self):
        self.storeTrajectories()

    def setEpochs(self, epochs):
        self.num_epochs = epochs

    def setLearningRate(self, learningRate):
        self.learning_rate = learningRate

    def trainTestNN(self, optim='SGD', normalize=False, loss_fn='mse', neurons=400, layers=4, act_fn='ReLU'):

        scale = 2

        print(len(self.trajectories))
        # plt.figure()
        for idx in range(len(self.trajectories)):
            traj = self.trajectories[idx]
            norm_vals = []
            for idy in range(round(len(traj)/scale)):
                norm_val = norm(traj[idy], -1)
                norm_vals += [norm_val]
            # plt.plot(norm_vals)

        # print(len(self.x_train))
        print(self.x_train.shape)
        print(self.y_train.shape)

        modules = []

        if optim == 'LBFGS':
            modules.append(nn.Linear(self.input_size, self.output_size))
            # model = nn.Linear(self.input_size, self.output_size)
        else:
            hidden_num_units = neurons
            modules.append(nn.Linear(self.input_size, hidden_num_units))
            if act_fn is 'ReLU':
                for idx in range(layers):
                    modules.append(nn.ReLU())
            elif act_fn is 'Tanh':
                for idx in range(layers):
                    modules.append(nn.Tanh())
            elif act_fn is 'Sigmoid':
                for idx in range(layers):
                    modules.append(nn.Sigmoid())
            elif act_fn is 'LogSigmoid':
                for idx in range(layers):
                    modules.append(nn.LogSigmoid())
            else:
                print("\nProvide a valid activation function: ReLU, Tanh, Sigmoid or LogSigmoid.\n")
                return
            # modules.append(RadialBasisFunction(gamma=0.5))
            modules.append(nn.Linear(hidden_num_units, self.output_size))

        model = nn.Sequential(*modules)
        # Loss and optimizer
        criterion = nn.MSELoss()

        def mre_loss(output, target):
            # relativeloss = torch.norm(output - target, dim=0)/torch.norm(output, dim=0)
            # rl_np = relativeloss.detach().numpy()
            # print(rl_np)
            # max_dim = 0
            # for idx in range(self.dimensions):
            #    if rl_np[idx] > rl_np[max_dim]:
            #        max_dim = idx

            # weights_np = np.ones(self.dimensions)
            # weights_np[max_dim] = 1.1
            # for idx in range(self.dimensions):
            #     if idx != max_dim:
            #        weights_np[idx] = 1
            # print(max_dim)
            # weights = torch.from_numpy(weights_np).float()
            # upd_rl = torch.mul(relativeloss, weights)

            # loss = torch.mean(torch.sum(((output-target)*(output-target))/(output*output), dim=0))
            # loss = torch.mean((((output - target) * (output - target)) / (output * output)).mean(-1))
            # diff = output - target
            # abc = torch.sqrt((diff * diff) / (output*output))
            # loss = torch.mean(abc)
            loss = torch.mean(torch.norm(output - target) / torch.norm(output))
            return loss

        def mse_loss(output, target):
            loss = torch.mean((output - target)**2)
            return loss

        # criterion = nn.MultiLabelSoftMarginLoss()
        # criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        if optim == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)
        elif optim == 'LBFGS':
            optimizer = torch.optim.LBFGS(model.parameters(), lr=self.learning_rate)
        elif optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Convert numpy arrays to torch tensors
        inputs_train = torch.from_numpy(self.x_train).float()
        targets_train = torch.from_numpy(self.y_train).float()
        inputs_test = torch.from_numpy(self.x_test).float()
        targets_test = torch.from_numpy(self.y_test).float()

        if normalize is True:
            input_min = np.min(self.x_train, axis=0)
            input_max = np.max(self.x_train, axis=0)
            inputs_train = torch.from_numpy((self.x_train-input_min)/(input_max-input_min)).float()

            target_min = np.amin(self.y_train, axis=0)
            target_max = np.amax(self.y_train, axis=0)
            targets_train = torch.from_numpy((self.y_train - target_min)/(target_max - target_min)).float()

            inputs_test = torch.from_numpy((self.x_test - input_min)/(input_max - input_min)).float()
            targets_test = torch.from_numpy((self.y_test - target_min) / (target_max - target_min)).float()

        prev_loss = None

        model_file = 'model_default.ckpt'
        if loss_fn is 'mse':
            model_file = 'model_mse.ckpt'
        elif loss_fn is 'mre':
            model_file = 'model_mre.ckpt'

        for epoch in range(self.num_epochs):

            if optim == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    outputs_train = model(inputs_train)
                    loss = criterion(outputs_train, targets_train)
                    loss.backward()
                    return loss

                optimizer.step(closure)

            else:
                # Forward pass
                outputs_train = model(inputs_train)

                if loss_fn == 'mse':
                    loss = mse_loss(outputs_train, targets_train)
                elif loss_fn == 'mre':
                    loss = mre_loss(outputs_train, targets_train)
                else:
                    loss = criterion(outputs_train, targets_train)

                if prev_loss is None or prev_loss.item() > loss.item():
                    prev_loss = loss
                    torch.save(model, model_file)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 5 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))

        model = torch.load(model_file)
        predicted_train = model(inputs_train)
        mse_train = mean_squared_error(predicted_train.detach().numpy(), targets_train.detach().numpy())
        print(mse_train)

        predicted_test = model(inputs_test)
        mse_test = mean_squared_error(predicted_test.detach().numpy(), targets_test.detach().numpy())
        print(mse_test)

        re_train = torch.mean(torch.norm(predicted_train - targets_train) / torch.norm(predicted_train))
        print(re_train.item())

        re_test = torch.mean(torch.norm(predicted_test - targets_test) / torch.norm(predicted_test))
        print(re_test.item())
        # relErrSum = 0.0
        # for idx in range(predicted_train.shape[0]):
        #    distVal = norm(predicted_train[idx] - self.y_train[idx], -1)
        #    relErrSum += (distVal / (norm(predicted_train[idx], -1)))
        # relErr = relErrSum/(predicted_train.shape[0])
        # print(relErr)

        # relErrSum = 0.0
        # print(predicted_test.shape[0])
        # for idx in range(predicted_test.shape[0]):
        #    distVal = norm(predicted_test[idx] - self.y_test[idx], -1)
        #    relErrSum += (distVal / (norm(predicted_test[idx], -1)))
        # relErr = relErrSum/(predicted_test.shape[0])
        # print(relErr)

        self.visualizePerturbation(targets_train, predicted_train)
        self.visualizePerturbation(targets_test, predicted_test)

    def visualizePerturbation(self, t, p):
        targets = t.detach().numpy()
        predicted = p.detach().numpy()

        if self.dim_train == -1:
            for dim in range(self.dimensions):

                y_test_plt = []
                predicted_test_plt = []

                print(self.y_test.shape)
                for idx in range(0, 999):
                    y_test_plt += [targets[idx][dim]]
                    predicted_test_plt += [predicted[idx][dim]]

                plt.figure()
                plt.plot(y_test_plt)
                plt.plot(predicted_test_plt)
                plt.show()
        else:
            y_test_plt = []
            predicted_test_plt = []

            print(self.y_test.shape)
            for idx in range(0, 999):
                y_test_plt.append(targets[idx])
                predicted_test_plt.append(predicted[idx])

            plt.figure()
            plt.plot(y_test_plt)
            plt.plot(predicted_test_plt)
            plt.show()

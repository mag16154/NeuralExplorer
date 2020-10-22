
class NNConfiguration(object):

    def __init__(self, dnn_rbf='dnn'):
        self.relativeError = []
        self.mseError = []
        self.input = None
        self.output = None
        self.input_size = None
        self.output_size = None
        self.epochs = 200
        self.learning_rate = 0.01
        self.batch_size = 64
        self.test_size = 0.1
        self.DNN_or_RBF = dnn_rbf

    def setInputSize(self, input_size):
        self.input_size = input_size

    def setOutputSize(self, output_size):
        self.output_size = output_size

    def setEpochs(self, epochs):
        self.epochs = epochs

    def setLearningRate(self, learningRate):
        self.learning_rate = learningRate

    def setBatchSize(self, batchSize):
        self.batch_size = batchSize

    def setInput(self, input):
        self.input = input

    def setOutput(self, output):
        self.output = output

    def setTestSize(self, test_size):
        self.test_size = test_size

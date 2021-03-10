import tensorflow as tf


class ModelCreator:
    def __init__(self, numOfLayers, numOfNeurons, neuronDecay, constantNeurons, batchNorm, dropout, dropoutRate):
        self.numOfLayers = int(numOfLayers)
        self.numOfNeurons = numOfNeurons
        self.neuronDecay = neuronDecay
        self.constantNeurons = constantNeurons
        self.batchNorm = batchNorm
        self.dropout = dropout
        self.dropoutRate = dropoutRate

    def makeModelWandB(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(2, 15)))
        # model.add(tf.keras.layers.BatchNormalization())

        # model.add(tf.keras.layers.Dense(10, activation='relu'))
        # model.add(tf.keras.layers.Dense(8, activation='relu'))
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(5, activation='relu')) # relu is more of a continuous activation, from 0 to 1
        # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # sigmoid function ensures the neuron has outputs like 0 and 1
        dropoutPlace = round(self.numOfLayers / 2)
        for i in range(self.numOfLayers):
            if not self.constantNeurons and i != 0:
                numOfNeurons = round(self.numOfNeurons * self.neuronDecay) + 1
            model.add(tf.keras.layers.Dense(self.numOfNeurons, activation='relu', name="dense" + str(i)))
            if self.batchNorm:
                model.add(tf.keras.layers.BatchNormalization())
            if self.dropout and i == dropoutPlace:
                model.add(tf.keras.layers.Dropout(self.dropoutRate))
            if self.numOfLayers - i == 2:
                model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(2, activation='linear', name='output'))
        # print(model.summary())
        return model

    @staticmethod
    def handMadeModel():
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(16, activation='relu'),
        #     tf.keras.layers.Dense(8, activation='relu'),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(2, 15)),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        print(model.summary())
        print(model.output_shape)
        return model

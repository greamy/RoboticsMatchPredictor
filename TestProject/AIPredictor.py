import tensorflow as tf
from tensorflow.keras import losses
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from TestProject.DataCollection import DataCollectionAnalysis


dataCollectMilford = DataCollectionAnalysis("FIM District Milford Event")
milford = dataCollectMilford.getDataAndLabels()
milfordData = milford[0]
milfordLabels = milford[1]
print("MILFORD EVENT RETRIEVED!!!")

dataCollectJackson = DataCollectionAnalysis("FIM District Jackson Event")
jackson = dataCollectJackson.getDataAndLabels()
jacksonData = jackson[0]
jacksonLabels = jackson[1]
print("JACKSON EVENT RETRIEVED!!!")

dataCollectKettering2 = DataCollectionAnalysis("FIM District Kettering University Event #2")
kettering2 = dataCollectKettering2.getDataAndLabels()
kettering2Data = kettering2[0]
kettering2Labels = kettering2[1]
print("KETTERING 2 RETRIEVED")

trainData = np.append(milfordData, kettering2Data, 0)
trainLabels = np.append(milfordLabels, kettering2Labels)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=[tf.keras.metrics.BinaryAccuracy()])
model.fit(trainData, trainLabels, epochs=100)

print("Evaluate on test data")
results = model.evaluate(jacksonData, jacksonLabels, batch_size=1)
print("test loss, test acc:", results)


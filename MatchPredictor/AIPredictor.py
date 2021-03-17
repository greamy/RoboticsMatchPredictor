import tensorflow as tf
from tensorflow.keras import losses
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from MatchPredictor.DataCollection import DataCollectionAnalysis

# Returns labels and data in numpyArrays, and data has been cleaned of missing info.
def getAllData(eventsToGet):
    allData = np.array([])
    allLabels = np.array([])
    for i in range(len(eventsToGet)):
        dataCollect = DataCollectionAnalysis(eventsToGet[i])
        full = dataCollect.getDataAndLabels()
        if i == 0:
            allData = full[0]
            allLabels = full[1]
        else:
            allData = np.append(allData, full[0], axis=0)
            allLabels = np.append(allLabels, full[1], axis=0)
        print(eventsToGet[i] + " EVENT RETRIEVED")
    return [allData, allLabels]


gotInput = False
while not gotInput:
    fileOrAPI = input("Load from Files or API (f/a)?")
    if fileOrAPI == "a":
        print("Loading data from The Blue Alliance API, then saving it to local files. This may take a while...")
        # Kettering #1 and Southfield doesnt work, JSON is bad in API.
        events = ["NE District Granite State Event", "Great Northern Regional", "FNC District Wake County Event",
                  "Regional Monterrey", "Greater Kansas City Regional", "ISR District Event #1", "ISR District Event #2",
                  "PCH District Gainesville Event presented by Automation Direct", "NE District Northern CT Event",
                  "Los Angeles North Regional", "FIM District Traverse City Event", "FIM District St. Joseph Event",
                  "FIM District Macomb Community College Event", "FIM District Milford Event", "FIM District Jackson Event",
                  "FIM District Kettering University Event #2", "FIM District Kingsford Event"]
        allData = getAllData(events)  # this returns [data, labels] where data and labels are compiled from each match
        x_train, x_test, y_train, y_test = train_test_split(allData[0], allData[1], test_size=0.2, shuffle=True)

        np.save("x_train", x_train)
        np.save("x_test", x_test)
        np.save("y_train", y_train)
        np.save("y_test", y_test)
        gotInput = True

    elif fileOrAPI == "f":
        x_train = np.load("x_train.npy", allow_pickle=True)
        x_test = np.load("x_test.npy", allow_pickle=True)
        y_train = np.load("y_train.npy", allow_pickle=True)
        y_test = np.load("y_test.npy", allow_pickle=True)
        gotInput = True
    else:
        print("Invalid Input. Please Try again")

model = tf.keras.Sequential([
    # tf.keras.layers.Conv1D(2, 2, activation='relu', input_shape=(2, 9)),
    tf.keras.layers.Dense(10, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation='relu'),  # relu is more of a continuous activation, from 0 to 1
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid function ensures the neuron has outputs like 0 and 1
  ])

epochs = 500

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Adam is a SGD (Stochastic Gradient Descent) Algorithm.
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer=opt,
              metrics=[tf.keras.metrics.BinaryAccuracy()])
history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

print("\nEvaluate on test data")
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, epochs+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

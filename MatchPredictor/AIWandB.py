import tensorflow as tf
from tensorflow.keras import losses
# from matplotlib import cm
# from matplotlib import gridspec
import os
from matplotlib import pyplot as plt
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from DataCollection import DataCollectionAnalysis
from ModelCreator import ModelCreator
from CustomException import APIException

# Returns labels and data in numpyArrays, and data has been cleaned of missing info.
# Can be used for many events because eventsToGet should be array.
def getAllData(scorePredict, year="2020", predict=False):
    allData = np.array([])
    allLabels = np.array([])
    if not predict:
        eventsToGet = DataCollectionAnalysis.getYearEvents(year)
    if predict:
        eventsToGet = ["Greater Kansas City Regional"]
        for i in range(len(eventsToGet)):
            dataCollect = DataCollectionAnalysis(eventsToGet[i], scorePredict, year)
            allData = dataCollect.allData[0]
            allLabels = dataCollect.allData[1]
    else:
        successCount = 0
        i = 0
        while successCount <= 20:
            try:
                dataCollect = DataCollectionAnalysis(eventsToGet[i], scorePredict, year)
            except APIException as e:
                print("Error occured. Skipping " + eventsToGet[i])
                print(str(e) + "\n")
                i += 1
                continue
            if len(allData) == 0:
                allData = dataCollect.allData[0]
                allLabels = dataCollect.allData[1]
            else:
                allData = np.append(allData, dataCollect.allData[0], axis=0)
                allLabels = np.append(allLabels, dataCollect.allData[1], axis=0)
            successCount += 1
            i += 1
    return [allData, allLabels]


# Reformat labels for use in the model.
# y = array of labels
# Returns: np Array of labels
def fixFormatLabels(y):
    newArray = np.array([[]])
    for i in range(len(y)):
        score1 = y[i][0]
        score2 = y[i][1]
        newArray = np.append(newArray, [score1, score2])
    y = newArray
    y = np.reshape(y, (-1, 2))
    return y


gotInput = False
gotPredict = False
scoresOrWin = ''
year = "2016"
while not gotInput:
    fileOrAPI = input("\nLoad from Files or API (f/a)?")
    # scoresOrWin = input("\nPredict scores, or Winners (s/w)?")
    # fileOrAPI = 'f'
    scoresOrWin = 's'
    if scoresOrWin != 's' and scoresOrWin != 'w':
        print("Invalid Input. Please Try again")
        gotInput = False
        continue
    if fileOrAPI == "a":
        gotPredict = False
        print("\nLoading data from The Blue Alliance API, then saving it to local files. This may take a while...")

        # Kettering #1, Southfield, and Great Northern Regional doesnt work, missing Data for them in JSON files.
        # events = ["NE District Granite State Event", "FNC District Wake County Event",
        #           "Regional Monterrey", "Greater Kansas City Regional", "ISR District Event #1", "ISR District Event #2",
        #           "PCH District Gainesville Event presented by Automation Direct", "NE District Northern CT Event",
        #           "Los Angeles North Regional", "FIM District Traverse City Event", "FIM District St. Joseph Event",
        #           "FIM District Macomb Community College Event", "FIM District Milford Event", "FIM District Jackson Event",
        #           "FIM District Kettering University Event #2", "FIM District Kingsford Event"]

        scores = False
        fileName = '_wins'
        if scoresOrWin == 's':
            scores = True
            fileName = '_scores'
            print("\nData is being collected to predict scores.")

        # this returns [data, labels] where data and labels are compiled from each match

        allData = getAllData(scores, year)
        allData[1] = fixFormatLabels(allData[1])
        x_train, x_test, y_train, y_test = train_test_split(allData[0], allData[1], test_size=0.2, shuffle=True)
        if not os.path.isdir("data/" + year):
            os.mkdir("data/" + year)
            print("data/" + year + " created")
        np.save("data/" + year + "/x_train" + fileName, x_train)
        np.save("data/" + year + "/x_test" + fileName, x_test)
        np.save("data/" + year + "/y_train" + fileName, y_train)
        np.save("data/" + year + "/y_test" + fileName, y_test)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
        gotInput = True

    elif fileOrAPI == "f":
        gotPredict = True
        if scoresOrWin == 's':
            fileName = '_scores'
            print("\nData is being collected from files to predict scores.")
        try:
            x_train = np.load("data/" + year + "/x_train" + fileName + ".npy", allow_pickle=True)
            x_test = np.load("data/" + year + "/x_test" + fileName + ".npy", allow_pickle=True)
            y_train = np.load("data/" + year + "/y_train" + fileName + ".npy", allow_pickle=True)
            y_test = np.load("data/" + year + "/y_test" + fileName + ".npy", allow_pickle=True)
        except FileNotFoundError:
            print("Data is not saved. You must load data from API before loading from files.")
            continue
        y_train = fixFormatLabels(y_train)
        y_test = fixFormatLabels(y_test)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
        gotInput = True
    else:
        print("Invalid Input. Please Try again")


# Uses the model.predict() function on prediction data to test accuracy of model. This is called after training.
# model = trained tf.keras.Model
def modelPredict(model, gotPredict):
    if gotPredict:
        print("Getting prediction data from API")
        if scoresOrWin == 's':
            score = True
        else:
            score = False
        predictionData = getAllData(scorePredict=score, year=year, predict=True)
        predictionData[1] = fixFormatLabels(predictionData[1])
        x_predict = predictionData[0]
        y_predict = predictionData[1]
        np.save("data/" + year + "/x_predict", x_predict)
        np.save('data/' + year + '/y_predict', y_predict)
        gotPredict = True
    else:
        try:
            x_predict = np.load('data/' + year + '/x_predict.npy')
            y_predict = np.load('data/' + year + '/y_predict.npy')
        except Exception:
            print("Could not find prediction data. Loading from API...")
            predictionData = getAllData(scorePredict=True, year=year, predict=True)
            predictionData[1] = fixFormatLabels(predictionData[1])
            print(predictionData)
            x_predict = predictionData[0]
            print(x_predict)
            y_predict = predictionData[1]
            np.save("data/" + year + "/x_predict", x_predict)
            np.save('data/' + year + '/y_predict', y_predict)


    predictions = model.predict(x_predict)
    correctCount = 0
    for i in range(len(predictions)):
        bluePScore = predictions[i][0]
        redPScore = predictions[i][1]
        blueRScore = y_predict[i][0]
        redRScore = y_predict[i][1]
        if bluePScore > redPScore and blueRScore > redRScore:
            correctCount += 1
        elif bluePScore < redPScore and blueRScore < redRScore:
            correctCount += 1
    accuracy = correctCount / len(predictions)
    return accuracy


wandb.login()
configs = {
    "learning_rate": 0.0005,
    "epochs": 100,
    "batch_size": 40,
    'layers': 3,
    'neurons': 32,
    'neuron_decay': 0.75,
    'constant_neurons': False,
    'batch_norm': False,
    'dropout': False,
    'dropout_rate': 0.2319
    }
run = wandb.init(project='2020MatchPredictor', config=configs)
config = wandb.config
# print(np.shape(x_train)[1:])
creator = ModelCreator(config.layers, config.neurons, config.neuron_decay, config.constant_neurons, config.batch_norm,
                  config.dropout, config.dropout_rate, np.shape(x_train)[1:])
model = creator.makeModelWandB()
# model = handMadeModel()

epochs = configs['epochs']
opt = tf.keras.optimizers.Adam(learning_rate=configs['learning_rate'])  # Adam is a SGD (Stochastic Gradient Descent) Algorithm.
if scoresOrWin == 'w':
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])
                    # metrics=[tf.keras.metrics.BinaryAccuracy()])
else:
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mean_squared_error'])
# history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

if scoresOrWin == 'w':
    history = {"epochs": [], "loss": [], "acc": [], "val_loss": [], "val_accuracy": []}
if scoresOrWin == 's':
    history = {"epochs": [], "loss": [], "val_loss": []}
for epoch in range(epochs):
    print("Epoch #: " + str(epoch) + "/" + str(epochs))
    run = model.fit(x_train, y_train, epochs=1, batch_size=configs['batch_size'], validation_data=(x_val, y_val))
    # print("loss " + str(run.history['loss']))
    # print("acc " + str(run.history['binary_accuracy']))
    # print("val loss " + str(run.history['val_loss']))
    # print("val acc " + str(run.history['val_binary_accuracy']))
    # print(run.history.keys())
    if scoresOrWin == 'w':
        wandb.log({'epochs': epoch,
                   'loss': run.history['loss'][0],
                   'acc': run.history['acc'][0],
                   'val_loss': run.history['val_loss'][0],
                   'val_accuracy': run.history['val_acc'][0]
        })
        history["epochs"].append(epoch)
        history["loss"].append(run.history['loss'])
        history["acc"].append(run.history['acc'][0])
        history["val_loss"].append(run.history['val_loss'][0])
        history["val_accuracy"].append(run.history['val_acc'][0])
    if scoresOrWin == 's':
        history["epochs"].append(epoch)
        history["loss"].append(run.history['loss'])
        history["val_loss"].append(run.history['val_loss'][0])
        if epoch % 10 == 0:
            accuracy = modelPredict(model, gotPredict)
            wandb.log({'epochs': epoch,
                       'loss': run.history['loss'][0],
                       'acc': accuracy,
                       'val_loss': run.history['val_loss'][0],
                       })
        else:
            wandb.log({'epochs': epoch,
                       'loss': run.history['loss'][0],
                       'val_loss': run.history['val_loss'][0],
                       })

print("\nEvaluate on test data")
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)


if scoresOrWin == 's':
    accuracy = modelPredict(model, gotPredict)
    print("ACCURACY: " + str(accuracy))

    path = "savedModels/" + year
    checkpoint_path = path + "/best_model"
    try:
        with open(path + "/bestLoss.txt", 'r') as reader:
            loss = float(reader.readline())
    except FileNotFoundError:
        try:
            os.makedirs(path + "/best_model")
        except FileExistsError:
            None
        with open(path + "/bestLoss.txt", 'w') as write:
            loss = 1000000
            write.write("1000000")
    if results[0] <= loss:
        print("Found new Lowest Loss, saving model!")
        lossFile = open(path + "/bestLoss.txt", 'w')
        parameters = open(checkpoint_path + "/hyperparameters.txt", 'w')
        try:
            model.save_weights(checkpoint_path + "/cp.ckpt")
            lossFile.write(str(results[0]))
            parameters.write(str(config["learning_rate"]) + "," + str(config["epochs"]) + "," +
                             str(config["batch_size"]) + "," + str(config["layers"]) + "," + str(config["neurons"]) + "," +
                             str(config["neuron_decay"]) + "," + str(config["constant_neurons"]) + "," +
                             str(config["batch_norm"]) + "," + str(config["dropout"]) + "," + str(config["dropout_rate"])
                             + ",\n")
        finally:
            lossFile.close()
            parameters.close()

# Prints out graph of loss over time after training and validation.
loss_train = history['loss']
loss_val = history['val_loss']
epochs = range(1, epochs+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

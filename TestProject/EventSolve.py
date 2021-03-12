import random
from . import DataCollection
from . import ModelCreator
import pandas as pd
import numpy as np
# import keyboard


class EventSolve:

    def __init__(self, model, dataCollector, teamData, targetTeam):
        self.model = model
        self.collector = dataCollector
        self.teamData = teamData
        self.teams = teamData['team_key']
        self.numTeams = len(self.teams.index)
        self.teamData = self.formatTeamData(teamData)
        self.targetTeam = "frc" + targetTeam
        self.targetTeamIndex = self.getTeamIndex(self.targetTeam)
        self.blue2Index = 0
        self.blue3Index = 0
        self.performance = [0 for x in range(self.numTeams)]
        self.scores = [[] for y in range(self.numTeams)]
        self.done = False
        self.count = 0

    # Continuously loops through possible alliance combination and makes predictions to find best team in the event
    def search(self):
        while not self.done:
            red = self.genRedAlliance()
            blue = self.genBlueAlliance()
            alliances = self.combineAlliances(blue, red)
            # print(alliances)
            allianceData = self.getAlliancesData(alliances)
            prediction = self.model(allianceData, training=False).numpy()
            prediction = np.reshape(prediction, newshape=2)
            # print(prediction)
            self.updateScores(prediction[0])
            # try:
            #     if keyboard.is_pressed('q'):
            #         break
            # except:
            #     continue
            if self.count >= 10:
                break
            # self.setDone()
            # keyboard.add_hotkey('shift+q', self.done)
        return self.output()

    def setDone(self):
        print("done!")
        self.done = True

    # Opposing alliance doesn't need to be optimized, so this creates random teams to fight. Also checks for duplicates
    # Doesn't check for duplicates from the blue side, because it is better to fight every team (including yourself)
    def genRedAlliance(self):
        numTeams = len(self.teams)
        red1 = random.randint(0, numTeams-1)
        red2 = random.randint(0, numTeams-1)
        red3 = random.randint(0, numTeams-1)
        if (red1 == red2) or (red1 == red3) or (red2 == red3):
            return self.genRedAlliance()
        red = [self.teams[red1], self.teams[red2], self.teams[red3]]
        return red

    #  Called every loop, and loops through the list of teams, skipping targetTeam and ensuring no duplicates.
    def genBlueAlliance(self):
        self.blue3Index += 1

        if self.blue3Index >= self.numTeams-1:  # Checks if we ran through the final position, then moves 2nd one up
            self.blue3Index = 0
            self.blue2Index += 1
        if self.blue2Index >= self.numTeams-1:  # Checks if we've done every combo of blue alliance, and resets if so.
            self.blue2Index = 0
            self.blue3Index = 1
            self.count += 1
            self.updatePerformances()
            self.output()

        self.checkDuplicates()

        return [self.targetTeam, self.teams[self.blue2Index], self.teams[self.blue3Index]]

    def checkDuplicates(self):
        if self.blue2Index == self.targetTeamIndex:  # These ifs are to check for duplicate teams.
            self.blue2Index += 1
        elif self.blue3Index == self.targetTeamIndex:
            self.blue3Index += 1
        if self.blue2Index == self.blue3Index:  # ensures that 2 and 3 are not the same team
            self.blue3Index += 1
            self.checkDuplicates()
    #  Simple helper function to combine arrays for the alliances.
    @staticmethod
    def combineAlliances(blue, red):
        blue.append(red[0])
        blue.append(red[1])
        blue.append(red[2])
        return blue

    #  This is called in __init__, just to reformat the data into a better format.
    def formatTeamData(self, teamData):
        # print(teamData)
        # print(self.teams)
        fullDataFrame = pd.DataFrame()
        for team in self.teams:
            tempData = teamData
            for count, value in enumerate(teamData['team_key']):
                if value == team:
                    tempData = teamData['scores'][count]
                    break
            avgRankPoints = tempData[0]
            autonScore = tempData[1]  # API returns an array like [ranking score, auto, end game, teleop]
            teleopScore = tempData[3]  # for some reason endgame isn't at end of array, so that's why numbers are weird
            endgameScore = tempData[2]
            eventRank = self.collector.getTeamEventRank(teamKey=team)
            seriesData = [autonScore, teleopScore, endgameScore, avgRankPoints, eventRank]
            finalSeries = pd.Series(data=seriesData, index=["autonScore", "teleopScore", "endgameScore",
                                                            "avgRankPoints", "eventRank"])
            # print("Calculated Team Data for: " + str(teamKey))
            fullDataFrame[team] = finalSeries
        return fullDataFrame.transpose()

    #  This gets data for each time in a given set of alliances and puts into a single array, to be fed into the model.
    def getAlliancesData(self, alliances):
        blueAlliance = []
        redAlliance = []
        for count, team in enumerate(alliances):
            # print("COUNT: " + str(count))
            # print("TEAM: " + team)
            temp = self.teamData.xs(key=team, axis=0)
            temp = np.array(temp)

            if count <= 2:
                blueAlliance.append(temp)
            else:
                redAlliance.append(temp)

        allianceData = [blueAlliance, redAlliance]
        alliancesData = np.array(allianceData)
        allianceData = np.reshape(allianceData, newshape=(-1, 2, 15))
        return allianceData

    def updateScores(self, score):
        self.scores[self.blue2Index].append(score)
        self.scores[self.blue3Index].append(score)

    # Updates the performance array, by using a rolling average of the scores teams achieved.
    def updatePerformances(self):
        for index, team in enumerate(self.scores):
            if index == self.targetTeamIndex:  # Don't care about performance of target Team.
                continue
            avg = 0
            for score in team:
                avg += score
            avg /= len(team)
            self.performance[index] = avg

    def output(self):
        print(self.performance)
        print(np.argmax(self.performance))
        bestIndex = np.argmax(self.performance)
        bestTeam = self.teams[bestIndex]
        print(bestTeam)
        return bestTeam

    def getTeamIndex(self, targetTeam):
        for id, team in enumerate(self.teams):
            if team == targetTeam:
                return id


def getTeams(event):
    collector = DataCollection.DataCollectionAnalysis(event, True)
    # print(np.array(collector.getAccTeamData()['team_key']))
    return [np.array(collector.getAccTeamData()['team_key']), collector]


def start(collector, team):
    # event = input("Please enter the official name of the event")
    # event = "FIM District Milford Event"
    # team = input("Please enter your team number.")
    # team = "1076"
    # team = "67"

    with open("/savedModels/best_model/hyperparameters.txt", 'r') as reader:
        parameters = reader.readline()
    buildStr = ""
    converted = []
    for element in parameters:  # Reads hyper parameters from text file and puts them into array.
        if element == ',':
            try:
                converted.append(float(buildStr))
                continue
            except:
                if buildStr == "True":
                    converted.append(True)
                elif buildStr == "False":
                    converted.append(False)
                continue
            finally:
                buildStr = ""
        buildStr += element  #

    # collector = DataCollectionAnalysis(event, True)
    creator = ModelCreator(converted[3], converted[4], converted[5], converted[6], converted[7], converted[8], converted[9])
    model = creator.makeModelWandB()
    model.load_weights("/savedModels/best_model/cp.ckpt").expect_partial()
    solver = EventSolve(model=model, dataCollector=collector, teamData=collector.getAccTeamData(), targetTeam=team)
    solver.search()


# start()
# getTeams("FIM District Milford Event")

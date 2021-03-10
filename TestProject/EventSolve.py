import random
from TestProject.DataCollection import DataCollectionAnalysis
from TestProject.ModelCreator import ModelCreator
import pandas as pd
import numpy as np

class EventSolve():
    def __init__(self, model, dataCollector, teamData, targetTeam):
        self.model = model
        self.collector = dataCollector
        self.teamData = teamData
        self.teams = teamData['team_key']
        self.numTeams = len(self.teams.index)
        self.teamData = self.formatTeamData(teamData)
        self.targetTeam = "frc" + targetTeam
        self.blue2Index = 0
        self.blue3Index = 0

    def search(self):
        while True:
            red = self.genRedAlliance()
            blue = self.genBlueAlliance()
            alliances = self.combineAlliances(blue, red)
            print(alliances)
            allianceData = self.getAlliancesData(alliances)
            print(model.predict(allianceData))
            break

    def genRedAlliance(self):
        numTeams = len(self.teams)
        red1 = random.randint(0, numTeams)
        red2 = random.randint(0, numTeams)
        red3 = random.randint(0, numTeams)
        if (red1 == red2) or (red1 == red3) or (red2 == red3):
            return self.genRedAlliance()
        red = [self.teams[red1], self.teams[red2], self.teams[red3]]
        return red

    def genBlueAlliance(self):
        self.blue3Index += 1
        if self.blue3Index >= self.numTeams:
            self.blue3Index = 0
            self.blue2Index += 1
        if self.blue2Index == self.blue3Index:
            self.blue3Index += 1

        if self.teams[self.blue2Index] == self.targetTeam:
            self.blue2Index += 1
        elif self.teams[self.blue3Index] == self.targetTeam:
            self.blue3Index += 1
        return [self.targetTeam, self.teams[self.blue2Index], self.teams[self.blue3Index]]

    def combineAlliances(self, blue, red):
        blue.append(red[0])
        blue.append(red[1])
        blue.append(red[2])
        return blue

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


# event = input("Please enter the official name of the event")
event = "FIM District Milford Event"
# team = input("Please enter your team number.")
team = "1076"

with open("savedModels/best_model/hyperparameters.txt", 'r') as reader:
    parameters = reader.readline()
buildStr = ""
converted = []
for element in parameters:
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
    buildStr += element

collector = DataCollectionAnalysis(event, True)
creator = ModelCreator(converted[3], converted[4], converted[5], converted[6], converted[7], converted[8], converted[9])
model = creator.makeModelWandB()
model.load_weights("savedModels/best_model/cp.ckpt")
solver = EventSolve(model=model, dataCollector=collector, teamData=collector.getAccTeamData(), targetTeam=team)
solver.search()

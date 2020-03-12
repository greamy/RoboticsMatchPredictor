import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import requests


class DataCollectionAnalysis():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    def __init__(self):
        # Needed for API Security
        self.headers = {'X-TBA-Auth-Key': 'uGgrrwF5M7RwNn2JmRr9UHFWw9gkYPevgzzWhF8VLequIboEbd5zcUvmPc800uHB'}
        # Base TBA API Link, so I don't have to retype it over and over
        self.link = "https://www.thebluealliance.com/api/v3/"

    def getTeamKey(self, teamNumber):
        # TBA Has a separate team key for every team. This gets it for you based on team number you input
        foundKey = False
        for i in range(0, 100):  # Only searches first 100 pages of the API Json files
            if foundKey:  # stops looping when it finds the right team
                break

            # This Gets JSon file into a string.
            json = requests.get(url=(self.link + "teams/2020/" + str(i)), headers=self.headers)
            if json.status_code != 200:
                # 200 is the Good return value. If We stop connecting to API correctly, stop looping.
                break

            # Converts Json String into Pandas DataFrame Object.
            data = pd.read_json(path_or_buf=json.text, typ="frame", convert_dates=False)
            for q in range(0, len(json.json())):
                # Checks DataFrame Object for the team number we are looking for.
                if data.at[q, "team_number"] == teamNumber:
                    key = data.at[q, "key"]
                    foundKey = True
                    break

        if foundKey:
            return key
        else:
            return -1  # If the function failed for some reason, it returns -1

    def getMatchKey(self, teamKey, matchNumber):  # Every Match Has its own unique Key, this finds it for you
        matchKeyJson = requests.get(url=(self.link + "team/" + str(teamKey) + "/matches/2020/keys"),
                                    headers=self.headers)
        matchKeyData = pd.read_json(path_or_buf=matchKeyJson.text, typ="series", convert_dates=False)
        try:
            matchExampleKey = matchKeyData[matchNumber]
        except IndexError:
            return
        return matchExampleKey

    def getTeamStats(self, numOfMatches, teamKey):
        # This function returns as many sample matches you need from a team, all in one data frame!
        # It also puts the score breakdown Stats into its own DataFrame, stored within the outer one.
        matchData = pd.DataFrame()
        print(numOfMatches)
        for i in range(0, numOfMatches):
            try:
                matchJson = requests.get(url=(self.link + "match/" + str(self.getMatchKey(teamKey, i))),
                                         headers=self.headers)
                matchData["Match #" + str(i + 1)] = pd.read_json(path_or_buf=matchJson.text, typ="series",
                                                                 convert_dates=False)
            except KeyError:
                print("Error occured. Using existing data")
                numOfMatches = i
                break

        # Creating Inner Data Frames for the data with multiple stats inside.
        matchData = matchData.transpose()
        for q in range(1, numOfMatches+1):
            tempAlliances = pd.json_normalize(data=matchData.at[("Match #" + str(q)), "alliances"])
            tempScoreBreakdown = pd.json_normalize(data=matchData.at[("Match #" + str(q)), "score_breakdown"])
            matchData.at[("Match #" + str(q)), "alliances"] = tempAlliances
            matchData.at[("Match #" + str(q)), "score_breakdown"] = tempScoreBreakdown

        return matchData

    @staticmethod
    def getTeamAllianceInfo(match, teamKey):
        allianceBreakdown = match["alliances"]

        blueAllianceTeamKeys = allianceBreakdown['blue.team_keys']
        redAllianceTeamKeys = allianceBreakdown['red.team_keys']
        for q in range(0, len(blueAllianceTeamKeys.get(0))):
            if teamKey == blueAllianceTeamKeys.get(0)[q]:
                return {"allianceColor": "blue", "teamAllianceNum": q+1}
        for j in range(0, len(redAllianceTeamKeys.get(0))):
            if teamKey == redAllianceTeamKeys.get(0)[j]:
                return {"allianceColor": "red", "teamAllianceNum": j+1}

    def calculateAutonScore(self, teamObject, teamKey):
        scoreBreakdown = teamObject["score_breakdown"]

        bottomGoals = 0
        upperGoals = 0
        innerGoals = 0
        initiationLineCt = 0
        numOfMatches = len(scoreBreakdown.index)
        for i in range(0, numOfMatches):
            allianceInfo = self.getTeamAllianceInfo(match=teamObject.xs("Match #" + str(i+1), axis=0), teamKey=teamKey)
            alliance = allianceInfo["allianceColor"]
            teamAllianceNumber = allianceInfo["teamAllianceNum"]
            match = scoreBreakdown["Match #" + str(i+1)]

            bottomGoals += match[str(alliance) + ".autoCellsBottom"][0]
            upperGoals += match[str(alliance) + ".autoCellsOuter"][0]
            innerGoals += match[str(alliance) + ".autoCellsInner"][0]
            initiationLine = match[str(alliance) + ".initLineRobot" + str(teamAllianceNumber)][0]
            if initiationLine == "Exited":
                initiationLineCt += 1

        bottomGoals = (float(bottomGoals)/float(numOfMatches))*2.
        upperGoals = (float(upperGoals) / float(numOfMatches))*4.
        innerGoals = (float(innerGoals) / float(numOfMatches))*6.

        # Multiply the point value of leaving the initiation line by how often they do it.
        initiationLineScore = (float(initiationLineCt)/float(numOfMatches))*5.
        return bottomGoals+upperGoals+innerGoals+initiationLineScore


    def calculateTeleopScore(self, teamObject, teamKey):
        # TO-DO: Add control panel and stage stuff and subtract foul points
        scoreBreakdown = teamObject["score_breakdown"]

        bottomGoals = 0.
        upperGoals = 0.
        innerGoals = 0.

        stage2Ct = 0
        stage3Ct = 0
        numOfMatches = len(scoreBreakdown.index)
        for i in range(0, numOfMatches):
            alliance = self.getTeamAllianceInfo(match=teamObject.xs("Match #" + str(i+1), axis=0), teamKey=teamKey)["allianceColor"]
            match = scoreBreakdown["Match #" + str(i+1)]

            # Gets the stats I want from the teleop portion of the match
            # For some reason, it returns as a series, so I'm getting the first (and only) data point of that series.
            bottomGoals += match[str(alliance) + ".teleopCellsBottom"][0]
            upperGoals += match[str(alliance) + ".teleopCellsOuter"][0]
            innerGoals += match[str(alliance) + ".teleopCellsInner"][0]

            # returns true/false values. We don't need stage 1 because that will show in the points from balls.
            stage2 = match[str(alliance) + ".stage2Activated"][0]
            stage3 = match[str(alliance) + ".stage3Activated"][0]

            if stage2:
                stage2Ct += 1
            if stage3:
                stage3Ct += 1

        # Find average amount of balls scored in each goal, and multiply by their point value.
        bottomGoals = bottomGoals/numOfMatches
        upperGoals = (upperGoals/numOfMatches)*2.
        innerGoals = (innerGoals/numOfMatches)*3.

        # Multiply the point value of each stage by how often they accomplish it.
        stage2Score = (stage2Ct / numOfMatches)*10.
        stage3Score = (stage3Ct / numOfMatches)*20.

        teleopScore = (bottomGoals+upperGoals+innerGoals+stage2Score+stage3Score)
        return teleopScore

    def calculateEndgameScore(self, teamObject, teamKey):
        scoreBreakdown = teamObject["score_breakdown"]
        parkedCount = 0
        hangingCount = 0
        levelCount = 0

        numOfMatches = len(scoreBreakdown.index)
        for i in range(0, numOfMatches):
            match = scoreBreakdown["Match #" + str(i+1)]
            allianceInfo = self.getTeamAllianceInfo(match=teamObject.xs("Match #" + str(i+1), axis=0), teamKey=teamKey)
            alliance = allianceInfo["allianceColor"]
            robotAllianceNum = allianceInfo["teamAllianceNum"]

            climbStatus = match[str(alliance) + ".endgameRobot" + str(robotAllianceNum)][0]
            levelStatus = match[str(alliance) + ".endgameRungIsLevel"][0]

            if climbStatus == "Park":
                parkedCount += 1
            elif climbStatus == "Hang":
                hangingCount += 1
            if levelStatus == "IsLevel":
                levelCount += 1

        percentParked = (float(parkedCount)/float(numOfMatches))
        percentHanging = (float(hangingCount)/float(numOfMatches))
        percentLevel = (float(levelCount)/float(numOfMatches))
        endgameScore = ((5.*percentParked) + (25.*percentHanging) + (15.*percentLevel))
        return endgameScore

    def getEventKey(self, eventName):
        eventKey = ""
        eventsJson = requests.get(url=(self.link + "events/" + "2020"), headers=self.headers)
        eventsData = pd.read_json(path_or_buf=eventsJson.text, typ="frame", convert_dates=False)
        # numOfEvents = eventsData.xs()


test = DataCollectionAnalysis()
test.getEventKey(eventName="FIM District Milford Event")
# teamKey = test.getTeamKey(teamNumber=67)
# teamStats = test.getTeamStats(numOfMatches=100, teamKey=teamKey)
# print(str(test.calculateAutonScore(teamObject=teamStats, teamKey=teamKey)) + ": Auton Score")
# print(str(test.calculateTeleopScore(teamObject=teamStats, teamKey=teamKey)) + ": TeleopScore")
# print(str(test.calculateEndgameScore(teamObject=teamStats, teamKey=teamKey)) + ": Endgame Score")

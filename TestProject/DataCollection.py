import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import requests


class DataCollectionAnalysis():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    def __init__(self, event):
        # Needed for API Security
        self.headers = {'X-TBA-Auth-Key': 'uGgrrwF5M7RwNn2JmRr9UHFWw9gkYPevgzzWhF8VLequIboEbd5zcUvmPc800uHB'}
        # Base TBA API Link, so I don't have to retype it over and over
        self.link = "https://www.thebluealliance.com/api/v3/"
        self.eventName = event
        self.eventKey = self.getEventKey()

    def getTeamKey(self, teamNumber):

        key = "frc" + str(teamNumber)
        return key
        # TBA Has a separate team key for every team. This gets it for you based on team number you input
        # foundKey = False
        # for i in range(0, 100):  # Only searches first 100 pages of the API Json files
        #     if foundKey:  # stops looping when it finds the right team
        #         break
        #
        #     # This Gets JSon file into a string.
        #     json = requests.get(url=(self.link + "teams/2020/" + str(i)), headers=self.headers)
        #     if json.status_code != 200:
        #         # 200 is the Good return value. If We stop connecting to API correctly, stop looping.
        #         break
        #
        #     # Converts Json String into Pandas DataFrame Object.
        #     data = pd.read_json(path_or_buf=json.text, typ="frame", convert_dates=False)
        #     for q in range(0, len(json.json())):
        #         # Checks DataFrame Object for the team number we are looking for.
        #         if data.at[q, "team_number"] == teamNumber:
        #             key = data.at[q, "key"]
        #             foundKey = True
        #             break
        # if foundKey:
        #     return key
        # else:
        #     return -1  # If the function failed for some reason, it returns -1

    def getMatchKey(self, teamKey, matchNumber):  # Every Match Has its own unique Key, this finds it for you
        matchKeyJson = requests.get(url=(self.link + "team/" + str(teamKey) + "/matches/2020/keys"),
                                    headers=self.headers)
        if matchKeyJson.status_code == 200:
            matchKeyData = pd.read_json(path_or_buf=matchKeyJson.text, typ="series", convert_dates=False)
        else:
            return -1
        try:
            matchExampleKey = matchKeyData[matchNumber]
        except IndexError:
            return -1
        return matchExampleKey

    def getTeamStats(self, teamKey, eventKey):
        # This function returns as many sample matches you need from a team, all in one data frame!
        # It also puts the score breakdown Stats into its own DataFrame, stored within the outer one.

        matchJson = requests.get(url=(self.link + "team/" + str(teamKey) + "/event/" + str(eventKey) + "/matches"), headers=self.headers)
        #print("Team stats request status code: " + str(matchJson.status_code))
        matchData = pd.read_json(path_or_buf=matchJson.text, typ="frame", convert_dates=False)
        numOfMatches = len(matchData.index)
        if numOfMatches == 0:
            return -1

        # # Creating Inner Data Frames for the data with multiple stats inside.
        # matchData = matchData.transpose()
        for q in range(0, numOfMatches):
            tempAlliances = pd.json_normalize(data=matchData.at[q, "alliances"])
            tempScoreBreakdown = pd.json_normalize(data=matchData.at[q, "score_breakdown"])
            matchData.at[q, "alliances"] = tempAlliances
            matchData.at[q, "score_breakdown"] = tempScoreBreakdown

        return matchData

    @staticmethod
    def getTeamAllianceInfo(match, teamKey=-1):
        allianceBreakdown = match["alliances"]
        blueAllianceTeamKeys = allianceBreakdown['blue.team_keys']
        redAllianceTeamKeys = allianceBreakdown['red.team_keys']
        if teamKey == -1:
            return {"blueAllianceKeys": blueAllianceTeamKeys[0], "redAllianceKeys": redAllianceTeamKeys[0]}
        else:
            for q in range(0, len(blueAllianceTeamKeys.get(0))):
                if teamKey == blueAllianceTeamKeys.get(0)[q]:
                    return {"allianceColor": "blue", "teamAllianceNum": q + 1}
            for j in range(0, len(redAllianceTeamKeys.get(0))):
                if teamKey == redAllianceTeamKeys.get(0)[j]:
                    return {"allianceColor": "red", "teamAllianceNum": j + 1}

    def calculateAutonScore(self, teamObject, teamKey):
        scoreBreakdown = teamObject.xs(key="score_breakdown", axis=1)

        bottomGoals = 0
        upperGoals = 0
        innerGoals = 0
        initiationLineCt = 0
        numOfMatches = len(scoreBreakdown.index)
        for i in range(0, numOfMatches):
            allianceInfo = self.getTeamAllianceInfo(match=teamObject.xs(key=i, axis=0), teamKey=teamKey)
            alliance = allianceInfo["allianceColor"]
            teamAllianceNumber = allianceInfo["teamAllianceNum"]
            match = scoreBreakdown[i]

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
            alliance = self.getTeamAllianceInfo(match=teamObject.xs(key=i, axis=0), teamKey=teamKey)["allianceColor"]
            match = scoreBreakdown[i]

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
            match = scoreBreakdown[i]
            allianceInfo = self.getTeamAllianceInfo(match=teamObject.xs(key=i, axis=0), teamKey=teamKey)
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

    def getEventKey(self):
        foundKey = False
        eventKey = ""
        eventsJson = requests.get(url=(self.link + "events/" + "2020"), headers=self.headers)
        eventsData = pd.read_json(path_or_buf=eventsJson.text, typ="frame", convert_dates=False)
        numOfEvents = len(eventsData.index)
        for i in range(0, numOfEvents):

            if eventsData.xs(key="name", axis=1).iat[i] == self.eventName:
                eventKey = eventsData.xs(key="key", axis=1).iat[i]
                foundKey = True
                break
        if foundKey:
            return eventKey
        else:
            return -1

    def getEventTeams(self, eventKey):
        teamsJson = requests.get(url=(self.link + "event/" + str(eventKey) + "/teams"), headers=self.headers)
        teamsData = pd.read_json(path_or_buf=teamsJson.text, typ="frame", convert_dates=False)
        numOfTeams = len(teamsData.index)
        for r in range(0, numOfTeams):
            tempNum = teamsData.xs(key="team_number", axis=1).iat[r]
            teamsData = teamsData.rename(index={r: tempNum})
            # print("Got team: " + str(tempNum))
        return teamsData

    def calculateTeamData(self, teamsFromEvent):
        numOfTeams = len(teamsFromEvent.index)
        fullDataFrame = pd.DataFrame()
        for q in range(0, numOfTeams):
            teamKey = self.getTeamKey(teamsFromEvent.index[q])
            teamObject = self.getTeamStats(teamKey=teamKey, eventKey=self.eventKey)
            if type(teamObject) == int:
                continue
            teamAutonScore = self.calculateAutonScore(teamObject=teamObject, teamKey=teamKey)
            teamTeleopScore = self.calculateTeleopScore(teamObject=teamObject, teamKey=teamKey)
            teamEndgameScore = self.calculateEndgameScore(teamObject=teamObject, teamKey=teamKey)
            seriesData = [teamAutonScore, teamTeleopScore, teamEndgameScore]
            finalSeries = pd.Series(data=seriesData, index=["autonScore", "teleopScore", "endgameScore"])
            # print("Calculated Team Data for: " + str(teamKey))
            fullDataFrame[teamKey] = finalSeries
        return fullDataFrame.transpose()

    def getEventMatches(self, eventKey):
        matchJson = requests.get(url=(self.link + "event/" + str(eventKey) + "/matches"), headers=self.headers)
        matchData = pd.read_json(path_or_buf=matchJson.text, typ="frame", convert_dates=False)
        return matchData

    def eventMatchesFormatted(self):
        allMatches = self.getEventMatches(self.eventKey)
        teamsFromEvent = self.getEventTeams(eventKey=self.eventKey)
        teamData = self.calculateTeamData(teamsFromEvent=teamsFromEvent)

        numOfMatches = len(allMatches.index)
        eventMatches = pd.DataFrame(index=range(0, numOfMatches), columns=["Blue Alliance", "Red Alliance", "winning_alliance"])
        for i in range(0, numOfMatches):
            currentMatch = allMatches.loc[i]

            tempAlliances = pd.json_normalize(data=currentMatch["alliances"])
            tempScoreBreakdown = pd.json_normalize(data=currentMatch["score_breakdown"])
            currentMatch.at["alliances"] = tempAlliances
            currentMatch.at["score_breakdown"] = tempScoreBreakdown
            allianceInfo = self.getTeamAllianceInfo(currentMatch)

            blueAllianceKeys = allianceInfo["blueAllianceKeys"]
            blueAllianceStats = pd.DataFrame()
            redAllianceKeys = allianceInfo["redAllianceKeys"]
            redAllianceStats = pd.DataFrame()
            for b in range(0, 3):
                teamKey = blueAllianceKeys[b]
                tempTeamData = teamData.xs(key=teamKey, axis=0)
                blueAllianceStats[teamKey] = tempTeamData
            for r in range(0, 3):
                teamKey = redAllianceKeys[r]
                tempTeamData = teamData.xs(key=teamKey, axis=0)
                redAllianceStats[teamKey] = tempTeamData
            winningAlliance = currentMatch["winning_alliance"]

            eventMatches.at[i, "Blue Alliance"] = blueAllianceStats
            eventMatches.at[i, "Red Alliance"] = redAllianceStats
            eventMatches.at[i, "winning_alliance"] = winningAlliance

        # fileTest = open("eventMatchesFormatted.txt", "w+")
        # for i in range(0, numOfMatches):
        #     fileTest.write("\n" + str(eventMatches.at[i, "Blue Alliance"].to_json))
        # for i in range(0, numOfMatches):
        #     fileTest.write("\n" + str(eventMatches.at[i, "Red Alliance"].to_json))
        #
        # fileTest.write("\n" + str(eventMatches["winning_alliance"].to_json()))
        # fileTest.close()

        return eventMatches

    # def eventMatchesFormattedFromFile(self):
    #     eventMatches = pd.read_json(path_or_buf="eventMatchesFormatted.json")
    #     numOfMatches = len(eventMatches.index)
    #     for i in range(0, numOfMatches):
    #         temp = open("temp.txt", "w+")
    #         blueAllianceJson = str(eventMatches.at[i, "Blue Alliance"])
    #         print(blueAllianceJson)
    #         temp.write(blueAllianceJson)
    #         temp.close()
    #
    #         tempRead = open("temp.txt", "r")
    #         blueAllianceStats = pd.read_json(tempRead.read(), typ="series")
    #         print(blueAllianceStats)
    #         tempRead.close()
    #     print(eventMatches)
    #     return eventMatches

    def numbersOnly(self, eventMatches):
        numOfMatches = len(eventMatches.index)
        # Creates Array with 0's, with enough spaces to hole each score.
        numberList = [[[0 for x in range(9)], [0 for x in range(9)]] for x in range(numOfMatches)]
        for i in range(numOfMatches):
            xsize = len(eventMatches.at[i, "Blue Alliance"].index)
            ysize = len(eventMatches.at[i, "Blue Alliance"].columns)
            count = 0
            for x in range(xsize):
                for y in range(ysize):
                    numberList[i][0][count] = eventMatches.at[i, "Blue Alliance"].iat[x, y]
                    numberList[i][1][count] = eventMatches.at[i, "Red Alliance"].iat[x, y]
                    count += 1
        return numberList

    @staticmethod
    def labelsToNums(labels):
        for i in range(len(labels)):
            if labels[i] == 'red':
                labels[i] = 0
            elif labels[i] == 'blue':
                labels[i] = 1
        return labels

    def cleanData(self, data, labels):
        countDeleted = 0
        for i in range(len(labels)):
            changeNum = (i-1)-countDeleted
            if labels[changeNum] != 'red' and labels[changeNum] != 'blue':
                labels = np.delete(labels, changeNum)
                data = np.delete(data, changeNum, 0)
                countDeleted += 1
        return [data, labels]

    def getDataAndLabels(self):
        data = self.eventMatchesFormatted()
        labels = data.pop("winning_alliance").to_numpy()
        data = self.numbersOnly(data)
        data = np.array(data)
        clean = self.cleanData(data, labels)
        data = clean[0]
        labels = clean[1]
        labels = self.labelsToNums(labels)
        return [data, labels]


# correctCount = 0
# for s in range(0, len(eventMatches.index)):
#     currentBlue = eventMatches.at[s, "Blue Alliance"]
#     avgBlueAuton = (currentBlue.iat[0, 0] + currentBlue.iat[0, 1] + currentBlue.iat[0, 2])/3
#     avgBlueTeleop = (currentBlue.iat[1, 0] + currentBlue.iat[1, 1] + currentBlue.iat[1, 2])/3
#     avgBlueEndgame = (currentBlue.iat[2, 0] + currentBlue.iat[2, 1] + currentBlue.iat[2, 2])/3
#     blueScore = avgBlueAuton + avgBlueTeleop + avgBlueEndgame
#
#     currentRed = eventMatches.at[s, "Red Alliance"]
#     avgRedAuton = (currentRed.iat[0, 0] + currentRed.iat[0, 1] + currentRed.iat[0, 2])/3
#     avgRedTeleop = (currentRed.iat[1, 0] + currentRed.iat[1, 1] + currentRed.iat[1, 2])/3
#     avgRedEndgame = (currentRed.iat[2, 0] + currentRed.iat[2, 1] + currentRed.iat[2, 2])/3
#     redScore = avgRedAuton + avgRedTeleop + avgBlueEndgame
#
#     winning_alliance = labels[s]
#     if blueScore > redScore and winning_alliance == "blue":
#         correctCount += 1
#     elif redScore > blueScore and winning_alliance == "red":
#         correctCount += 1
#
# correctPercent = correctCount/len(eventMatches.index)
# print(correctPercent)

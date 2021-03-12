import numpy as np
import pandas as pd
import requests


class DataCollectionAnalysis():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    def __init__(self, event, scorePredict):
        self.scorePredict = scorePredict
        # Needed for API Security
        self.headers = {'X-TBA-Auth-Key': 'uGgrrwF5M7RwNn2JmRr9UHFWw9gkYPevgzzWhF8VLequIboEbd5zcUvmPc800uHB'}
        # Base TBA API Link, so I don't have to retype it over and over
        self.link = "https://www.thebluealliance.com/api/v3/"
        self.eventName = event
        self.eventKey = self.getEventKey()
        self.eventRankings = self.getEventRankings()
        self.allData = self.getAllData()

    # Function returns the 'key' for a team number. There is code to pull it from TheBlueAllianceAPI, but each key
    # is just the number with frc in front of it, simplifying like this vastly reduces time.
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

    # Function returns the key of a specific match within the event.
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
        # print("Team stats request status code: " + str(matchJson.status_code))
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

    # This function returns which alliance a given team is on within a given match, if given teamKey
    # If not given the teamKey, returns the teams on each alliance within the match.
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

    # Simple method to pull array of scores (auton, teleop, endgame) from a match.
    @staticmethod
    def getAllianceScores(match):
        scores = np.array([match['alliances']['blue.score'][0], match['alliances']['red.score'][0]])
        return scores

    # Takes a team and returns their autonScore, which is an estimation of how good they are at auton.
    # It is only an estimation because API only has scores for entire alliance, bad teams are shown better, vice versa
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

    # Takes a team and returns their teleop Score, which is an estimation of how good they are at teleop.
    # It is only an estimation because API only has scores for entire alliance, bad teams are shown better, vice versa
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

    # Takes a team and returns their endgame Score, which is an estimation of how good they are at endgame.
    # It is only an estimation because API only has scores for entire alliance, bad teams are shown better, vice versa
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

    # Collects the data from the above function and gets the scores and ranks for each team from the event.
    def calculateTeamData(self, teamsFromEvent):
        numOfTeams = len(teamsFromEvent.index)
        fullDataFrame = pd.DataFrame()
        for q in range(0, numOfTeams):
            teamKey = self.getTeamKey(teamsFromEvent.index[q])
            teamObject = self.getTeamStats(teamKey=teamKey, eventKey=self.eventKey)
            if type(teamObject) == int:
                print("team " + str(teamKey) + " had an error")
                continue
            autonScore = self.calculateAutonScore(teamObject=teamObject, teamKey=teamKey)
            teleopScore = self.calculateTeleopScore(teamObject=teamObject, teamKey=teamKey)
            endgameScore = self.calculateEndgameScore(teamObject=teamObject, teamKey=teamKey)
            eventRank = self.getTeamEventRank(teamKey=teamKey)
            seriesData = [autonScore, teleopScore, endgameScore, eventRank]
            finalSeries = pd.Series(data=seriesData, index=["autonScore", "teleopScore", "endgameScore", "eventRank"])
            # print("Calculated Team Data for: " + str(teamKey))
            fullDataFrame[teamKey] = finalSeries
        return fullDataFrame.transpose()

    def getAccTeamData(self):
        rankingsJson = requests.get(url=(self.link + "event/" + self.eventKey + "/rankings"), headers=self.headers)
        allData = rankingsJson.json()  # response object has .json() function, which puts the plaintext into actual
        # python objects (lists, dicts, etc)
        allData = pd.DataFrame(data=allData['rankings'])
        filteredData = pd.DataFrame(data={'scores': allData['sort_orders'], 'team_key': allData['team_key']})
        return filteredData

    def calculateAccTeamData(self, teamsFromEvent):
        filteredData = self.getAccTeamData()

        numOfTeams = len(teamsFromEvent.index)
        fullDataFrame = pd.DataFrame()
        for q in range(0, numOfTeams):
            teamKey = self.getTeamKey(teamsFromEvent.index[q])
            tempData = filteredData
            for count, value in enumerate(filteredData['team_key']):
                if value == teamKey:
                    tempData = filteredData['scores'][count]
                    break
            avgRankPoints = tempData[0]
            autonScore = tempData[1]  # API returns an array like [ranking score, auto, end game, teleop]
            teleopScore = tempData[3]  # for some reason endgame isn't at end of array, so that's why numbers are weird
            endgameScore = tempData[2]
            eventRank = self.getTeamEventRank(teamKey=teamKey)
            seriesData = [autonScore, teleopScore, endgameScore, avgRankPoints, eventRank]
            finalSeries = pd.Series(data=seriesData, index=["autonScore", "teleopScore", "endgameScore",
                                                            "avgRankPoints", "eventRank"])
            # print("Calculated Team Data for: " + str(teamKey))
            fullDataFrame[teamKey] = finalSeries

        # print(fullDataFrame)
        return fullDataFrame.transpose()

    # def getDistrictKey(self, teamKey):
    #     districtsJson = requests.get(url=(self.link + "team/" + teamKey + "/districts"), headers=self.headers)
    #     districtsData = pd.read_json(path_or_buf=districtsJson.text, typ="frame", convert_dates=False)
    #     print(districtsData)
    #     return districtsData.iat[len(districtsData.index)-2, 2]
    #     # The -2 is to get 2019 district of given team (-3 would be 2018, etc.)
    #     # The , 2] is to get the district key, rather than name or some other info.

    # This returns a DataFrame (probably should be series) with each team at the index of the rank within the event.
    def getEventRankings(self):
        # print(type(self.link + "event/" + self.eventKey + "/rankings"))
        # print(type(self.headers))
        rankingsJson = requests.get(url=(str(self.link) + "event/" + str(self.eventKey) + "/rankings"), headers=self.headers)
        rankingsData = rankingsJson.json()  # response object has .json() function, which puts the plaintext into actual
        # python objects (lists, dicts, etc)
        print(rankingsData)
        rankingsData = pd.DataFrame(data=rankingsData['rankings'])
        # formatting dataFrame for easy iteration
        rankingsData = pd.DataFrame(data={'team_key': rankingsData['team_key']})
        return rankingsData

    # Returns the ranking of given team within the event, using output from above Function.
    def getTeamEventRank(self, teamKey):
        # loops through the rankings for the event, and the index of each team within the list is their rank.
        for count, value in enumerate(self.eventRankings['team_key']):
            if value == teamKey:
                return count
        return -1

    # Returns the key for an event, for use in API calls.
    # This is called in __init__ and is an instance variable.
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

    # Returns a DataFrame of all the teams within an event, and data about each team given by the API
    def getEventTeams(self):
        teamsJson = requests.get(url=(self.link + "event/" + str(self.eventKey) + "/teams"), headers=self.headers)
        teamsData = pd.read_json(path_or_buf=teamsJson.text, typ="frame", convert_dates=False)
        numOfTeams = len(teamsData.index)
        for r in range(0, numOfTeams):
            tempNum = teamsData.xs(key="team_number", axis=1).iat[r]
            teamsData = teamsData.rename(index={r: tempNum})
            # if type(teamsData.loc[tempNum]["team_number"]) == pd.core.series.Series:
            #     print(teamsData.at[tempNum, "team_number"][0])
            #     teamsData.at[tempNum, "team_number"] = teamsData.at[tempNum, "team_number"][0]
            # print("Got team: " + str(teamsData.at[tempNum, "team_number"]))
            # print(type(teamsData.at[tempNum, "team_number"]))

        return teamsData

    def getEventMatches(self, eventKey):
        matchJson = requests.get(url=(self.link + "event/" + str(eventKey) + "/matches"), headers=self.headers)
        matchData = pd.read_json(path_or_buf=matchJson.text, typ="frame", convert_dates=False)
        return matchData

    # this function gets all the matches of the event and brings it all into a numpy array for reading into AI.
    # This should be called for use of binary classification algorithm, classifying blue or red win.
    def eventMatchesFormatted(self):
        allMatches = self.getEventMatches(self.eventKey)
        teamsFromEvent = self.getEventTeams(eventKey=self.eventKey)
        teamData = self.calculateTeamData(teamsFromEvent=teamsFromEvent)

        numOfMatches = len(allMatches.index)
        eventMatches = pd.DataFrame(index=range(0, numOfMatches), columns=["Blue Alliance", "Red Alliance", "winning_alliance"])
        for i in range(numOfMatches):
            try:
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

            except IndexError:
                print("Index out of bounds, skipping to end")
                break
            # except Exception as e:
            #     print(e.__traceback__.print)
            #     print("Something went wrong with match #" + str(i - matchesRemoved) + " in " + str(self.eventName))
            #     print("Deleting match from data, and moving on.")
            #     allMatches.drop(i, axis=0)
            #     matchesRemoved += 1

        return eventMatches

    # this function gets all the matches of the event and brings it all into a numpy array for reading into AI.
    # This should be called for use of regression model, which should attempt to predict real scores for each match.
    def eventMatchesFormattedForScorePredict(self):
        allMatches = self.getEventMatches(self.eventKey)
        teamsFromEvent = self.getEventTeams()
        # teamData = self.calculateTeamData(teamsFromEvent=teamsFromEvent)
        teamData = self.calculateAccTeamData(teamsFromEvent=teamsFromEvent)
        # print(teamData)

        numOfMatches = len(allMatches.index)
        eventMatches = pd.DataFrame(index=range(0, numOfMatches), columns=["Blue Alliance", "Red Alliance", "scores"])
        for i in range(numOfMatches):
            try:
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

                # these for loops split the teamData into separate alliances and separate teams.
                for b in range(0, 3):
                    teamKey = blueAllianceKeys[b]
                    tempTeamData = teamData.xs(key=teamKey, axis=0)
                    blueAllianceStats[teamKey] = tempTeamData
                for r in range(0, 3):
                    teamKey = redAllianceKeys[r]
                    tempTeamData = teamData.xs(key=teamKey, axis=0)
                    redAllianceStats[teamKey] = tempTeamData

                # this is the actual scores for each match
                scores = self.getAllianceScores(match=currentMatch)

                eventMatches.at[i, "Blue Alliance"] = blueAllianceStats
                eventMatches.at[i, "Red Alliance"] = redAllianceStats
                eventMatches.at[i, "scores"] = scores
                # eventMatches.at[i, "eventRank"] =
                # test = self

            except IndexError:
                print("Index out of bounds, skipping to end")
                break
            # except Exception as e:
            #     print(e.__traceback__.print)
            #     print("Something went wrong with match #" + str(i - matchesRemoved) + " in " + str(self.eventName))
            #     print("Deleting match from data, and moving on.")
            #     allMatches.drop(i, axis=0)
            #     matchesRemoved += 1
        return eventMatches

    def numbersOnly(self, eventMatches):
        numOfMatches = len(eventMatches.index)
        # Creates Array with 0's, with enough spaces to hold each score.
        # shape=(12, 2, numMatches), 12 is because 4 stats x 3 teams (auton, teleop, endgame scores, and Event Rank),
        # 2 because 2 alliances, and then each match is a separate entry
        numberList = [[[0 for x in range(15)], [0 for x in range(15)]] for x in range(numOfMatches)]
        for i in range(numOfMatches):
            xsize = len(eventMatches.at[i, "Blue Alliance"].index)
            ysize = len(eventMatches.at[i, "Blue Alliance"].columns)
            count = 0
            for x in range(xsize):
                for y in range(ysize):
                    # i is the match we are on, 0 or 1 is blue or red alliance,
                    # count is the specific team we are on (blue/red, x3 teams each)
                    numberList[i][0][count] = eventMatches.at[i, "Blue Alliance"].iat[x, y]
                    numberList[i][1][count] = eventMatches.at[i, "Red Alliance"].iat[x, y]
                    count += 1
        return numberList

    # Converts winning alliance labels from 'blue' and 'red' to 0 and 1.
    @staticmethod
    def labelsToNums(labels):
        for i in range(len(labels)):
            if labels[i] == 'red':
                labels[i] = 0
            elif labels[i] == 'blue':
                labels[i] = 1
        return labels

    # checks for missing data, and clears that match from data and label array.
    @staticmethod
    def cleanData(self, data, labels):
        countDeleted = 0
        for i in range(len(labels)):
            changeNum = (i-1)-countDeleted
            if labels[changeNum] != 'red' and labels[changeNum] != 'blue':
                labels = np.delete(labels, changeNum)
                data = np.delete(data, changeNum, 0)
                countDeleted += 1
        return [data, labels]

    # gets both match data and label data in separate numpy arrays for use for binary classification algorithm
    # also does a lot of cleaning of data.
    # returns list of numpy arrays with equal length. (not equal shape) [numpyArray, numpyArray]
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

    # Same as above, except used for score predictions for use in a regression-based model.
    def getDataAndLabelsForScorePredict(self):
        data = self.eventMatchesFormattedForScorePredict()
        labels = data.pop("scores").to_numpy()
        data = self.numbersOnly(data)
        data = np.array(data)
        return [data, labels]

    def getAllData(self):
        if not self.scorePredict:
            full = self.getDataAndLabels()
        else:
            full = self.getDataAndLabelsForScorePredict()
        print(self.eventName + " EVENT RETRIEVED")
        return [full[0], full[1]]

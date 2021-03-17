class APIException(Exception):

    def __init__(self, reason, method):
        # Exception.__init__(Exception(reason))
        self.reason = reason
        self.method = method

    def __str__(self):
        return "REASON: " + self.reason + " during: " + str(self.method)

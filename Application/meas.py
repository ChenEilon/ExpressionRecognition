import time

class meas:
    def __init__(self, message="Time measured", average_on=1):
        self.time_a = 0
        self.time_b = 0
        self.times = []
        self.message = message
        self.average_on = average_on

    def start(self):
        self.time_a = time.time()

    def end(self):
        self.time_b = time.time()
        self.times.append(self.time_b - self.time_a)
        if len(self.times) >= self.average_on:
            print("{0}: {1:.2f}s".format(self.message, sum(self.times) / len(self.times)))
            self.average_on *= 2

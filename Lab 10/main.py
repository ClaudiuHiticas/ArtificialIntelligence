
class Humidity:
    def __init__(self, value):
        self.dry = 0
        self.normal = 0
        self.wet = 0
        self.fuzzy(value)

    def triangleFuzzy(self, value, l):
        res = max(0, min((value - l[0])/(l[1] - l[0]), 1, (l[2] - value)/(l[2] - l[1])))
        print("humidity:", res)
        return res

    def fuzzy(self, value):
        self.dry = self.triangleFuzzy(value, [-0.1, 0, 50])
        self.normal = self.triangleFuzzy(value, [-0.1, 50, 100.1])
        self.wet = self.triangleFuzzy(value, [50, 100, 100.1])


class Temperature:
    def __init__(self, value):
        self.veryCold = 0
        self.cold = 0
        self.normal = 0
        self.warm = 0
        self.hot = 0
        self.fuzzy(value)

    def fuzzy(self, value):
        self.veryCold = self.trapezoidFuzzy(value, [-31, -30, -20, 5])
        self.cold = self.triangleFuzzy(value, [5, 0, 10])
        self.normal = self.trapezoidFuzzy(value, [5, 10, 15, 20])
        self.warm = self.triangleFuzzy(value, [15, 20, 25])
        self.hot = self.trapezoidFuzzy(value, [25, 30, 35, 36])

    def triangleFuzzy(self, value, l):
        res =  max(0, min((value - l[0])/(l[1] - l[0]), 1, (l[2] - value)/(l[2] - l[1])))
        print("temp:",res)
        return res

    def trapezoidFuzzy(self, value, l):
        res =  max(0, min((value-l[0])/(l[1]-l[0]), 1, (l[3]-value)/(l[3]-l[2])))
        print("tempp:",res)
        return res


class Time:
    def __init__(self, temperature, humidity):
        self.short = max(min(humidity.wet, 1-temperature.hot), min(humidity.normal, temperature.veryCold))
        self.long = max(min(humidity.dry, 1-temperature.veryCold), min(humidity.normal, temperature.hot))
        self.medium = min(self.short, self.long)

    def defuzzy(self):
        res = 50 * (1 - self.short) + 50 * self.long
        f = open("description.txt", "a+")
        f.write(" \n defuzzyficate = " + str(res) + " = " + str( "50 * (1 - " + str(self.short) + ") + (50 * "+ str(self.long)+")"))
        print("time:", self.short, self.long, res)
        return res


class Problem:
    def __init__(self):
        self.temperatureList = [-11, 22, 18, -10, -6, 25]
        self.humidityList = [88, 55, 100, 0, 74, 22, 99]
        self.l = []
        self.computeTime()

    def computeTime(self):
        for i in range(0, len(self.temperatureList)):
            temperature = Temperature(self.temperatureList[i])
            humidity = Humidity(self.humidityList[i])
            time = Time(temperature, humidity)
            self.l.append(int(time.defuzzy()))

    def getTime(self):
        return self.l


app = Problem()
print(app.getTime())

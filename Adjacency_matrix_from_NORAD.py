# -*- coding: utf-8 -*-


from skyfield.api import load
# from pathos.multiprocessing import ProcessPool
import numpy as np
from numba import cuda
import math


class Connection:
    stop = None
    duration = 0

    def __init__(self, sat1, sat2, start):
        self.sat1 = sat1
        self.sat2 = sat2
        self.start = start

    def toString(self):
        return (str(self.sat1) + "\t\t" + str(self.sat2) + "\t\t" + str(self.start) + "\t\t" + str(
            self.stop) + "\t\t" + str(self.duration))


stations_url = 'http://celestrak.com/NORAD/elements/active.txt'
satellites = load.tle(stations_url)
satellite = satellites['COSMOS 2545 [GLONASS-M]']
TRANSMISSION_RANGE = 150
keys = []
sats = []
connections = []

current = 0
total = len(satellites.values());
print(satellite.epoch.utc_jpl())


def coordCalc(sat, t):
    return sat.at(t).position.km


@cuda.jit()
def getLocation(m, transRange, inRange):
    # Matrix index.
    x = cuda.threadIdx.x
    size = m.shape[0]
    # Skip threads outside the matrix.
    earthRad = 6371;
    maxRange = 5100

    if x >= size:
        return
    # Run the simulation.
    for i in range(x, 2627, 1024):
        for j in range(i + 1, size):
            dx = m[j][0] - m[i][0]
            dy = m[j][1] - m[i][1]
            dz = m[j][2] - m[i][2]
            dMag = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if (dMag > maxRange):
                inRange[i][j] = False
                continue
            if(dMag == 0):
                inRange[i][j] = True
                continue
            dx = dx / dMag
            dy = dy / dMag
            dz = dz / dMag
            b = m[i][0] * dx + m[i][1] * dy + m[i][2] * dz
            c = (m[i][0] ** 2 + m[i][1] ** 2 + m[i][2] ** 2) - earthRad ** 2
            h = b ** 2 - c;
            if (h < 0):
                inRange[i][j] = True  # no intersection
                continue
            h = math.sqrt(h)
            if (min(-b - h, -b + h) > dMag):
                inRange[i][j] = True
                continue
            inRange[i][j] = False
    return
    # inRange[i][j] = earthRad < (np.linalg.norm((np.cross(m[i]), np.negative(m[j]))))/np.linalg.norm(m[j]-m[i]))
    # inRange[i][j] = (math.sqrt((m[i][0]-m[j][0])**2 + (m[i][1]-m[j][1])**2 + (m[i][2]-m[j][2])**2) <= transRange)


#print(cuda.gpus[0].name)
total_sat=0
for satellite in satellites.values():
    if satellite.name in keys:
        continue
    keys.append(satellite.name)
    sats.append(satellite)
    total_sat=total_sat+1
    if total_sat==2000:
        break
print(total_sat)

    #if total_sat==100: #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
     #   break #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

ts = load.timescale()
t = ts.utc(2022, 1, 19, 0, 0, 0)

for sat in sats:
    if abs(t - sat.epoch) > 2:
        print("flag", sat.name) #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        sats.remove(sat)

inclinations = {}
for sat in sats:
    i = round(sat.model.inclo, 1)
    if i in inclinations:
        inclinations[i] = inclinations[i] + 1
    else:
        inclinations[i] = 1
for k in inclinations.keys():
    print(k, inclinations[k])

for inc in inclinations:
    print(inc)
print(len(sats))
m = np.array([sat.at(t).position.km for sat in sats])
inRange = np.zeros((len(m), len(m)))
lastRange = np.zeros((len(m), len(m)))
connectionStart = np.zeros((len(m), len(m)))
names = [sat.name for sat in sats]

# 16x16 threads per block.
bs = 1024
# Number of blocks in the grid.
bpg = int(np.ceil(len(sats) / bs))
# We prepare the GPU function.
getLoc = getLocation[(bpg), (bs)]

with open("key.txt", 'w') as file:
    for name in names:
        file.write(str(name))
        file.write('\n')

lent = []


for hour in range(1, 5):
    for minute in range(0, 60, 20):
             # for second in range(0, 60, 20):
        with open("E:\\Project\\Test\\output" + str(TRANSMISSION_RANGE) + "_" + str(hour) + "_" + str(minute) + ".txt", "w") as file:

            t = ts.utc(2022, 1, 19, hour, minute, 0)
            m = np.array([sat.at(t).position.km for sat in sats])
            getLoc(m, TRANSMISSION_RANGE, inRange)
            file.write("[")
            for i in range(len(inRange)):
                file.write("[")
                for j in range(len(inRange)):
                    file.write(str(int(inRange[i][j])))
                    if (j != len(inRange) - 1):
                        file.write(",")
                    # print(inRange[i][j])
                file.write("]")
                if (i != len(inRange) - 1):
                    file.write(",")
            file.write("]") # MY TEST
#### Data visualization

import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.ticker import MultipleLocator
import os 
import pandas as pd
from datetime import datetime

# create a new figure object 
fig = plt.figure()
# create subplots, must create at least one plot
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

# plot noise 
plt.plot(np.random.randn(50).cumsum(), 'k--')
# ax1 calls the first plot, histogram
_ = ax1.hist(np.random.randn(100), bins=20, color='y', alpha=0.3)
# scatter plot
ax2.scatter(np.arange(30), np.arange(30) + 3*np.random.randn(30))

# Various options include linestyle, color, and marker
plt.plot(xx, yy, linestyle='--', color='g', marker='o')

# Create figure and axes at the same time 
# this is just one plot but subplots(2,3) gives 2x3 grid of plots
fig, ax = plt.subplots()

my_age = np.arange(3, 45)
knees_hurt = 0.03 * np.arange(50) ** 2
all_ages = np.arange(50)

# two plots on the same subplot 
plt.plot(all_ages[4:46], my_age, marker='D', label='My age')
plt.plot(all_ages, knees_hurt, marker='o',
         label='How much my knees hurt')
         
# for the minor ticks, use no labels; default NullFormatter
# create grid
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)

plt.grid(b=True, which='major', color='0.65', linestyle='-')
# title 
plt.title('My knee pain and my age', fontsize=20)
# labels 
plt.xlabel(r'Age')
plt.ylabel(r'Pain units')
# axis length 
plt.xlim([-1.0, 50])
plt.ylim([-2.0, 80])
# legend
plt.legend(loc='upper left')

# mess with the xticks
xtickvals = [0, 8, 10, 20, 28, 30, 40, 41, 50]
xticklabs = [0, 'bt', 10, 20, 'mt', 30, 40, 'cr', 50]
plt.xticks(xtickvals, xticklabs, rotation='vertical')

#######################################
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('DataFiles/spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']
spx.plot(ax=ax, style='k-')

# choose points to label 
crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'), 
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    # annotates the data 
    ax.annotate(label, xy=(date, spx.asof(date) + 50),
                xytext=(date, spx.asof(date) + 200),
                arrowprops=dict(facecolor='black'),
                horizontalalignment='left', verticalalignment='top')

# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])


# ####### Save figures
# cur_path = os.path.split(os.path.abspath(__file__))[0]
# output_dir = os.path.join(cur_path, 'images')
# output_path = os.path.join(output_dur,'xyplot')

# plt.savefig(output_path)
# plt.close() # otherwise kill your memory

###################
#### HISTOGRAM ####
###################

# A histogram is an empirical estimator for the probability density function
from pandas import DataFrame, Series

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
pennies = pd.read_csv('DataFiles/penny.csv', names =['year', 'penny1', 'penny2'],
                      index_col='year', skiprows=[0, 1, 2, 3, 4, 5])
pennies_all = pennies.stack()
ax.hist(pennies_all,10) # ten is the number of bins, pennies_all is the array 
# number per bin, where cut bin, idk
n, bin_cuts, patches = plt.hist(pennies_all, 4, facecolor='green')
# transform into frequency
num_bins = 10
weights = (1 / pennies_all.shape[0]) * np.ones_like(pennies_all)
n, bin_cuts, patches = plt.hist(pennies_all, num_bins, weights=weights)
# edit the xticks
plt.xticks(np.round_(bin_cuts, 1))
plt.title('Histogram of penny thickness (mils): 1945-1989', fontsize=17)
plt.xlabel(r'Thickness (mils)')
plt.ylabel(r'Percent of observations in bin')

####### 3-D Histogram
geyser_dur = pd.read_csv('DataFiles/geyser.csv', names =['duration'])
dur_t = geyser_dur['duration'][:-1]
dur_tp1 = geyser_dur['duration'][1:]
# plan is to use the NumPy function histogram2d() 
# then port the objects into a 3D bar chart using matplotlib's bar3D() function.

from mpl_toolkits.mplot3d import Axes3D
iris_data = pd.read_csv(
    'DataFiles/iris.csv',
    names=['length_sep', 'width_sep', 'length_pet', 'width_pet', 'species'], skiprows=[0])
length_sep = iris_data['length_sep']
width_sep = iris_data['width_sep']
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
bin_num = int(9)
hist, xedges, yedges = np.histogram2d(dur_t, dur_tp1, bins=bin_num)
hist = hist / hist.sum()
x_midp = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
y_midp = yedges[:-1] + 0.5 * (yedges[1] - yedges[0])
elements = (len(xedges) - 1) * (len(yedges) - 1)
ypos, xpos = np.meshgrid(y_midp, x_midp)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = (xedges[1] - xedges[0]) * np.ones_like(bin_num)
dy = (yedges[1] - yedges[0]) * np.ones_like(bin_num)
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='g', zsort='average')
ax.set_xlabel('duration(t) (mins.)')
ax.set_ylabel('duration(t+1) (mins.)')
ax.set_zlabel('Percent of obs.')
plt.title('Histogram by Old Faithful duration(t) and duration(t-1)')

##### 3-d scatter plot with iris data 
from mpl_toolkits.mplot3d import Axes3D

length_sep_set = iris_data[iris_data['species']=='setosa']['length_sep']
width_sep_set = iris_data[iris_data['species']=='setosa']['width_sep']
length_pet_set = iris_data[iris_data['species']=='setosa']['length_pet']
length_sep_ver = iris_data[iris_data['species']=='versicolor']['length_sep']
width_sep_ver = iris_data[iris_data['species']=='versicolor']['width_sep']
length_pet_ver = iris_data[iris_data['species']=='versicolor']['length_pet']
length_sep_vrg = iris_data[iris_data['species']=='virginica']['length_sep']
width_sep_vrg = iris_data[iris_data['species']=='virginica']['width_sep']
length_pet_vrg = iris_data[iris_data['species']=='virginica']['length_pet']

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.scatter(length_sep_set, width_sep_set, length_pet_set,
           color='k', label='setosa')
ax.scatter(length_sep_ver, width_sep_ver, length_pet_ver,
           color='r', label='versicolor')
ax.scatter(length_sep_vrg, width_sep_vrg, length_pet_vrg,
           color='y', label='virginica')

# plt.xticks(np.round_(bin_cuts, 1))
plt.title('Sepal length, sepal width, and petal length by species', fontsize=15)
ax.set_xlabel(r'sepal length (mm)')
ax.set_ylabel(r'sepal width (mm)')
ax.set_zlabel(r'petal length (mm)')
plt.legend(loc=6)

############################
# Line Plot 
###########################

# plt.plot()
# call two on same ax just by calling one after another 
emat = np.loadtxt('DataFiles/emat.csv', delimiter=',')

import matplotlib
from matplotlib import cm
cmap1 = matplotlib.cm.get_cmap('summer')
ages = np.arange(21, 101)
abil_midp = np.array([12.5, 37.5, 60, 75, 85, 94, 99.5])
abil_mesh, age_mesh = np.meshgrid(abil_midp, ages)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(age_mesh, abil_mesh, emat, rstride=8,
                cstride=1, cmap=cmap1)
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'ability type -$j$')
ax.set_zlabel(r'ability $e_{j,s}$')

## level curves are better if printed
abil_pcts = np.array([0.0, 0.25, 0.50, 0.70, 0.80, 0.90, 0.99, 1.0])
J = 7

fig = plt.figure()
ax = plt.subplot(111)
linestyles = np.array(["-", "--", "-.", ":",])
markers = np.array(["x", "v", "o", "d", ">", "|"])
pct_lb = 0
for j in range(J):
    this_label = (str(int(np.rint(pct_lb))) + " - " +
        str(int(np.rint(pct_lb + 100*abil_pcts[j]))) + "%")
    pct_lb += 100*abil_pcts[j]
    if j <= 3:
        ax.plot(ages, np.log(emat[:, j]), label=this_label,
            linestyle=linestyles[j], color='black')
    elif j > 3:
        ax.plot(ages, np.log(emat[:, j]), label=this_label,
            marker=markers[j-4], color='black')
ax.axvline(x=80, color='black', linestyle='--')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'log ability $log(e_{j,s})$')


###################
# Dynamic graphics 
##################
from PIL import Image, ImageDraw
# linked brushing 
import mpld3

# D3 graphics are the wave of the future bro
import Bokeh 
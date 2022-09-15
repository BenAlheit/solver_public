import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np

# type = "Viscoelastic"
type = "Elastic"

results = f"./{type}VolumeAverages/"

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']

SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 13
FONT = 14

plt.rc('font', size=FONT)  # controls default text sizes
plt.rc('axes', titlesize=FONT)  # fontsize of the axes title
plt.rc('axes', labelsize=FONT)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


F = pd.read_csv(f"{results}VolumeAveragedDeformationGradient.csv")
sig = pd.read_csv(f"{results}VolumeAveragedStress.csv")

plt.plot(F[" DeformationGradient11"], sig[" Stress11"], color='black')

scat = plt.scatter(F[" DeformationGradient11"], sig[" Stress11"], c=F[" time"], cmap='viridis')
plt.grid()
plt.ylabel("$\sigma_{11}$ (MPa)")
plt.xlabel("$F_{11}$")
c_bar = plt.colorbar(scat)

c_bar.set_label('time (s)')

plt.title(f"{type} cyclical loading at increasing loading rate")

plt.tight_layout()
plt.savefig(f"{results}/{type}.pdf")
plt.savefig(f"{results}/{type}.png")
plt.show()
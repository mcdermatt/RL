import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep

fig = plt.figure(0)
SA_patch = mpatches.Patch(color = 'red', label = 'Sample-Average')
WA_pt125_patch = mpatches.Patch(color = 'navy', label = 'Weighted-Average α = 0.125')


sa = np.load('saStationary.npy')
wapt125 = np.load('wa_pt125Stationary.npy')
saOpt = np.load('saOptStationary.npy')
waOptpt125 = np.load('waOpt_pt125Stationary.npy')

ax1 = fig.add_subplot(211)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Average Reward')
ax1.set_ylim([0,1])
ax1.legend(handles = [SA_patch, WA_pt125_patch])
ax1.set_title('Stationary Bandits ε = 0.1')

ax2 = fig.add_subplot(212)
ax2.set_xlabel('Steps')
ax2.set_ylabel('% Optimal Action')
ax2.legend(handles = [SA_patch, WA_pt125_patch])

ax1.plot(sa[sa != 0], color = 'red', lw = 0.25)
ax1.plot(wapt125[wapt125 != 0], color ='navy', lw = 0.25)

ax2.plot(saOpt[saOpt != 0], color = 'red', lw = 0.25)
ax2.plot(waOptpt125[waOptpt125 != 0], color ='navy', lw = 0.25)

plt.draw()
plt.savefig('Results2.png')
plt.pause(30)
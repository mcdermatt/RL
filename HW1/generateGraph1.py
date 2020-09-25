import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep

fig = plt.figure(0)
SA_patch = mpatches.Patch(color = 'red', label = 'Sample-Average')
WA_1_patch = mpatches.Patch(color = 'slategrey', label = 'Weighted-Average α = 1')
WA_pt5_patch = mpatches.Patch(color = 'cornflowerblue', label = 'Weighted-Average α = 0.5')
WA_pt25_patch = mpatches.Patch(color = 'blue', label = 'Weighted-Average α = 0.25')
WA_pt125_patch = mpatches.Patch(color = 'navy', label = 'Weighted-Average α = 0.125')
WA_pt01_patch = mpatches.Patch(color = 'mediumpurple', label = 'Weighted-Average α = 0.01')

sa = np.load('sa.npy')
wa1 = np.load('wa_1.npy')
wapt5 = np.load('wa_pt5.npy')
wapt25 = np.load('wa_pt25.npy')
wapt125 = np.load('wa_pt125.npy')
wapt01 = np.load('wa_pt01.npy')

saOpt = np.load('saOpt.npy')
waOpt1 = np.load('waOpt_1.npy')
print(waOpt1)
waOptpt5 = np.load('waOpt_pt5.npy')
waOptpt25 = np.load('waOpt_pt25.npy')
waOptpt125 = np.load('waOpt_pt125.npy')
waOptpt01 = np.load('waOpt_pt01.npy')

ax1 = fig.add_subplot(211)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Average Reward')
ax1.set_ylim([0,2])
ax1.legend(handles = [SA_patch, WA_1_patch, WA_pt5_patch, WA_pt25_patch, WA_pt125_patch, WA_pt01_patch])
ax1.set_title('Nonstationary Bandits ε = 0.1')

ax2 = fig.add_subplot(212)
ax2.set_xlabel('Steps')
ax2.set_ylabel('% Optimal Action')
ax2.legend(handles = [SA_patch, WA_1_patch, WA_pt5_patch, WA_pt25_patch, WA_pt125_patch, WA_pt01_patch])

ax1.plot(sa[sa != 0], color = 'red', lw = 0.25)
ax1.plot(wa1[wa1 != 0], color ='slategrey', lw = 0.25)
ax1.plot(wapt5[wapt5 != 0], color ='cornflowerblue', lw = 0.25)
ax1.plot(wapt25[wapt25 != 0], color ='blue', lw = 0.25)
ax1.plot(wapt125[wapt125 != 0], color ='navy', lw = 0.25)
ax1.plot(wapt01[wapt01 != 0], color ='mediumpurple', lw = 0.25)

ax2.plot(saOpt[saOpt != 0], color = 'red', lw = 0.25)
ax2.plot(waOpt1[waOpt1 != 0], color ='slategrey', lw = 0.25)
ax2.plot(waOptpt5[waOptpt5 != 0], color ='cornflowerblue', lw = 0.25)
ax2.plot(waOptpt25[waOptpt25 != 0], color ='blue', lw = 0.25)
ax2.plot(waOptpt125[waOptpt125 != 0], color ='navy', lw = 0.25)
ax2.plot(waOptpt01[waOptpt01 != 0], color ='mediumpurple', lw = 0.25)

plt.draw()
plt.savefig('Results.png')
plt.pause(30)
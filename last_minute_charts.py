import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

palette = sns.color_palette()

plt.plot([2,5,10,20], [3.82,8.72,9.29,np.nan], color=palette[0], label='ABM.2')
plt.plot([2,5,10,20], [7,7,7,7], color=palette[1], label='SMOTEBoost1')
plt.plot([2,5,10,20], [7.08,7.08,7.08,7.08], color=palette[2], label='PFITCCBoost1')
plt.plot([2,5,10,20], [4.62,4.75,5.46,np.nan], color=palette[3], label='PFFCBoost1')
plt.xlabel('Number of estimators')
plt.ylabel('MC Los Logg')
plt.title('AdaBoost Performance At Varying # Of Estimators (syn_ratio=1, proj_ratio=1)')
plt.legend(loc='best')
plt.savefig('boostingchart.png')
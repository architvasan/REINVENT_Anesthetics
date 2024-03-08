import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('staged_learning_1.csv')
av_scores = [np.mean(df[df['step']==st]['Score']) for st in range(1, 28)]
plt.plot(av_scores)
plt.savefig('reward_1.png', dpi=300)
plt.close()

df = pd.read_csv('staged_learning_2.csv')
av_scores = [np.mean(df[df['step']==st]['Score']) for st in range(1, 28)]
plt.plot(av_scores)
plt.savefig('reward_2.png', dpi=300)
plt.close()


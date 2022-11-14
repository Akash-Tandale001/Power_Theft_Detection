# 2D data plot
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def visualisation2D():
    rawData1 = pd.read_csv('./visualization.csv', nrows=3) 
    cols = rawData1.columns
    rawData2 = pd.read_csv('./visualization.csv', skiprows=187)
    rawData2.columns = cols
    data = pd.concat([rawData1, rawData2], ignore_index=True)

    fig4, axs4 = plt.subplots(2, 1)
    fig4.suptitle('Four Week Consumption', fontsize=16)
    plt.subplots_adjust(hspace=0.5)

    for i in range(59, 83, 7):
        axs4[0].plot(data.iloc[1, i:i + 7].to_numpy(), marker='>', linestyle='-',label='$week {i}$'.format(i=(i % 58) % 6))
    axs4[0].legend(loc='best')
    axs4[0].set_title('With Fraud', fontsize=14)
    axs4[0].set_ylabel('Consumption')
    axs4[0].grid(True)

    for i in range(59, 83, 7):
        axs4[1].plot(data.iloc[6, i:i + 7].to_numpy(), marker='>', linestyle='-',label='$week {i}$'.format(i=(i % 58) % 6))
    axs4[1].legend(loc='best')
    axs4[1].set_title('Without fraud', fontsize=14)
    axs4[1].set_ylabel('Consumption')
    axs4[1].grid(True)
    plt.show()

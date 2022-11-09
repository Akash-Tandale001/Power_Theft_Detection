import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def visualisation1D():
    rawData1 = pd.read_csv('./visualization.csv', nrows=3) #taking first 3 rows
    cols = rawData1.columns
    rawData2 = pd.read_csv('./visualization.csv', skiprows=187) #removing first 189 rows
    rawData2.columns = cols
    data = pd.concat([rawData1, rawData2], ignore_index=True) #ignore_index=True to make row index number 

    #plot 1D graph for consumer
    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Consumers With Fraud', fontsize=18)
    plt.subplots_adjust(hspace=0.8)


    data.loc[0].plot(ax=axs[0], color='firebrick', grid=True)
    axs[0].set_title('Consumer 0', fontsize=16)
    axs[0].set_xlabel('Dates of Consumption')
    axs[0].set_ylabel('Consumption')

    data.loc[2].plot(ax=axs[1], color='firebrick', grid=True)
    axs[1].set_title('Consumer 1', fontsize=16)
    axs[1].set_xlabel('Dates of Consumption')
    axs[1].set_ylabel('Consumption')

    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Consumers Without Fraud', fontsize=18)
    plt.subplots_adjust(hspace=0.8)

    data.loc[3].plot(ax=axs[0], color='teal', grid=True)
    axs[0].set_title('Consumer 40255', fontsize=16)
    axs[0].set_xlabel('Dates of Consumption')
    axs[0].set_ylabel('Consumption')

    data.loc[4].plot(ax=axs[1], color='teal', grid=True)
    axs[1].set_title('Consumer 40256', fontsize=16)
    axs[1].set_xlabel('Dates of Consumption')
    axs[1].set_ylabel('Consumption')

    #statistics for consumer 
    #with fraud
    fig2, axs2 = plt.subplots(2, 1)
    fig2.suptitle('Statistics for Consumers with Fraud', fontsize=18)
    plt.subplots_adjust(hspace=0.8)

    data.loc[0].plot(ax=axs2[0], color='firebrick', grid=True)
    axs2[0].set_title('Consumption of Consumer 0', fontsize=16)
    axs2[0].set_xlabel('Dates of Consumption')
    axs2[0].set_ylabel('Consumption')


    data.loc[0].plot.kde(color='firebrick', ax=axs2[1], grid=True)
    axs2[1].set_title('Density Estimation', fontsize=16)
    axs2[1].set_xlabel('Values')
    axs2[1].set_ylabel('Density')


    #without fraud
    fig3, axs3 = plt.subplots(2,1)
    fig3.suptitle('Statistics for Consumers without Fraud', fontsize=18)
    plt.subplots_adjust(hspace=0.8)
    data.loc[4].plot(ax=axs3[0], color='teal', grid=True)
    axs3[0].set_title('Consumption of Consumer 40256', fontsize=16)
    axs3[0].set_xlabel('Dates of Consumption')
    axs3[0].set_ylabel('Consumption')


    data.loc[4].plot.kde(color='teal', ax=axs3[1], grid=True)
    axs3[1].set_title('Density Estimation', fontsize=16)
    axs3[1].set_xlabel('Values')
    axs3[1].set_ylabel('Density')


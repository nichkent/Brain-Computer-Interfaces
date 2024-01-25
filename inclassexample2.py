# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:41:39 2024

@author: nicho
"""

from matplotlib import pyplot as plt
import numpy as np

def plot_balance(weekly_salary=200):
    bank_balance = [0,200,400,600]
    iphone_price = 1000
    #weekly_salary = 200
    #bank_balance = np.array([0,200,400,600])
    bank_balance = np.arange(0,2000,weekly_salary) # amount in the bank
    bank_balance = bank_balance + 20 # find $20 in the couch
    
    # plot results
    plt.figure(1,clear=True)
    plt.plot(bank_balance)
    plt.plot([0,9], [iphone_price,iphone_price])
    
    # annotate plot
    plt.xlabel('time (weeks)')
    plt.ylabel('bank balance ($)')
    plt.title('Savings over time')
    plt.legend(['bank balance', 'iPhone price'])
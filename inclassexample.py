# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:50:41 2024

@author: nicho
"""
from inclassexample2 import plot_balance

# variables and printing
iphone_price = 1000
print("iphone price=" + str(iphone_price))
print("iphone price =", iphone_price)
print(f"2 iphones price = {iphone_price*2}")

# declare a time series
bank_balance = [0,200,400,600]
#bank_balance = bank_balance + 20 # throws an error
    
plot_balance(300)
plot_balance(600)
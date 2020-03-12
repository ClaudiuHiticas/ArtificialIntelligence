#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:35:04 2020

@author: claudiuhiticas
"""

import matplotlib.pyplot as plt
from scipy.stats import binom
import scipy.special as sps
import numpy as np
import scipy


def showMenu():
    print("1. Gamma Distribution")
    print("2. Poisson Distribution")
    print("3. Normal Distribution")
    

def gamma():
    shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
    s = np.random.gamma(shape, scale, 1000)
    count, bins, ignored = plt.hist(s, 50, density=True)
    y = bins**(shape-1)*(np.exp(-bins/scale) /
                         (sps.gamma(shape)*scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')
    plt.ylabel('gamma distribution')
    plt.show()


def poisson():
    lambda_, size = 5, 10000
    s = np.random.poisson(lambda_, size)
    count, bins, ignored = plt.hist(s, 18, density=True)
    pdf = lambda_ ** bins * np.exp(-lambda_) / scipy.special.factorial(bins)
    plt.plot(bins, pdf, linewidth=3, color='r')
    plt.ylabel('poisson distribution')
    plt.show()
    

def normal():
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                   np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    plt.ylabel('normal distribution')
    plt.show()


def run():
    showMenu()
    while(True):
        n = int(input("n = "))
        if n == 1:
            gamma()
        elif n == 2:
            poisson()
        elif n == 3:
            normal()
        elif n == 0:
            break
        else:
            print("Wrong input")


run()

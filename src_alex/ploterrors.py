# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 20:44:25 2025

@author: Alex Tang
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import plot_errs

wide = np.load("errs_tf_tr29.npz")
narrow = np.load("errs_tf_tr47.npz")

errs29 = [wide["e29"], narrow["e29"]]
errs47 = [wide["e47"], narrow["e47"]]
errs27 = [wide["e27"], narrow["e27"]]
labels = ["Train [0.2, 0.9)", "Train [0.4, 0.7)"]

plt.figure()
plot_errs(labels, errs29)
plt.title("Test Distribution [0.2, 0.9)", fontsize=32)

plt.figure()
plot_errs(labels, errs47)
plt.title("Test Distribution [0.4, 0.7)", fontsize=32)

plt.figure()
plot_errs(labels, errs27)
plt.title("Test Distribution [0.2, 0.7)", fontsize=32)

longtimes = np.load("longtimes.npz")
errs_long = [longtimes["e2929"], longtimes["e2947"], longtimes["e4747"], longtimes["e4729"]]
plt.figure()
plot_errs(labels, [errs_long[0], errs_long[-1]])
plt.title("Test Distribution [0.2, 0.9)", fontsize=32)
plt.ylim(3*10**-1)
plt.figure()
plot_errs(labels, [errs_long[1], errs_long[2]])
plt.title("Test Distribution [0.4, 0.7)", fontsize=32)
plt.ylim(3*10**-1)
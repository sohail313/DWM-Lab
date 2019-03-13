#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:05:37 2019

0"""

import numpy as np

def pagerank(M, eps=1.0e-8, d=0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    last_v = np.ones((N, 1), dtype=np.float32) * 100
    
    while np.linalg.norm(v - last_v, 2) > eps:
        last_v = v
        v = d * np.matmul(M, v) + (1 - d) / N
    return v
print("Enter the adjacency matrix")
for i in range(5):
    for j in range (5):
        M[i][j]=float(input("Enter the values"))
print(M)

v = pagerank(M, 0.001, 0.85)

print (v)
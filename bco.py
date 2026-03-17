import numpy as np
from fun_range import fun_range
from benchmark import ben_functions
from space_bound import space_bound
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Bezier Curve-based Optimization (BCO)       %%
% FunInd: Index of function                         %
% MaxIt: Maximum number of iterations               %
% PopSize: Size of population                       %
% PopPos: Position of individual population         %
% PopFit: Fitness of individual population          %
% Dim: Dimensionality of problem                    %
% BestPos: Best solution found so far               %
% BestFit: Best fitness corresponding to BestX      %
% HisBestFit: History best fitness over iterations  %
% Low: Low bound of search space                    %
% Up: Up bound of search space                      %
% A: Exploration-exploitation balance operator      %
% Alpha:The weight                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def bco(FunInd, MaxIt, PopSize):

    Low, Up, Dim = fun_range(FunInd)
    PopPos = np.zeros((PopSize, Dim))
    PopFit = np.zeros(PopSize)
    for i in range(PopSize):
        PopPos[i] = np.random.rand(Dim) * (Up - Low) + Low
        PopFit[i] = ben_functions(PopPos[i], FunInd, Dim)
    HisBestFit = np.zeros(MaxIt)
    idx = np.argmin(PopFit)
    BestFit = PopFit[idx]
    BestPos = PopPos[idx].copy()
    for It in range(1, MaxIt + 1):
        Alpha0 = 1 + 80 * It / MaxIt + 0.02 * (10 * It / MaxIt) ** 3
        A0 = np.sin(np.pi / 2 * (1 - It / MaxIt))
        for i in range(PopSize):
            Alpha = 1 + (2 * np.random.rand() - 1) / Alpha0  # Eq.(10)
            A = (0.4 + 2 * np.log(1 / np.random.rand())) * A0  # Eq.(19)
            if A > 1:
                if np.random.rand() > 0.5:
                    j = i
                    while j == i:
                        j = np.random.randint(PopSize)
                    r1 = np.random.rand()
                    k = i
                    while k == i or k == j:
                        k = np.random.randint(PopSize)
                    if r1 < 0.8:  # Eq.(17)
                        P3 = np.mean(PopPos, axis=0) + 2 * np.random.randn(Dim) * (PopPos[k] - PopPos[j])
                    else:
                        if r1 > 0.9:
                            rand_dim = np.random.randint(Dim)
                            P3 = PopPos[j] + 2 * np.random.randn() * (PopPos[k, rand_dim] - PopPos[j])
                        else:
                            P3 = PopPos[j] + 2 * np.random.randn() * (Low + np.random.rand() * (Up - Low))
                    P0 = PopPos[i]
                    P1 = PopPos[j]
                    P2 = PopPos[k]
                    NewPos = (1 - Alpha) ** 3 * P0 + 3 * (1 - Alpha) ** 2 * Alpha * P1 + \
                             3 * (1 - Alpha) * Alpha ** 2 * P2 + Alpha ** 3 * P3  # Eq.(18)
                else:
                    j = i
                    while j == i:
                        j = np.random.randint(PopSize)
                    P0 = PopPos[i]
                    P1 = PopPos[j]
                    if np.random.rand() > 0.5:  # Eq.(15)
                        P2 = PopPos[i] + np.random.randn() * (BestPos - PopPos[j])
                    else:
                        P2 = PopPos[i] + np.random.randn() * (np.mean(PopPos, axis=0) - PopPos[j])
                    NewPos = (1 - Alpha) ** 2 * P0 + 2 * (1 - Alpha) * Alpha * P1 + Alpha ** 2 * P2  # Eq.(16)
            else:  # A <= 1
                if np.random.rand() > 0.5:
                    j = i
                    while j == i:
                        j = np.random.randint(PopSize)
                    P0 = PopPos[i]
                    if np.random.rand() < 0.5:  # Eq.(8)
                        P1 = BestPos + (PopPos[j] - PopPos[i]) / 2
                    else:
                        P1 = BestPos - (PopPos[j] + PopPos[i]) / 2
                    NewPos = (1 - Alpha) * P0 + Alpha * P1  # Eq.(9)
                else:
                    if np.random.rand() > 0.5:
                        P0 = PopPos[i]
                        if np.random.rand() > 0.5:  # Eq.(11)
                            P1 = np.mean(PopPos, axis=0) + np.random.randn() * (PopPos[i] - np.mean(PopPos, axis=0))
                        else:
                            P1 = np.mean(PopPos, axis=0) + np.random.randn() * np.mean(PopPos, axis=0)
                        NewPos = (1 - Alpha) * P0 + Alpha * P1  # Eq.(12)
                    else:
                        P0 = PopPos[i]
                        P1 = np.zeros(Dim)
                        for j in range(Dim):
                            if np.random.rand() > 0.5:  # Eq.(13)
                                k = i
                                while k == i:
                                    k = np.random.randint(PopSize)
                                P1[j] = (1 - Alpha) * PopPos[i, j] + Alpha * PopPos[k, j]
                            else:
                                P1[j] = PopPos[i, j]
                        NewPos = (1 - Alpha) * P0 + Alpha * P1  # Eq.(14)
            NewPos = space_bound(NewPos, Up, Low)
            NewFit = ben_functions(NewPos, FunInd, Dim)
            if NewFit < PopFit[i]:
                PopFit[i] = NewFit
                PopPos[i] = NewPos
                if NewFit < BestFit:
                    BestPos = NewPos.copy()
                    BestFit = NewFit
        HisBestFit[It - 1] = BestFit
    return BestPos, BestFit, HisBestFit
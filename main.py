import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from bco import bco
"""
%--------------------------------------------------------------------------%
% Bezier Curve-based Optimization (BCO)                                    %
% Source codes demo version 1.0                                            %
%--------------------------------------------------------------------------%
% The code is based on the following paper:                                %
% Zhao, W., Xie Y., Wang, L., Zhang. Z., Khodadadi, N., Mirjalili, S.      %
% (2026). An effective Bezier curve-based optimization (BCO) for           %
% large-scale numerical problems and 3D unmanned aerial vehicle path       %
% planning with efficient multiple threats evasion, Advanced Engineering   %
% Informatics, 73, 104524. https://doi.org/10.1016/j.aei.2026.104524.      %
%--------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BestX:The best solution                   %
% BestFit:The best fitness                  %
% HisBestFit:History of the best fitness    %
% FunIndex:Index of functions               %
% MaxIteration: Maximum number of iterations%
% PopSize: Size of population               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
if __name__ == "__main__":
    MaxIteration = 500
    PopSize = 50
    FunIndex = 1

    BestX, BestFit, HisBestFit = bco(FunIndex, MaxIteration, PopSize)
    print(f'The best fitness of F{FunIndex} is: {BestFit}')

    plt.figure()
    if BestFit > 0:
        plt.semilogy(HisBestFit, 'r', linewidth=2)
    else:
        plt.plot(HisBestFit, 'r', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'F{FunIndex}')
    plt.grid(True)
    plt.show()
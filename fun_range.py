import numpy as np

def fun_range(FunIndex):

    Dim = 30
    if FunIndex == 1:
        Low, Up = -100, 100
    elif FunIndex == 2:
        Low, Up = -10, 10
    elif FunIndex == 3:
        Low, Up = -100, 100
    elif FunIndex == 4:
        Low, Up = -100, 100
    elif FunIndex == 5:
        Low, Up = -30, 30
    elif FunIndex == 6:
        Low, Up = -100, 100
    elif FunIndex == 7:
        Low, Up = -1.28, 1.28
    elif FunIndex == 8:
        Low, Up = -500, 500
    elif FunIndex == 9:
        Low, Up = -5.12, 5.12
    elif FunIndex == 10:
        Low, Up = -32, 32
    elif FunIndex == 11:
        Low, Up = -600, 600
    elif FunIndex == 12:
        Low, Up = -50, 50
    elif FunIndex == 13:
        Low, Up = -50, 50
    elif FunIndex == 14:
        Low, Up = -65.536, 65.536
        Dim = 2
    elif FunIndex == 15:
        Low, Up = -5, 5
        Dim = 4
    elif FunIndex == 16:
        Low, Up = -5, 5
        Dim = 2
    elif FunIndex == 17:
        Low = [-5, 0]
        Up = [10, 15]
        Dim = 2
    elif FunIndex == 18:
        Low, Up = -2, 2
        Dim = 2
    elif FunIndex == 19:
        Low, Up = 0, 1
        Dim = 3
    elif FunIndex == 20:
        Low, Up = 0, 1
        Dim = 6
    elif FunIndex == 21:
        Low, Up = 0, 10
        Dim = 4
    elif FunIndex == 22:
        Low, Up = 0, 10
        Dim = 4
    else:
        Low, Up = 0, 10
        Dim = 4

    if isinstance(Low, (int, float)):
        Low = np.full(Dim, Low)
        Up = np.full(Dim, Up)
    else:
        Low = np.array(Low)
        Up = np.array(Up)
    return Low, Up, Dim
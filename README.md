# RandomArrayGenerator-python
Creates ndarray of random numbers based on a specific method:         

NR - a linear congruential generator (LCG) with coefficients a, b, c = 1664525, 1013904223, 2**32         
RANDU - a LCG with coefficients a, b, c = 65539, 0, 2**31       
LFG - Lagged Fibonacci Generator         

size (rows, columns), columns are x,y,z values         

returnSeed=
  True p1 returns tuple (output, seed)
  False only outputs array

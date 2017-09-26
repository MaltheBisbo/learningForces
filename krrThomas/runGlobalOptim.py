import numpy as np
import matplotlib.pyplot as plt
from globalOptim import globalOptim

def main():
    Eparams = (1, 1.4, np.sqrt(0.02))
    optim = globalOptim(Natoms=12, Eparams=Eparams)
    optim.makeInitialStructure()
    optim.relax()
    optim.plotCurrentStructure()

if __name__ == '__main__':
    main()

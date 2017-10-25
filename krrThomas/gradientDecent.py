import numpy as np

def gradientDecent(x0, Efun, gradfun, stepsize=100, precision=1e-4, maxLSsteps=15, maxSteps=100, c=0.1, tau=0.5):
    """
    ---Input---
    x0: Initial coordinates for the minimization
    
    Efun: Function calculating energy based on coordiantes.

    gradfun: Function calculating gradient based on coordinates

    stepsize: Initial step size for backtracking line-search. Must
           be larger than the the expected, since the algorithm
           backtracks.

    precision: convergence criteria based on the size of the steps taken.

    maxLSsteps: The maximum number of line-search iterations performed.

    maxSteps: The maximum number of decent steps.

    c: The minimal fraction of energy decrease per stepsize (as indicated by
       first order taylor expansion) Which is demanded during linesearch.

    tau: reduction of the stepsize per line-search iteration.
    """
        
    def linesearch(x, grad):
        E0 = Efun(x)
        gamma = stepsize
        E = Efun(x)
        m = np.linalg.norm(grad)
        t=c*m
        
        for i in range(maxLSsteps):
            print(gamma)
            Enew = Efun(x-gamma*grad)
            if E - Enew > gamma*t:
                return gamma, Enew
            else:
                gamma *= tau
            E = Enew
        assert E < E0
        return gamma/tau, E
                
    cur_x = x0
    previous_step_size = None
    E = None
    for i in range(maxSteps):
        prev_x = cur_x
        grad = gradfun(prev_x)
        gamma, E = linesearch(prev_x, grad)
        cur_x += -gamma*grad
        previous_step_size = np.linalg.norm(cur_x - prev_x)
        if previous_step_size > precision:
            print('# decent steps:', i)
            return cur_x, E

        print('Maximum number of iterations exceeded')
        return cur_x, E

if __name__ == '__main__':
    from doubleLJ import doubleLJ, doubleLJ_energy, doubleLJ_gradient
    import matplotlib.pyplot as plt
    
    def makeConstrainedStructure(Natoms):
        boxsize = 1.5 * np.sqrt(Natoms)
        rmin = 0.9
        rmax = 1.5
        def validPosition(X, xnew):
            Natoms = int(len(X)/2) # Current number of atoms
            if Natoms == 0:
                return True
            connected = False
            for i in range(Natoms):
                r = np.linalg.norm(xnew - X[2*i:2*i+2])
                if r < rmin:
                    return False
                if r < rmax:
                    connected = True
            return connected

        Xinit = np.zeros(2*Natoms)
        for i in range(Natoms):
            while True:
                xnew = np.random.rand(2) * boxsize
                if validPosition(Xinit[:2*i], xnew):
                    Xinit[2*i:2*i+2] = xnew
                    break
        return Xinit

    def Efun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_energy(X, params[0], params[1], params[2])
    def gradfun(X):
        params = (1.5, 1, np.sqrt(0.02))
        return doubleLJ_gradient(X, params[0], params[1], params[2])
    
    Natoms = 7
    Ndata = 10
    X = np.array([makeConstrainedStructure(Natoms) for i in range(Ndata)])

    for i in range(Ndata):
        x = X[i]
        E = Efun(x)
        xrelaxed, Erelaxed = gradientDecent(x, Efun, gradfun)

        plt.figure(i)
        plt.scatter(x[0::2], x[1::2], s=22, color='g', marker='x', label='Initial positions')
        plt.scatter(xrelaxed[0::2], xrelaxed[1::2], s=22, color='r', marker='o', label='Relaxed positions')
        plt.legend()
    plt.show()
    
        
        
    

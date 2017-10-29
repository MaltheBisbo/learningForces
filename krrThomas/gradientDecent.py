import numpy as np

def gradientDecent(x0, Efun, gradfun, stepsize=0.05, precision=1e-3, maxLSsteps=15, maxSteps=100, c=0.5, tau=0.5):
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
            Enew = Efun(x-gamma*grad)
            if E0 - Enew > gamma*t:
                return gamma, Enew
            else:
                gamma *= tau
        assert Enew < E0
        return gamma/tau, E
                
    cur_x = x0.copy()
    previous_step_size = None
    E = None
    print('E0:', Efun(x0))
    for i in range(maxSteps):
        prev_x = cur_x.copy()
        grad = gradfun(prev_x)
        gamma, E = linesearch(prev_x, grad)
        cur_x += -gamma*grad
        previous_step_size = np.linalg.norm(cur_x - prev_x)
        if previous_step_size < precision:
            print('# decent steps:', i)
            return cur_x, E

    print('Maximum number of iterations exceeded')
    return cur_x, E

if __name__ == '__main__':
    from doubleLJ import doubleLJ, doubleLJ_energy, doubleLJ_gradient
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
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
    Ndata = 1
    X = np.array([makeConstrainedStructure(Natoms) for i in range(Ndata)])

    # options = {'gtol': 1e-5}

    def callback(x_cur):
        global Xtraj
        Xtraj.append(x_cur)
                
    def localMinimizer(X, tol=1e-1):
        global Xtraj
        Xtraj = []
        
        res = minimize(Efun, X, method="BFGS", jac=gradfun, tol=tol, callback=callback)#, options=options)
        print('nfev:', res.nfev,' , njev:', res.njev)
        Xtraj = np.array(Xtraj)
        return res.x, res.fun, Xtraj
    
    for i in range(Ndata):
        x = X[i]
        E = Efun(x)
        xrel1, Erel1 = gradientDecent(x, Efun, gradfun)
        xrel2, Erel2, Xtraj = localMinimizer(x, tol=1e-1)
        Etraj = np.array([Efun(x) for x in Xtraj])
        
        plt.figure(2*i)
        plt.scatter(x[0::2], x[1::2], s=22, color='g', marker='x', label='Init. pos')
        plt.scatter(xrel1[0::2], xrel1[1::2], s=22, color='r', marker='o', label='Rel. pos MY')
        plt.scatter(xrel2[0::2], xrel2[1::2], s=22, color='b', marker='o', label='Rel. pos BFGS1')
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')

        
        Nsteps = len(Etraj)
        plt.figure(2*i+1)
        plt.plot(np.arange(Nsteps), Etraj)
        plt.scatter(np.arange(Nsteps)[0::3], Etraj[0::3])
        """
        Eend = Etraj[-1]
        Etraj = Etraj[0::3]
        
        Etraj_reduced = []
        for i in range(len(Etraj)):
            if len(Etraj_reduced) == 0:
                Etraj_reduced.append(Etraj[i])
                continue
            elif Etraj_reduced[-1] - Etraj[i] > 0.1:
                Etraj_reduced.append(Etraj[i])
        Etraj_reduced[-1] = Eend
        """
        Nskip = 3
        min_Ediff = 0.1
        def trimData(Xtraj):
            Etraj = np.array([Efun(x) for x in Xtraj])
            Nstep = len(Etraj)
            index = []
            k = 0
            Ecur = Etraj[0]
            while k < Nstep:
                if len(index) == 0:
                    index.append(0)
                    continue
                elif Ecur - Etraj[k] > min_Ediff:
                    index.append(k)
                    Ecur = Etraj[k]
                k += Nskip
            index[-1] = Nstep - 1
            return index

        trimIndices = trimData(Xtraj)
        E = Etraj[trimIndices]
        
        #plt.scatter(np.arange(len(Etraj_reduced))*3, Etraj_reduced, color='r')
        plt.scatter(np.arange(len(E))*3, E, color='y')
        
    plt.show()
    
        
        
    

from ase.ga.utilities import closest_distances_generator
from ase.io import read, write, Trajectory
from ase.calculators.calculator import PropertyNotImplementedError
from ase.visualize import view
from ase import Atoms, Atom
import copy
import numpy as np
import os
from subprocess import Popen, PIPE
from random import random
from ase.optimize import BFGS, LBFGS, QuasiNewton
from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer
from gpaw import extra_parameters
extra_parameters['blacs'] = True
from gpaw.utilities import h2gpts
from ase.ga.relax_attaches import VariansBreak
from startgenerator import StartGenerator

import clusterOptimization.LocalEnergies.coordinateSet as cs
import clusterOptimization.LocalEnergies.energyModel as em
import clusterOptimization.LocalEnergies.visualizeData as vd
import clusterOptimization.LocalEnergies.featureClustering as fc
import random as rd
import time
import ase.parallel as mpi
from sklearn.externals import joblib

from ase.ga.standard_comparators import InteratomicDistanceComparator

world = mpi.world


def rattleAtom(coordinate, coordinates, distRange):
    """
    Rattle a single atom

    Parameters
    ----------
    coordinates : coordinates as numpy array [x1, y1]

    distRange : range of the rattle mutation

    Returns
    -------
    coordinates : new coordinates after the mutation
    """
    
    atoms = coordinates.shape[0]
    
    strucStart = Atoms('24C', positions = coordinates.copy())
    strucRattle = Atoms('24C', positions = coordinates.copy())
    mindis = 0
    mindisAtom = 10
    
    while mindis < 1 or mindisAtom > 3:  # If the atom is too close or too far away

        # First load original positions
        strucRattle.positions = strucStart.positions.copy()
        
        # Then Rattle within a circle 
        r = distRange * np.sqrt(rd.random())
        theta = np.random.uniform(low = 0, high = 2 * np.pi)
        coordinates[coordinate] += r * np.array([np.cos(theta), np.sin(theta), 0])

        strucRattle.positions = coordinates
        dis = strucRattle.get_all_distances()
        mindis = np.min(dis[np.nonzero(dis)]) # Check that we are not too close 
        mindisAtom = np.min(strucRattle.get_distances(coordinate, np.delete(np.arange(atoms), coordinate))) # check that we are not too far 

        # If it does not fit, try to wiggle it into place using small circle 
        if mindis < 1:
            for i in range(10):
                r = 0.5 * np.sqrt(rd.random())
                theta = np.random.uniform(low = 0, high = 2 * np.pi)
                coordinates[coordinate] += r * np.array([np.cos(theta), np.sin(theta), 0])
                strucRattle.positions = coordinates
                dis = strucRattle.get_all_distances()
                mindis = np.min(dis[np.nonzero(dis)])

                # If it works break
                if mindis > 1:
                    break

                # Otherwise reset coordinate 
                else:
                    coordinates[coordinate] -= r * np.array([np.cos(theta), np.sin(theta), 0])

    return coordinates[coordinate].copy()


def rattleAtomFromCenter(coordinate, coordinates, distRange):
    """
    Rattle a single atom by placing it in the center of the super cell and then rattle 

    Parameters
    ----------
    coordinate : index of the atom to be rattled

    coordinates : numpy array of all the positions 

    distRange : range of the rattle mutation

    Returns
    -------
    coordinate : new coordinate after the mutation

    """

    atoms = coordinates.shape[0]
    
    strucStart = Atoms('24C', positions = coordinates.copy())
    strucRattle = Atoms('24C', positions = coordinates.copy())

    COM = strucStart.get_center_of_mass()

    dis = strucRattle.get_all_distances()
    mindis = 0
    mindisAtom = 10
    
    while mindis < 1 or mindisAtom > 3:
        # First load original positions
        strucRattle.positions = strucStart.positions.copy()
        
        # Then Rattle within a circle 
        r = distRange * np.sqrt(rd.random())        
        theta = np.random.uniform(low = 0, high = 2 * np.pi)
        coordinates[coordinate] = COM + r * np.array([np.cos(theta), np.sin(theta), 0]) # CENTER HERE
        strucRattle.positions = coordinates
        dis = strucRattle.get_all_distances()
        mindis = np.min(dis[np.nonzero(dis)])
        mindisAtom = np.min(strucRattle.get_distances(coordinate, np.delete(np.arange(atoms), coordinate)))

        # If it does not fit, try to wiggle it into place using small circle 
        if mindis < 1:
            for i in range(10):
                r = 0.5 * np.sqrt(rd.random())
                theta = np.random.uniform(low = 0, high = 2 * np.pi)
                coordinates[coordinate] += r * np.array([np.cos(theta), np.sin(theta), 0])
                strucRattle.positions = coordinates
                dis = strucRattle.get_all_distances()
                mindis = np.min(dis[np.nonzero(dis)])
                mindisAtom = np.min(strucRattle.get_distances(coordinate, np.delete(np.arange(atoms), coordinate)))

                # If it works break
                if mindis > 1:
                    break

                # Otherwise reset coordinate 
                else:
                    coordinates[coordinate] -= r * np.array([np.cos(theta), np.sin(theta), 0])

    return coordinates[coordinate].copy()

def crossValidationSearch(X, Y, folds, clusters):
    """
    Perform cross validation to find the optimal number of clusters

    Parameters
    ----------
    X : feature vectors in a numpy array of shape (atoms * sets, dim) 
        Here sets are the number of training sets and dim is the dimension of the feature vector

    Y : energies of the structures in a numpy array of shape (sets) 

    folds : the number of folds in the k-fold cross validation

    clusters : list of clusters considered in the cross validation


    Returns
    -------
    optCluster : the optimal number of clusters 

    error : the test error using the optimal number of clusters 

    """

    N = Y.size
    atoms = X.shape[0] / N
    dim = X.shape[1]

    # Shuffle the data 
    t1 = time.time()    
    permutation = np.random.permutation(N)
    X = X.reshape(N, atoms * dim)
    X = X[permutation]
    X_divided = np.array_split(X, folds, axis = 0)
    
    Y = Y[permutation]
    Y_divided = np.array_split(Y, folds, axis = 0)

    ErrorAvg = [0] * len(clusters)

    for i, cluster in enumerate(clusters):
        Error = 0
        for k in range(folds):
            XTest = X_divided[k]
            XTest = XTest.reshape(atoms * XTest.shape[0], dim)
            YTest = Y_divided[k]

            XTrain = np.vstack([X for idx, X in enumerate(X_divided) if idx != k])
            XTrain = XTrain.reshape(atoms * XTrain.shape[0], dim)
            YTrain = np.hstack([Y for idx, Y in enumerate(Y_divided) if idx != k])

            # Create clusters and regression model
            clf, kmeans = em.createEnergyModelFeat(atoms, XTrain, YTrain, cluster)

            # Create cluster matrix for test set
            cMatrix = fc.createCmatrixF(XTest, kmeans, atoms)

            # Predict energy
            Epred = em.getEnergy(cMatrix, clf)

            # Calculate error (MAE)
            Error += np.sum(abs(Epred - YTest)) / YTest.size

        ErrorAvg[i] = Error / folds

    t2 = time.time()
    print('Cross validation took ', t2 - t1, ' seconds')
    
    optCluster_idx = np.argmin(ErrorAvg)
    optCluster = clusters[optCluster_idx]
    
    return optCluster, ErrorAvg[optCluster_idx]

def createInitalStructure():
    '''
    Creates an initial structure of 24 Carbon atoms 
    '''
    
    number_type1 = 6 # Carbon
    number_opt1 = 24 # number of atoms
    atom_numbers = number_opt1 * [number_type1]

    cell = np.array([[24, 0, 0],
                     [0, 24, 0],
                     [0, 0, 18]])
    pbc = [False, False, False]

    template = Atoms('')
    template.set_cell(cell)
    template.set_pbc(pbc)
    # define the volume in which the adsorbed cluster is optimized
    # the volume is defined by a a center position (p0)
    # and three spanning vectors 

    a = np.array((4.5, 0., 0.))
    b = np.array((0, 4.5, 0))
    z = np.array((0, 0, 1.5))
    p0 = np.array((0., 0., 9-0.75))
    center = np.array((11.5, 11.5))

    # define the closest distance two atoms of a given species can be to each other
    cd = closest_distances_generator(atom_numbers=atom_numbers,
                                     ratio_of_covalent_radii=0.7)

    # create the start structure
    sg = StartGenerator(slab = template,
                        atom_numbers = atom_numbers,
                        closest_allowed_distances = cd,
                        box_to_place_in = [p0, [a, b, z], center],
                        elliptic = True,
                        cluster = False)

    structure = sg.get_new_candidate()
    return structure


def relax(structure, i, ranks):
    '''
    Relax a structure and saves the trajectory based in the index i

    Parameters
    ----------
    structure : ase Atoms object to be relaxed

    i : index which the trajectory is saved under

    ranks: ranks of the processors to relax the structure 

    Returns
    -------
    structure : relaxed Atoms object
    '''

    label = 'relax' + str(i) + '_' + str(ranks[0])
    
    # Create calculator
    calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),
              mode = 'lcao',
              basis = 'dzp',
              xc='PBE',
              gpts = h2gpts(0.2, structure.get_cell(), idiv = 8),
              occupations=FermiDirac(0.1),
              maxiter=99,
              mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
              nbands=-50,
              kpts=(1,1,1),
              communicator=ranks,
              txt = label+ '_lcao.txt')

    # Set calculator 
    structure.set_calculator(calc)

    # loop a number of times to capture if minimization stops with high force
    # due to the VariansBreak calls
    forcemax = 0.1
    niter = 0

    # If the structure is already fully relaxed just return it
    if (structure.get_forces()**2).sum(axis = 1).max()**0.5 < forcemax:
        return structure
    
    traj = Trajectory(label+'_lcao.traj', 'w', structure,
                      master=world.rank==ranks[0])
    
    while (structure.get_forces()**2).sum(axis = 1).max()**0.5 > forcemax and niter < 10:
        dyn = BFGS(structure,
                   logfile=label+'.log')
        vb = VariansBreak(structure, dyn, min_stdev = 0.01, N = 15)
        dyn.attach(vb)
        dyn.attach(traj)
        dyn.run(fmax = forcemax, steps = 500)
        niter += 1

    return structure


def checkProlong(nmaster, master):
    '''
    Check if there are trajectory files present in the folder
    If so, then contenue the search based on these files
    
    Parameters
    ----------
    
    nmaster : string of the master core running the simulation

    master : Boolean describing whether this process is the master or not  
    '''

    i = 0

    myCoordinateSet = cs.CoordinateSet()

    # Try to read files (Maybe check for duplicates?)
    try:
        featVecListTotal = np.zeros((24 * 4, 13)) # Hardcoded for the moment 
        energyListTotal = np.array([0., 0., 0., 0.])
        while True:
            structure = Trajectory('relax0_' + nmaster + '_lcao.traj')[-1]
            myCoordinateSet.Coordinates = structure.positions
            myCoordinateSet.calculateFeatures()
            atoms, dim = myCoordinateSet.FeatureVectors.shape
            for idx, nmasterAll in enumerate([(world.size // 4 ) * x for x in range(4)]): # FIX FOR CORRECT CORES
                structure = Trajectory('relax' + str(i) + '_' + str(nmasterAll) + '_lcao.traj')[-1]
                myCoordinateSet.Coordinates = structure.positions
                myCoordinateSet.calculateFeatures()
                if i == 0:
                    featVecListTotal[idx * atoms:(idx + 1) * atoms, :] = myCoordinateSet.FeatureVectors.copy()
                    energyListTotal[idx] = np.array([structure.get_potential_energy()])
                else:
                    featVecListTotal = np.append(featVecListTotal, myCoordinateSet.FeatureVectors.copy(), axis = 0)
                    energyListTotal = np.append(energyListTotal, structure.get_potential_energy())
            i += 1
    except IOError:
        if i == 0:
            return 1, None, None, None
        atoms, dim = myCoordinateSet.FeatureVectors.shape
        
        # Remove the latest structure from the energy model, since it might not have been fully relaxed
        for _ in range(4):
            energyListTotal = np.delete(energyListTotal, -1)
        for _ in range(atoms * 4):
            featVecListTotal = np.delete(featVecListTotal, -1, axis = 0)
            
        # Now we need to backtrack to find the structure we are currently rattling
        j = i - 1
        while True:
            try:
                oldStruc = Trajectory('preRattle' + str(j) + '_' + nmaster +  '.traj')[0]
                oldStruc.get_potential_energy() # We only save energies for structures we accept as new starting points
                return i, featVecListTotal, energyListTotal, j
            except PropertyNotImplementedError:
                j -= 1

                
def monteCarloSearch(atoms, N, T, distRange, clusters, rattlePercentage, ranks, cv = False):
    '''
    Runs the Monte Carlo search based on local energies from linear regression 
    
    Parameters
    ----------
    
    atoms : number of atoms

    N : number of MC iterations

    distRange : range of the rattle parameter 

    clusters : number of clusters used in the regression model 

    rattlePercentage : percentage of atoms to be rattled in earch iterations

    ranks : ranks of all the cores running this specific search 

    Returns
    -------
    
    Eback : energy of the structure currently being rattled 

    '''

    # Set up cores 
    master = world.rank == ranks[0]
    nmaster = str(ranks[0])
    
    # Create comparator for deciding whether to add new structures to regression or not
    comp = InteratomicDistanceComparator(pair_cor_cum_diff = 0.02, pair_cor_max = 0.7, dE = 0.1)

    # Empthy arrays for distribution
    positions = np.empty((atoms, 3))
    bestEnergy = np.empty((1, 1))

    # MC settings
    nRattle = int(atoms * rattlePercentage)

    # Check if files are already present in the folder
    start, featVecList, energyList, oldStruc_idx = checkProlong(nmaster, master)

    # Create inital structure if we are just starting
    if start == 1: # Assume we have not failed during first relaxataion iteration!!!! 
        structure = createInitalStructure()
        structure.set_scaled_positions(structure.get_scaled_positions())

    # Else read previous structure
    else:
        structure = Trajectory('relax' + str(start - 1) + '_' + nmaster + '_lcao.traj')[-1]

    # Then broadcast the structure to all cores 
    if master:
        positions = structure.positions
    world.broadcast(positions, ranks[0])
    structure.positions = positions
    world.barrier()

    # Relax structure
    structure = relax(structure, start - 1, ranks) # Relax initial structure
    if start == 1:
        preRattle = Trajectory('preRattle0_' + nmaster +  '.traj', 'w', master=master)
        preRattle.write(structure)
    
    # Save trajectory file of all structures
    world.barrier()
    allStructures = Trajectory('allStructures_' + nmaster + '.traj', 'a', master=master)
    allStructures.write(structure)
    world.barrier()

    # Now read the structure we started from during this relaxation
    if oldStruc_idx is not None:
        oldStruc = Trajectory('preRattle' + str(oldStruc_idx) + '_' + nmaster + '.traj')[0]
        oldCoords = oldStruc.positions.copy()
        oldEnergy = oldStruc.get_potential_energy()

    world.barrier()

    # Now create a CoordinateSet object for containing the preovious structure and calculating feature vectors 
    if master:
        myCoordinateSet = cs.CoordinateSet()
        myCoordinateSet.Coordinates = structure.positions 
        myCoordinateSet.calculateFeatures()
        myCoordinateSet.Energy = structure.get_potential_energy()

        # If we are starting a new run 
        if featVecList is None:
            bestEnergy[0] = myCoordinateSet.Energy
            featVecList = myCoordinateSet.FeatureVectors.copy()
            energyList = np.array([myCoordinateSet.Energy])

        # We are continueing a run
        else:
            featVecList = np.append(featVecList, myCoordinateSet.FeatureVectors.copy(), axis = 0)
            energyList = np.append(energyList, myCoordinateSet.Energy)
            
            bestEnergy[0] = np.min(energyList)

            # Now we must check if the newly relaxed structure should be should be our new starting point
            E2 = structure.get_potential_energy() 
            E1 = oldEnergy # Energy of previous structure 
            dE = E2 - E1

            # If new energy is the best energy save it
            if E2 < bestEnergy[0]: 
                bestEnergy[0] = E2 

            # Metropolis criterion for acceptance 
            if dE > 0:
                if np.exp(-dE / T) < np.random.random():
                    structure.positions = oldCoords.copy()
                    myCoordinateSet.Coordinates = oldCoords.copy()
                    myCoordinateSet.Energy = oldEnergy
                else:
                    print('Iteration {0}_{1} \t using newly relaxed start structure').format(start, nmaster)
            else:
                print('Iteration {0}_{1} \t using newly relaxed start structure').format(start, nmaster)
            

    # save structure before rattle
    preRattle = Trajectory('preRattle' + str(start) + '_' + nmaster + '.traj', 'w', master=master)
    preRattle.write(structure)       

    world.barrier()
    for i in range(start, N + start):
        if master:
            # Save coordinates if need to revert
            oldCoords = myCoordinateSet.Coordinates.copy()
            oldEnergy = myCoordinateSet.Energy
        
            # Create the machine learning model
            if atoms * energyList.shape[0] >= clusters:  # Check that we have enough feature vectors for clustering

                # Create energy model and save it
                if cv is True and i % 5 == 1 and i > 1:
                    clusterList = [10, 20, 40, 80]
                    optCluster, error = crossValidationSearch(featVecList, energyList, folds = 4, clusters = clusterList)
                    clusters = optCluster
                    print('Iteration {0} \t using {1} clusters with test error of {2}'.format(i, optCluster, error))
                    
                clf, kmeans = em.createEnergyModelFeat(atoms, featVecList, energyList, clusters)
                joblib.dump(clf, 'clf{0}_{1}.pkl'.format(i - 1, nmaster))

                # Save the current configuration
                vd.plotConfiguration(oldCoords.copy(), show = False, name = 'preRattle' + str(i) + '_' + nmaster , clf = clf, kmeans = kmeans)

                # Find clusters in current configuration
                myCoordinateSet.calculateFeatures()
                myCoordinateSet.calculateClusters(kmeans)
                np.save('cluster{0}_{1}.npy'.format(i - 1, nmaster), myCoordinateSet.clusterList)

                # Find best clusters
                clusterEnergy = clf.coef_
                clusterEnergyIndex = np.argsort(clusterEnergy) # First index is the most stable cluster 
                clusterEnergyIndex = np.flip(clusterEnergyIndex, axis = 0) # First index is the most unstable cluster

                # Rank atoms according to stability 
                stabilityList = np.zeros(atoms)
                for j in range(atoms):
                    stabilityList[j] = np.where(clusterEnergyIndex == myCoordinateSet.clusterList[j])[0][0]

                # Now sort according to stability and then randomly for equal elements in terms of stability
                randNumber = np.random.random(atoms)
                stabilityList = np.lexsort((randNumber, stabilityList)) 

                # If we want to rattle the nRattle most unstable atoms 
                # uAtoms = stabilityList[:nRattle].copy()

                # If we want to sample randomly from the 50% most unstable atoms
                # uAtoms = rd.sample(stabilityList[:int(atoms / 2)].tolist(), nRattle) # from 50% most unstable

                # If we want to sample randomly during start of the run, with greater rattle percentage
                #if i < 15:
                #    uAtoms = rd.sample(stabilityList[:int(atoms / 1)].tolist(), 5) # Rattle 5 atoms

                # Simply just take a random atom 
                # uAtoms = rd.sample(stabilityList[:int(atoms / 1)].tolist(), 1) # from 100% most unstable
            
                # If we want to rattle randomly with some probability
                '''
                r = rd.random()
                if i <= 20:
                    uAtoms = rd.sample(stabilityList[:int(atoms / 1)].tolist(), 1) # from 100% most unstable
#                    print('Iteration {0} \t 4 random atoms').format(i)    
                else:
                    if 0 <= r < 0.8:
                        uAtoms = rd.sample(stabilityList[:int(atoms / 1)].tolist(), 1) # from 100% most unstable
                       # uAtoms = stabilityList[:nRattle].copy()                        
#                        print('Iteration {0} \t random atom').format(i)    
                    if 0.8 <= r < 1:
#                        uAtoms = stabilityList[:4].copy()
                        uAtoms = rd.sample(stabilityList[:int(atoms / 1)].tolist(), 4) # from 100% most unstable
#                        print('Iteration {0} \t 4 random atoms').format(i)

                '''
                # If we want to sample from a probability distrubtion
                prob = [(1./2) ** X for X in range(1, 25)]
                prob = np.array(prob) / np.sum(prob)
                size = np.random.choice(np.arange(1, 25), p = prob) # How many atoms to rattle

                # Sample atoms randomly 
                # uAtoms = rd.sample(stabilityList.tolist(), size) # from 100% most unstable

                # Choose atoms with same probability distribution
                uAtoms = np.random.choice(stabilityList, size = size, replace = False, p = prob)
                
                # Choose atoms with same probability distribution
                # Find most unstable atoms in 50% most unstable clusters
                '''
                uAtoms = np.array([100000])
                for j in range(int(clusters / 2)):
                    index = clusterEnergyIndex[j]
                    if myCoordinateSet.Clusters[index] != 0:
                        uAtoms = np.where(myCoordinateSet.clusterList == index)[0]
                        break

          
                # Find more unstable clusters in 50% most unstable clusters
                uClusters = 1
                for k in range(j + 1, int(clusters / 2)):
                    if uClusters == 2: # Two clusters in total 
                        break
                    index = clusterEnergyIndex[k]
                    if myCoordinateSet.Clusters[index] != 0:
                        uAtoms = np.append(uAtoms, np.where(myCoordinateSet.clusterList == index)[0])
                        uClusters += 1

                # If no atoms in the most unstable clusters
                if uAtoms[0] == 100000:
                    uAtoms = rd.sample([h for h in range(atoms)], nRattle) # Sample atoms randomly
                '''

                # Find most unstable cluster
                '''
                for j in range(clusters):
                    index = clusterEnergyIndex[j]
                    if myCoordinateSet.Clusters[index] != 0:
                        uAtoms = np.where(myCoordinateSet.clusterList == index)[0]
                        break
                '''
            
                # Find more unstable clusters
                '''
                uClusters = 1
                for k in range(j + 1, clusters):
                    if uClusters == 2: # Two clusters in total 
                        break
                    index = clusterEnergyIndex[k]
                    if myCoordinateSet.Clusters[index] != 0:
                        uAtoms = np.append(uAtoms, np.where(myCoordinateSet.clusterList == index)[0])
                        uClusters += 1             
                '''
                
                # Rattle the unstable atoms 
                for atom in uAtoms:
                    myCoordinateSet.Coordinates[atom] = rattleAtomFromCenter(atom, myCoordinateSet.Coordinates.copy(), distRange)

            else: # Normal rattle if not enough data for model yet
                nRattleAtoms = int(atoms * rattlePercentage)
                indexRattleAtoms = rd.sample(range(atoms), nRattleAtoms)
                print('Not enough data for ML model')
                for atom in indexRattleAtoms:
                    myCoordinateSet.Coordinates[atom] = rattleAtomFromCenter(atom, myCoordinateSet.Coordinates.copy(), distRange)

        # Broadcast positions if we reverted in last iteration 
        if master:
            positions = structure.positions
        world.broadcast(positions, ranks[0])
        structure.positions = positions
        
        # Save structure before rattle
        if i > start:
            preRattle = Trajectory('preRattle' + str(i) + '_' + nmaster + '.traj', 'w', master=master)
            preRattle.write(structure)

        # Update structure with rattled coordinates 
        if master:
            structure.positions = myCoordinateSet.Coordinates.copy()
            structure.set_scaled_positions(structure.get_scaled_positions())        
            positions = structure.positions
        world.broadcast(positions, ranks[0])
        structure.positions = positions
        
        # Save the rattled structure 
        preRelax = Trajectory('preRelax' + str(i) + '_' + nmaster + '.traj', 'w', master=master)
        preRelax.write(structure)

        # Then relax the strucuture
        world.barrier()
        structure = relax(structure, i, ranks)

        # Write the new structure to the file containing all the seen structures
        allStructures.write(structure)
        world.barrier()

        # Now check if the structure should be added to the regression pool
        if master:
            all_structures = Trajectory('allStructures_' + nmaster + '.traj')
            myCoordinateSet.Coordinates = structure.positions.copy()
            myCoordinateSet.Energy = structure.get_potential_energy()
            myCoordinateSet.calculateFeatures()

            # Compare energies with new and old structure
            E2 = structure.get_potential_energy() # Energy of new relaxed structure
            E1 = oldEnergy # Energy of previous structure 
            dE = E2 - E1

            # Now potentially add new structure to regression
            duplicate = False
            for s in range(len(all_structures) - 1):
                struc = all_structures[s]
                if comp.looks_like(structure, struc):
                    print('Iteration {0}_{1} \t already seen structure'.format(i, nmaster))
                    duplicate = True
                    break # We have already seen the structure

            if duplicate is False:
                if energyList.shape[0] == 100000: # If we have exactly 100000 structures check if the new one is better
                    highestEnergy_idx = np.argmax(energyList)
                    if energyList[highestEnergy_idx] > E2:
                        print('Iteration {0} \t replacing structure in regression pool'.format(i))
                        energyList[highestEnergy_idx] = E2
                        featVecList[highestEnergy_idx * atoms : (highestEnergy_idx + 1) * atoms, :] = myCoordinateSet.FeatureVectors.copy()
                    else:
                        print('Iteration {0} \t new structure not good enough to enter regression pool'.format(i))    
                else:
#                    print('Iteration {0} \t adding structure to regression'.format(i))
                    featVecList = np.append(featVecList, myCoordinateSet.FeatureVectors.copy(), axis = 0)
                    energyList = np.append(energyList, E2)            

            # If new energy is the best energy save it
            if E2 < bestEnergy[0]: 
                bestEnergy[0] = E2

            if dE > 0:
                if np.exp(-dE / T) < np.random.random():
                    print('Iteration {0}_{1} \t Disregarding new structure').format(i,nmaster) 
                    structure.positions = oldCoords.copy()
                    myCoordinateSet.Coordinates = oldCoords.copy()
                    myCoordinateSet.Energy = oldEnergy
                else:
                    print('Iteration {0}_{1} \t Accepting new structure').format(i, nmaster) 
            else:
                print('Iteration {0}_{1} \t Accepting new structure').format(i, nmaster)
                    
        world.barrier()
        world.broadcast(bestEnergy, ranks[0])
        
        # Check if we have found minimum (second term is error)
        #if bestEnergy[0] < -188.48598307896674 + 0.01:
        #    break
        
        world.barrier()
    Eback = np.empty(1, dtype=float)

    # Now update the energy
    if master:
        if bestEnergy[0] < -188.48598307896674 + 0.01:
            Eback[:] = bestEnergy[0]
        else:
            Eback[:] = myCoordinateSet.Energy
    world.broadcast(Eback, ranks[0])
    world.barrier()
    return Eback[0]


### SETUP FOR THE ALGORITHM ### 

atoms = 24
num = 4 # number parralel runs
mc_iter = 5 # Number of MC iterations between temperature check
N = 40 # How many temperature checks we should do 
distRange, clusters, rattlePercentage = 4, 10, 0.05
cv = False

# Temperature settings
#Ts = np.array([0.2, 0.342, 0.585, 1.])
Ts = np.array([0.2, 0.293, 0.425, 0.62])
Tstart = Ts.copy()
Es = np.zeros(Ts.shape)

# CPU's to each MC run
n = world.size // num 
assert num * n == world.size
masters = [n * x for x in range(num)]

# Statistics
finished = False
swapAttemps = 0.
succesSwap = 0.

tempMatrix = np.zeros((N + 1, Tstart.shape[0]))
tempMatrix[0] = Tstart.copy()
swapMatrix = np.zeros(num - 1)
energyMatrix = np.zeros((N, Tstart.shape[0]))

if world.rank == 0:
    np.save('tempMatrix.npy', tempMatrix)
    np.save('swapMatrix.npy', swapMatrix)
    np.save('energyMatrix.npy', energyMatrix)

# Now run the parallel tempering algorithm 
for a in range(N):
    if world.rank == 0:
        print('Current temperatures are', tempMatrix[a])
        print('Current energies are', Es)
    for j, T in enumerate(Ts):
        ranks = np.arange(j * n, (j + 1) * n)
        if world.rank in ranks:
            Es[j] = monteCarloSearch(atoms, mc_iter, T, distRange, clusters, rattlePercentage, ranks, cv = cv)

    world.barrier()
    data = []
    world.barrier()

    if world.rank == 0:
        # Gather parts from the slaves
        data[0:num] = Es
        buf = np.empty(num, dtype=float)
        for i, master in enumerate(masters[1:]):
            world.receive(buf, master, tag=123)
            data[num * (i + 1): num*(i+2)] = buf
    elif world.rank in masters:
        # Send to the master
        world.send(Es, 0, tag=123)

    if world.rank == 0:
        for i in range(num):
            Es[i] = data[i * num + i]

    # Make sure that world indeed is all ranks and broadcast from master to all slaves 
    comm = world.new_communicator(np.array(range(world.size)))
    comm.barrier()
    comm.broadcast(Es, 0)

    # Attempt temperature swaps 
    for k in range(num - 1):
        Tk = Tstart[k]
        Tk1 = Tstart[k + 1]
        h = np.where(Ts==Tk)[0][0]
        h1 = np.where(Ts==Tk1)[0][0]
        Ek = Es[h]
        Ek1 = Es[h1]
        
        kB = 8.6173303e-05 # eV/K
        kB = 1 # T is Temperature * kB 
        if Ek > Ek1:
            A = 1.
        else:
            A = np.exp(1 / kB * (1 / Tk - 1 / Tk1) * (Ek - Ek1))

        ran = np.empty(1, dtype=float)
        if world.rank == 0:
            ran[0] = np.random.random()
        comm.broadcast(ran, 0)
        comm.barrier()

        if A > ran[0]:
            Ts[h] = Tk1
            Ts[h1] = Tk
            swapAttemps += 1
            succesSwap += 1
            swapMatrix[k] += 1
        else:
            swapAttemps += 1

    tempMatrix[a + 1] = Ts.copy()
    energyMatrix[a] = Es.copy()

    for j, _ in enumerate(Ts):
        if Es[j] < -188.48:
            finished = True     
    
    if world.rank == 0:
        print('Swap', a, succesSwap / swapAttemps, 'succes')
        print('Current swap matrix is', swapMatrix)
        print('Total swap attemps are', swapAttemps)
        np.save('tempMatrix.npy', tempMatrix)
        np.save('energyMatrix.npy', energyMatrix)
        np.save('swapMatrix.npy', swapMatrix)

    world.barrier()
    if finished is True:
        break     
    
if world.rank == 0:
    print('Temperature matrix is', tempMatrix)
    print('Energy matrix is', energyMatrix)
    print('Swap matrix is', swapMatrix)



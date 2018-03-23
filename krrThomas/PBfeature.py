import numpy as np
import math as math
from scipy.spatial import distance, distance_matrix

def EuclidianDistance(xData,nDim):
        nData = xData.shape[0]
        Y = distance.pdist(xData, 'euclidean')
        tri = np.zeros((nData,nData))
        tri[np.triu_indices(nData,1)] = Y
        R = tri + np.transpose(tri)
        return R

def CutoffFunction(Rij,Rc):
        if Rij <= Rc:
                return 0.5*(1+math.cos(math.pi*Rij/Rc))
        else:
                return 0
	
def CutoffGrad(Rij,Rc):
	if Rij <= Rc:
		dfCdr = -0.5 * math.sin(math.pi*Rij/Rc) * math.pi/Rc
	else:
		dfCdr = 0  # undefined
	return dfCdr	

def LocalFeatureVectorComponentI(R,Rs,Rc,eta):  # for pairs
	sizeR = R.shape
	n = sizeR[0]
	m = sizeR[1]
	f = np.zeros(n)	
	for i in range(0,n):
		fi = 0
		for j in range(0,m):
			Rij = R[i,j]
			if j!=i:
				fi += math.exp(-eta*pow(Rij - Rs,2)/pow(Rc,2))*CutoffFunction(Rij,Rc)	
		f[i] = fi
	return f

def LocalFeatureVectorComponentII(positions,R,Rc,eta,lambd,zeta):  # for triples
	n = R.shape[0]
	f = np.zeros(n)	
	for i in range(0,n):
		fi = 0
		for j in range(0,n):
			for k in range(0,n):
				if j!=i and k!=i and k>j:
					RijVec = positions[j,:] - positions[i,:]
					RikVec = positions[k,:] - positions[i,:]
					Rij = R[i,j]
					Rik = R[i,k]
					Rjk = R[j,k]
					cosTheta = np.inner(RijVec,RikVec)/(Rij*Rik)
					fi1 = (1 + lambd*cosTheta)**zeta
					fi2 = math.exp(-eta * (Rij**2 + Rik**2 + Rjk**2)/Rc**2)
					fi3 = CutoffFunction(Rij,Rc) * CutoffFunction(Rik,Rc) * CutoffFunction(Rjk,Rc)
					fi += 2.0**(1-zeta)	* fi1 * fi2 * fi3

		f[i] = fi
	return f

def LocalFeatureVector(positions,etaVector,lambdaVector,zetaVector,R,Rs,Rc):
	sizeR = R.shape
	n = sizeR[0]
	fI = np.zeros((len(etaVector),n))
	fII = np.zeros((len(lambdaVector)*len(zetaVector),n))

	for i, eta in enumerate(etaVector):
		fI[i,:] = LocalFeatureVectorComponentI(R,Rs,Rc,eta)

	N = -1
	eta = np.min(etaVector)
	for j, zeta in enumerate(zetaVector):
		for k, lambd in enumerate(lambdaVector):
			N += 1
			fII[N,:] = LocalFeatureVectorComponentII(positions,R,Rc,eta,lambd,zeta)


	f = np.append(fI,fII,axis = 0)

	return np.transpose(fI),np.transpose(fII),np.transpose(f)
	
def JacobianBehlers(positions,etaVector,lambdaVector,zetaVector,R,Rs,Rc,atomIndex):
	nAtoms = positions.shape[0]
	nDim = positions.shape[1]
	i = atomIndex
	Ji = np.zeros((len(lambdaVector)*len(zetaVector) + len(etaVector), nAtoms*nDim))
	for k, eta in enumerate(etaVector):
		dfiIdxi = 0
		dfiIdx = np.zeros((nAtoms,nDim))
		for j in range(nAtoms):
			if j!=i:
				Rij = R[i,j]
				RijVec = positions[j,:] - positions[i,:]
				dfiIdxj = RijVec/Rij * np.exp(-eta*(Rij-Rs)**2/Rc**2) * ( -2*eta/Rc**2 * (Rij-Rs) * CutoffFunction(Rij,Rc) +  CutoffGrad(Rij,Rc) )
				dfiIdx[j,:] = dfiIdxj
				dfiIdxi += (-1)*dfiIdxj
		dfiIdx[i,:] = dfiIdxi
		dfiIdx = np.reshape(dfiIdx,nAtoms*nDim)
		Ji[k,:] = dfiIdx

	N = -1
	eta = np.min(etaVector)
	for m, zeta in enumerate(zetaVector):
		for n, lambd in enumerate(lambdaVector):
			N = N + 1
			dfiIIdxi = 0
			dFii = np.zeros((nAtoms,nDim))
			dfii = np.zeros(nDim)
			for j in range(nAtoms):					
				dfij = np.zeros(nDim)
				for k in range(nAtoms):
					if j!=k and k!=i and j!=i:
						Rij = R[i,j]
						Rik = R[i,k]
						Rjk = R[j,k]
						RijVec = positions[j,:] - positions[i,:]
						RikVec = positions[k,:] - positions[i,:]
						RjkVec = positions[k,:] - positions[j,:]
						RijkDot = np.inner(RijVec,RikVec)/(Rij*Rik)

						dfiFirstPartjInner = - RijkDot * RijVec/Rij**2                   + RikVec/(Rij*Rik)  # vector of dimension 3
						dfiFirstPartiInner =   RijkDot * (RijVec/Rij**2 + RikVec/Rik**2) - (RikVec + RijVec)/(Rij*Rik)
						dfiFirstPartj = zeta * math.pow(1 + lambd * RijkDot ,zeta-1) * lambd*dfiFirstPartjInner
						dfiFirstParti = zeta * math.pow(1 + lambd * RijkDot ,zeta-1) * lambd*dfiFirstPartiInner

						dfiSecondPartj = 2*eta/Rc**2 * math.exp(-eta*(Rij**2 + Rik**2 + Rjk**2)/Rc**2) * (RjkVec - RijVec)
						dfiSecondParti = 2*eta/Rc**2 * math.exp(-eta*(Rij**2 + Rik**2 + Rjk**2)/Rc**2) * (RikVec + RijVec)

						dfiThirdPartj = (CutoffFunction(Rik,Rc) * (CutoffGrad(Rij,Rc)*CutoffFunction(Rjk,Rc)*RijVec/Rij
							 - CutoffGrad(Rjk,Rc)*CutoffFunction(Rij,Rc)*RjkVec/Rjk))
						dfiThirdParti = (-CutoffFunction(Rjk,Rc) * (CutoffGrad(Rij,Rc)*CutoffFunction(Rik,Rc)*RijVec/Rij
							 + CutoffGrad(Rik,Rc)*CutoffFunction(Rij,Rc)*RikVec/Rik))

						firstPart = math.pow(1 + lambd * RijkDot, zeta)
						secondPart = math.exp(-eta*(Rij**2 + Rik**2 + Rjk**2)/Rc**2)
						thirdPart = CutoffFunction(Rij,Rc) * CutoffFunction(Rik,Rc) * CutoffFunction(Rjk,Rc)

						dfii += math.pow(2,1-zeta)*(dfiFirstParti*secondPart*thirdPart + firstPart*dfiSecondParti*thirdPart + firstPart*secondPart*dfiThirdParti)  #  to be summed over k and j
						dfij += math.pow(2,1-zeta)*(dfiFirstPartj*secondPart*thirdPart + firstPart*dfiSecondPartj*thirdPart + firstPart*secondPart*dfiThirdPartj) #  to be summed over k and j
				dFii[j,:] = 2 * dfij # Rjk = Rkj and RijkDot = RikjDot --> xj (=xk) occurs in both Rjk and Rkj
			dFii[i,:] = dfii
			dFii = np.reshape(dFii,nAtoms*nDim)
			Ji[len(etaVector) + N,:] = dFii/2

					
	# one Jacobian per atomFeatureVector Ji. axis=0 are the different feature vector 
	# coordinates len(etaVector)*len(lambdaVector)*len(zetaVector) + len(etaVector), 
	# axis=1 are the cartesian coordinates nAtoms*nDimension
	return Ji

def NumGradient(positions,etaVector,lambdaVector,zetaVector,Rs,Rc,atomIndex):
	nAtoms = positions.shape[0]
	nDim = positions.shape[1]
	featureDim = len(lambdaVector)*len(zetaVector) + len(etaVector)
	dx = .00001
	R = EuclidianDistance(positions,nDim)
	fI,fII,f = LocalFeatureVector(positions,etaVector,lambdaVector,zetaVector,R,Rs,Rc)
	fi = f[atomIndex,:]
	Ji = np.zeros((len(lambdaVector)*len(zetaVector) + len(etaVector), nAtoms*nDim))
	for i in range(0,nAtoms):
		for j in range(nDim):
			transPositions = np.copy(positions)
			transPositions[i,j] = transPositions[i,j]+dx
			transR = EuclidianDistance(transPositions,nDim)
			transfI,transfII,transf = LocalFeatureVector(transPositions,etaVector,lambdaVector,zetaVector,transR,Rs,Rc)
			transfi = transf[atomIndex,:]
			df = (transfi - fi)/dx			
			Ji[:,i*nDim+j] = df
	return Ji

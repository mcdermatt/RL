import numpy as np

def sweep(DOF = 3, fidelity = 3):
	fidelity = fidelity #number of discrete values to test for each parameter
	dof = DOF
	numBandits = fidelity**dof
	bandits = np.zeros([numBandits,dof])

	for d in range(dof):
		for i in range(fidelity**(d+1)):
			bandits[(fidelity**(dof - d - 1))*i:(fidelity**(dof - d - 1))*(i+1),d] = i % fidelity
	return bandits/fidelity

if __name__ == "__main__":
	keep_em_clean = sweep(4,3)
	print(keep_em_clean)
	
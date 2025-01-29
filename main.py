import numpy as np
import tqdm

class sde_solver():

    def __init__(self,a_drif=None,a_diff=None,b_drif=None,b_diff=None,a_init=None,b_init=None):

        self.a_drif = a_drif
        self.a_diff = a_diff
        self.b_drif = b_drif
        self.b_diff = b_diff

        self.a_init = a_init
        self.b_init = b_init

    def ito_solver(self,t_tot=1,t_step=0.01,n_simulations=1000):
        # from 0 to t, n_steps
        n_steps = int(t_tot/t_step)+1
        times = np.linspace(0, t_tot, n_steps)

        a_trajs = np.zeros((n_simulations,n_steps),dtype=complex)
        b_trajs = np.zeros((n_simulations,n_steps),dtype=complex)

        a_trajs[:,0] = self.a_init
        b_trajs[:,0] = self.b_init

        # Euler-Maruyama method for multiple simulations
        for i in tqdm.tqdm(range(n_simulations)):

            dW1 = np.random.normal(0,np.sqrt(t_step),size=n_steps)
            dW2 = np.random.normal(0,np.sqrt(t_step),size=n_steps)

            for t in range(1, n_steps):
                a_trajs[i][t] = a_trajs[i][t-1] + self.a_drif(a_trajs[i][t-1],b_trajs[i][t-1]) * t_step + self.a_diff(a_trajs[i][t-1],b_trajs[i][t-1]) * dW1[t]
                b_trajs[i][t] = b_trajs[i][t-1] + self.b_drif(a_trajs[i][t-1],b_trajs[i][t-1]) * t_step + self.b_diff(a_trajs[i][t-1],b_trajs[i][t-1]) * dW2[t]

        self.times = times
        self.a_trajs = a_trajs
        self.b_trajs = b_trajs

        self.n_simulations = n_simulations
        
        return [times, a_trajs, b_trajs]
    
    def observables(self):
        # Calculate the average photon number
        photon_number = np.sum(self.a_trajs*self.b_trajs,axis=0) / self.n_simulations
        self.photon_number = photon_number

        # Calculate the alpha in coherent states
        coherent_alpha = np.sqrt(np.sum(self.a_trajs * self.a_trajs, axis=0) / self.n_simulations)
        self.coherent_alpha = coherent_alpha

        parity = np.sum(np.exp(-2*self.a_trajs*self.b_trajs),axis=0) / self.n_simulations
        self.parity = parity
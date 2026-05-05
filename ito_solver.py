import numpy as np
import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ito_sde_solver():

    def __init__(self,a_drif=None,a_diff=None,b_drif=None,b_diff=None,drift_gauge=None,a_init=None,b_init=None,n_mode=1):

        self.a_drif = a_drif
        self.a_diff = a_diff
        self.b_drif = b_drif
        self.b_diff = b_diff

        self.a_init = a_init
        self.b_init = b_init

        self.drift_gauge = drift_gauge
        self.n_mode = n_mode

    def run(self,t_tot=1,t_step=0.01,n_simulations=1000,save_matrices=False,progress_bar=True):

        n_mode = self.n_mode
        # from 0 to t, n_steps
        n_steps = int(t_tot/t_step)+1
        times = np.linspace(0, t_tot, n_steps)

        n_simulations = int(n_simulations)

        a_trajs = np.zeros((n_simulations,n_mode,n_steps),dtype=complex)
        b_trajs = np.zeros((n_simulations,n_mode,n_steps),dtype=complex)

        a_trajs[:,:,0] = self.a_init
        b_trajs[:,:,0] = self.b_init
        
        if self.drift_gauge != None:
            gauge_Omega = np.zeros((n_simulations,n_mode,n_steps),dtype=complex)
            gauge_Omega[:,:,0] = 1.

        if save_matrices:

            mat_A_0_list = np.zeros((n_simulations,n_mode,n_steps),dtype=complex)
            mat_B_00_list = np.zeros((n_simulations,n_mode,n_steps),dtype=complex)
            mat_A_1_list = np.zeros((n_simulations,n_mode,n_steps),dtype=complex)
            mat_B_11_list = np.zeros((n_simulations,n_mode,n_steps),dtype=complex)


        segment_length = 10000
        a_trajs_divide = [a_trajs[i:i + segment_length] for i in range(0, a_trajs.shape[0], segment_length)]
        #b_trajs_divide = [b_trajs[i:i + segment_length] for i in range(0, b_trajs.shape[0], segment_length)]
        
        segments_length_list = [segment.shape[0] for segment in a_trajs_divide]
        segments_length = len(a_trajs_divide)

        index_start = 0
        index_stop = segments_length_list[0]

        for i in range(segments_length):

            length = segments_length_list[i]

            #a_trajs[index_start:index_stop,:,0] = a_trajs_divide[i]
            #b_trajs[index_start:index_stop,:,0] = b_trajs_divide[i]

            dW1 = np.random.normal(0,np.sqrt(t_step),size=length*n_mode*n_steps)
            dW1 = dW1.reshape(length,n_mode,n_steps)

            dW2 = np.random.normal(0,np.sqrt(t_step),size=length*n_mode*n_steps)
            dW2 = dW2.reshape(length,n_mode,n_steps)


            a_trajs_sub = a_trajs[index_start:index_stop,:,:]
            b_trajs_sub = b_trajs[index_start:index_stop,:,:]
            
            if self.drift_gauge != None:
                gauge_Omega_sub = gauge_Omega[index_start:index_stop,:,:]

            for t in tqdm.tqdm(range(1, n_steps),disable=not progress_bar):

                matA_0 = self.a_drif(a_trajs_sub[:,:,t-1],b_trajs_sub[:,:,t-1])
                matB_00 = self.a_diff(a_trajs_sub[:,:,t-1],b_trajs_sub[:,:,t-1])

                matA_1 = self.b_drif(a_trajs_sub[:,:,t-1],b_trajs_sub[:,:,t-1])
                matB_11 = self.b_diff(a_trajs_sub[:,:,t-1],b_trajs_sub[:,:,t-1])
                
                if self.drift_gauge != None:
                    
                    drift_gauge_value_0 = self.drift_gauge[0](a_trajs_sub[:,:,t-1],b_trajs_sub[:,:,t-1])
                    drift_gauge_value_1 = self.drift_gauge[1](a_trajs_sub[:,:,t-1],b_trajs_sub[:,:,t-1])
                    matA_0 += - drift_gauge_value_0 * matB_00
                    matA_1 += - drift_gauge_value_1 * matB_11

                    gauge_Omega_sub[:,:,t] = gauge_Omega_sub[:,:,t-1] + gauge_Omega_sub[:,:,t-1]*(drift_gauge_value_0 * dW1[:,:,t] + drift_gauge_value_1 * dW2[:,:,t])

                a_trajs_sub[:,:,t] = a_trajs_sub[:,:,t-1] + matA_0 * t_step + matB_00 * dW1[:,:,t]
                b_trajs_sub[:,:,t] = b_trajs_sub[:,:,t-1] + matA_1 * t_step + matB_11 * dW2[:,:,t]

                if save_matrices:
                    mat_A_0_list[index_start:index_stop,:,t] = matA_0
                    mat_B_00_list[index_start:index_stop,:,t] = matB_00
                    mat_A_1_list[index_start:index_stop,:,t] = matA_1
                    mat_B_11_list[index_start:index_stop,:,t] = matB_11

            if i < segments_length-1:
                index_start += segments_length_list[i]
                index_stop += segments_length_list[i+1]



        self.times = times
        self.a_trajs = a_trajs
        self.b_trajs = b_trajs
        self.n_simulations = n_simulations

        if self.drift_gauge != None:
            self.gauge_Omega = gauge_Omega

        if save_matrices:
            self.mat_A_0_list = mat_A_0_list
            self.mat_B_00_list = mat_B_00_list
            self.mat_A_1_list = mat_A_1_list
            self.mat_B_11_list = mat_B_11_list
    

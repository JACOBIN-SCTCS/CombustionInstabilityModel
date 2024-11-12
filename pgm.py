import math
import numpy as np
import copy
import matplotlib.pyplot as plt


class VortexSheddingModel:

    def __init__(self, N=10, L=0.7, a_bar=700, c = 0.006, L_c=0.05, zeta_1 = 29, delta_t = 0.00001, d = 0.025, S_t = 0.35, gamma = 1, initial_value = 0.001, p_bar=100000, u_bar = 8):    
        '''
            Parameters
            -------------------------------------------
            N = Number of basis functions
            L = Length of the combustor
            a_bar = mean sound velocity
            c = (2*(gamma-1)*beta)/(L*p_bar)
            L_c = Position of flameholder
            zeta_1 = damping rate of the combustion chamber
            delta_t = timestep size
            d = height of backward facing step
            S_t =  Strouhal number for the step of height d and mean flow velocity u_bar
            initial value  = value for each eta_i
            p_bar = mean static pressure
            u_bar = mean flow velocity
        '''

        self.N = N
        self.L = L
        self.a_bar = a_bar
        self.c = c
        self.L_c = L_c
        self.zeta_1 = zeta_1
        self.delta_t = delta_t
        self.d = d
        self.S_t = S_t
        self.gamma = gamma
        self.initial_value = initial_value
        self.u_bar = u_bar
        self.p_bar = p_bar

        self.CIRC_CRIT_KEY = 'CIRC_CRIT'
        self.VORTEX_POSITIONS_KEY = 'POSITIONS' 

    def k_N_generator(self):
        return np.array([ ((2*i - 1)*math.pi)/(2*self.L) for i in range(1,self.N+1) ])
    
    def w_N_generator(self):
        return self.a_bar * self.k_N_generator()

    def zeta_generator(self):
        return np.array([ ((2*i-1)*(2*i-1))*self.zeta_1 for i in range(1,self.N+1)])

    def A_mat(self):
        A = np.zeros((2*self.N, 2*self.N))
        A[0:self.N, self.N:] = np.identity(self.N)

        w_N = self.w_N_generator()
        omega = np.diag(w_N)
        omega_square = np.matmul(omega, omega)
        A[self.N:, 0:self.N] = - omega_square

        zeta = self.zeta_generator()
        A[self.N:, self.N:] = - np.diag(zeta)

        return A

    def E_mat(self):
        E = np.zeros((2*self.N,1))
        k_N = self.k_N_generator()
        cos_scaled_kn = np.cos(self.L_c * k_N)
        w_N = self.w_N_generator()
        E[self.N:,0]= self.c*(w_N * cos_scaled_kn)
        return E
    
    def A_tilde_mat(self):
        A = self.A_mat()
        identity_matrix = np.identity(2*self.N)

        matrix_1 = (identity_matrix - (self.delta_t/2)*A)
        matrix_2 = (identity_matrix + (self.delta_t/2)*A)

        A_tilde = np.matmul(np.linalg.inv(matrix_1), matrix_2)
        return A_tilde

    def E_tilde_mat(self):
        E = self.E_mat()
        A = self.A_mat()
        identity_matrix = np.identity(2*self.N)
        matrix_1 = (identity_matrix - (self.delta_t/2)*A)
        E_tilde = np.matmul(matrix_1, E)
        return E_tilde

    def circ_critical(self, u_bar = None):
        if u_bar is None:
            res = (self.u_bar * self.d)/ (2 * self.S_t)
        else:
            res = (u_bar * self.d)/(2 * self.S_t)
        return res

    def Heaviside(self,x):
        if(x>=0):
            return 1
        else:
            return 0

    def pressure(self,x,eta_n_dot, k_n):
        sum_term = 0.0
        for i in range(eta_n_dot.shape[0]):
            w_n = self.a_bar * k_n[i]
            fractional_term = ((eta_n_dot[i] / w_n)* math.cos(k_n[i] * x))    
            sum_term += fractional_term

        return self.p_bar * sum_term

    def u_dash(self,x,eta_dot,k_n):    
        sum_term = 0.0
        for i in range(len(eta_dot)):
            sum_term = sum_term + eta_dot[i] * math.sin(k_n[i] * x)
        
        return ((self.a_bar/self.gamma)* sum_term)
    
    def simulation_loop(self,steps, x=0.09):

        circulation_available = 0
        circulation_remaining = 0
        u_k = self.u_bar
        
        alpha_0 = 1

        X = np.array([[0.0] for i in range(2*self.N)])
        X[:self.N] = self.initial_value

        pressure_values = []
        time_values = []

        k_N = self.k_N_generator()
        A_tilde = self.A_tilde_mat()
        E_tilde = self.E_tilde_mat()

        vortices_dict = dict()
        Heaviside_vectorized = np.vectorize(self.Heaviside)
  
        time = 0
        
        for i in range(steps):
            circulation_available = circulation_remaining + (0.5 * self.delta_t * u_k * u_k) 
            c_crit = self.circ_critical()
            vortex_formed = math.floor((circulation_available/ c_crit))
            
            previous_vortices_dict = copy.deepcopy(vortices_dict)

            for j in previous_vortices_dict.keys():
                vortex_position = previous_vortices_dict[j][self.VORTEX_POSITIONS_KEY]
                vortex_position = vortex_position + (alpha_0 * u_k * self.delta_t)*np.ones(vortex_position.shape[0])
                vortices_dict[j][self.VORTEX_POSITIONS_KEY] = vortex_position
            
            if(vortex_formed > 0):
                new_vortices_positions = np.array([ alpha_0*u_k*(self.delta_t + ( ((2*c_crit)/(u_k * u_k))*(m- (circulation_remaining/c_crit))) ) for m in range(1, vortex_formed+1)])
                new_vortex_dict = {}
                new_vortex_dict[self.CIRC_CRIT_KEY] = c_crit
                new_vortex_dict[self.VORTEX_POSITIONS_KEY] = new_vortices_positions
                vortices_dict[i] = new_vortex_dict
        
            S_h_k = 0.0
            
            dict_keys = previous_vortices_dict.keys()
            for j in dict_keys:
                critical_circulation = previous_vortices_dict[j][self.CIRC_CRIT_KEY ]
                previous_vortices_circulation = previous_vortices_dict[j][self.VORTEX_POSITIONS_KEY]
                if(previous_vortices_circulation.shape[0] <= 0):
                    del vortices_dict[j]
                    continue
            
                current_vortices_circulation = vortices_dict[j][self.VORTEX_POSITIONS_KEY]
                collided_mask = Heaviside_vectorized(current_vortices_circulation - self.L_c) - Heaviside_vectorized(previous_vortices_circulation - self.L_c)
                vortices_collided = np.sum(collided_mask)

                S_h_k = S_h_k + critical_circulation * vortices_collided

                collided_mask = collided_mask > 0
                current_vortices_circulation = current_vortices_circulation[~collided_mask]
                vortices_dict[j][self.VORTEX_POSITIONS_KEY] = current_vortices_circulation
            


    
            new_X = np.matmul(A_tilde, X) + S_h_k* E_tilde
            X = new_X

            circulation_remaining = circulation_remaining + (0.5 * self.delta_t * u_k * u_k)  - (vortex_formed * c_crit)

  
            p = self.pressure(x,X[self.N:], k_N)
            pressure_values.append(p)
            time_values.append(time)
            time = time + self.delta_t
        
        return (time_values,pressure_values)


w  = VortexSheddingModel(u_bar=9)
time_values, pressure_values = w.simulation_loop(10000,0.09)

plt.plot(time_values, pressure_values)
plt.ylabel('Pressure (Pa)')
plt.xlabel('Time (s)')
plt.show()

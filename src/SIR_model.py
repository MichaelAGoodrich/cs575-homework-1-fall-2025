from typing import Tuple
import matplotlib.pyplot as plt
class SIR_simulation:
    def __init__(self, 
                 m: float,          # Probability of contact with another person
                 p: float,          # Probability of transfer of infectious person to susceptible person
                 gamma: float,      # Probability that infectious person recovers in current time step
                 dt: float,         # Size of time step
                 duration: int,     # Amount of time for simulation
                 s0: int,           # Number of people initially susceptible
                 i0: int,           # Number of people initially infectious
                 r0: int            # Number of people initially recovered
                 ) -> None:  
        # Global SIR parameters
        self.N: int = s0 + i0 + r0  # Initial population size
        self.beta: float = m * p    # Beta parameter is rate at which susceptible people become infectious
        self.gamma: float = gamma
        # List of population number per compartment in SIR model
        self.S: list[float] = [s0]
        self.I: list[float] = [i0]
        self.R: list[float] = [r0]
        # Simulation parameters
        self.t: list[float] = [0]
        self.dt: float = dt
        self.duration: int = duration
    
    def run_simulation(self) -> None:
        while self.t[-1] < self.duration:
            s_next, i_next, r_next = self.euler_integration(self.S[-1], self.I[-1], self.R[-1])
            self.S.append(s_next)
            self.I.append(i_next)
            self.R.append(r_next)
            self.t.append(self.t[-1] + self.dt)
    
    ###########################
    ## Integration functions ##
    ###########################
    def euler_integration(self,
                          s: float, 
                          i: float,
                          r: float
                          ) -> Tuple[float, float, float]:
        #############################
        # Euler integration method ##
        #############################    
        s_next: float = s + self.dt * self.f(s, i, r)
        i_next: float = i + self.dt * self.g(s, i, r)
        r_next: float = r + self.dt * self.h(s, i, r)
        return (s_next, i_next, r_next)
    
    def midpoint_integration(self,
                             s: float,
                             i: float,
                             r: float
                             ) -> Tuple[float, float, float]:
        ################################
        # Midpoint integration method ##
        ################################                     
        s_next: float = s + self.dt/2 * self.f(s,i,r)
        i_next: float = i + self.dt/2 * self.h(s,i,r)
        r_next: float = r + self.dt/2 * self.g(s,i,r)
        return (s_next, i_next, r_next)
        
    #################################
    ## Functions used in SIR Model ##
    #################################
    def f(self,s,i,r) -> float: 
        # From dS/dt equation
        return -self.beta * i * s /self.N 
    def g(self,s,i,r) -> float: 
        # From dI/dt equation
        return self.beta * i * s /self.N - self.gamma*i
    def h(self,s,i,r) -> float: 
        # From dR/dt equation
        return self.gamma*i
    
    ## Plotting functions ##
    def show_plot(self) -> None:
        plt.plot(self.t,self.S,'r',
                 self.t,self.I,'g--',
                 self.t,self.R,'b:')
        plt.legend(['S','I','R'])
        title = 'SIR model via Euler: beta = ' \
            + str(self.beta) \
            + ', gamma = ' + str(self.gamma) \
            + ', and R0 = ' + str(self.beta/self.gamma)
        plt.title(title)

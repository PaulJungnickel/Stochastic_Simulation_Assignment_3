import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import scipy
import scipy.spatial




def basic_cooling_schedule(T, step):
    return 0.999*T

def paper_cooling_schedule(T, step):
    if step % 100 == 0:
        return 0.9*T
    return T

def exponential_cooling_schedule(T, step, alpha=0.99):
    return alpha * T

def log_cooling_schedule(T, step):
    
    return 1/(np.log(step + 3))


def const_step_size_schedule(step, *args):
    return 0.1

def random_step_size_schedule(step, *args):
    return np.random.rand()

def hyperbolic_step_size_schedule(step, *args):
    return np.maximum(0.001, 1./(1 + 0.1*step))

def linear_step_size_schedule(step, *args):
    return np.maximum(0.0001, 1 - 1/10000 * step)


class CircleParticleSim:
    
    def __init__(
            self,
            N,
            initial_temperature = 10,
            steps=10000,
            seed=42,
            cooling_schedule = basic_cooling_schedule,
            step_size_schedule = const_step_size_schedule
            ) -> None:
        
        
        # rand.seed(seed)

        # parameters for the simulation
        self.N = N
        self.T = initial_temperature
        self.initial_locations()
        self.initial_energy()
        self.cooling_schedule = cooling_schedule
        self.step_size_schedule = step_size_schedule

        print('initial energy', self.E)


        # initialize statistics datastructures
        self.energy_values = np.zeros(steps)
        self.temp_values = np.zeros(steps)
        self.num_internal_pts = np.zeros(steps)
        

        
        # for step in range(steps):
        #     self.step = step
        #     self.single_move()
        #     self.T = self.cooling_schedule(self.T, step)
        
        # self.plot_positions()
        # print('mimimal energy', self.E)


          
          
    def plot_positions(self):
        thetas = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(thetas), np.sin(thetas), linestyle=':', color='gray')
        plt.scatter(self.particle_locations[:,0], self.particle_locations[:,1])
        plt.grid()
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        plt.show()
        


    def initial_locations(self):
        r = rand.rand(self.N)
        theta = 2*np.pi * rand.rand(self.N)
        
        self.particle_locations = np.dstack((r*np.cos(theta), r*np.sin(theta))).reshape(-1,2)
            
    def initial_energy(self):
        self.distance_matrix = scipy.spatial.distance_matrix(self.particle_locations, self.particle_locations)
        np.fill_diagonal(self.distance_matrix, np.inf)
        self.E = np.sum(1/self.distance_matrix)

    def step_consequences(self, particle_index, new_location):
        energy_contribution = 2 * np.sum(1/self.distance_matrix[particle_index,:])
        new_distances = scipy.spatial.distance_matrix(new_location[None, :], self.particle_locations).reshape(-1)
        new_distances[particle_index] = np.inf
        new_energy_contribution = 2 * np.sum(1/new_distances)
        return (new_energy_contribution - energy_contribution), new_distances

    def single_move(self):
        i = rand.randint(self.N)
        particle_location = self.particle_locations[i]

        r =  self.step_size_schedule(self.step) # * rand.rand()
        theta = 2*np.pi * rand.rand()     
        step = [r*np.cos(theta), r*np.sin(theta)]
        new_location = particle_location+step
        if np.linalg.norm(new_location)>1:
            new_location /=  np.linalg.norm(new_location)
        dE, new_distances = self.step_consequences(i, new_location)
        acceptance = min(1, np.exp(-dE/self.T))
        if rand.rand() < acceptance:
            self.particle_locations[i] = new_location
            self.E += dE
            self.distance_matrix[i,:] = new_distances
            self.distance_matrix[:,i] = new_distances
        
        self.energy_values[self.step] = self.E
        self.temp_values[self.step] = self.T

    def run_simulation(self, steps):
        energies_over_time = []
        for step in range(steps):
            self.step = step
            self.single_move()
            self.T = self.cooling_schedule(self.T, step)
            energies_over_time.append(self.E)
        return np.array(energies_over_time)





# plotting functions
def evaluate_multiple_runs(N, cooling_schedule, steps=10000, num_runs=10):
    all_energy_values = np.zeros((num_runs, steps))
    
    for run in range(num_runs):
        sim = CircleParticleSim(N, cooling_schedule=cooling_schedule)
        energies = sim.run_simulation(steps)
        all_energy_values[run] = energies
    
    mean_energy = np.mean(all_energy_values, axis=0)
    std_energy = np.std(all_energy_values, axis=0)
    
    return mean_energy, std_energy

def plot_shadow(mean_energy, std_energy):
    plt.figure(figsize=(10, 6))
    plt.plot(mean_energy, label="Mean Energy", color='b')
    plt.fill_between(range(len(mean_energy)), mean_energy - std_energy, mean_energy + std_energy, color='b', alpha=0.3)
    plt.xlabel("Steps")
    plt.ylabel("Energy")
    plt.title("Minimal Energy with Standard Deviation Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    num_particles = 5
    steps = 10000 
    num_runs = 20  
    
    mean_energy, std_energy = evaluate_multiple_runs(num_particles, cooling_schedule=basic_cooling_schedule, steps=steps, num_runs=num_runs)
    
    plot_shadow(mean_energy, std_energy)
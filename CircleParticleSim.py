import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import scipy
import scipy.spatial

def basic_cooling_schedule(sim):
    """
    Basic cooling schedule that reduces temperature slightly each step.

    Parameters:
    - T: Current temperature.
    - step: Current step number.

    Returns:
    - Updated temperature.
    """
    return 0.999 * sim.T

def paper_cooling_schedule(sim):
    """
    Cooling schedule inspired by a specific paper; reduces temperature
    significantly every 1000 steps.

    Parameters:
    - T: Current temperature.
    - step: Current step number.

    Returns:
    - Updated temperature.
    """
    if sim.extra_args is None:
        steps_until_decrease = 100
    else:
        steps_until_decrease = int(sim.extra_args['cooling_schedule_scaling'] * 100)
    if sim.step % steps_until_decrease == 0:
        return 0.9 * sim.T
    return sim.T

def exponential_cooling_schedule(sim):
    """
    Exponential cooling schedule with a fixed decay rate.

    Parameters:
    - T: Current temperature.
    - step: Current step number.
    - alpha: Decay factor (default 0.99).

    Returns:
    - Updated temperature.
    """
    return 0.999 * sim.T

def log_cooling_schedule(sim):
    """
    Logarithmic cooling schedule based on the step count.

    Parameters:
    - T: Current temperature.
    - step: Current step number.

    Returns:
    - Updated temperature.
    """
    return 1 / (np.log(sim.step + 3))

def linear_cooling_schedule(sim):
    """
    Linear cooling schedule that decreases temperature linearly with steps.

    Returns:
    - Updated temperature.
    """
    return max(0.01, sim.T - 0.0001)

def quadratic_cooling_schedule(sim):
    """
    Quadratic cooling schedule that decreases temperature quadratically.

    Returns:
    - Updated temperature.
    """
    return max(0.01, sim.T * (1 - (sim.step / 10000) ** 2))

def sigmoid_cooling_schedule(sim):
    """
    Sigmoid cooling schedule for gradual temperature decrease.

    Returns:
    - Updated temperature.
    """
    return sim.T * (1 / (1 + np.exp((sim.step - 5000) / 1000)))

def inverse_sqrt_cooling_schedule(sim):
    """
    Inverse square root cooling schedule.

    Returns:
    - Updated temperature.
    """
    return sim.T / np.sqrt(sim.step + 1)

def cosine_annealing_cooling_schedule(sim):
    """
    Cosine annealing cooling schedule.

    Returns:
    - Updated temperature.
    """
    return 0.5 * sim.T * (1 + np.cos(np.pi * sim.step / 10000))

def stepwise_cooling_schedule(sim):
    """
    Stepwise cooling schedule that reduces temperature in steps.

    Returns:
    - Updated temperature.
    """
    if sim.step % 500 == 0:
        return 0.8 * sim.T
    return sim.T


def const_step_size_schedule(sim):
    """
    Step size schedule that returns a constant step size.

    Parameters:
    - step: Current step number.

    Returns:
    - Fixed step size (0.1).
    """
    return 0.1

def random_step_size_schedule(sim):
    """
    Step size schedule that returns a random step size.

    Parameters:
    - step: Current step number.

    Returns:
    - Random step size.
    """
    return np.random.rand()

def hyperbolic_step_size_schedule(sim):
    """
    Step size schedule that decreases hyperbolically with steps.

    Parameters:
    - step: Current step number.

    Returns:
    - Updated step size.
    """
    return np.maximum(0.001, 1. / (1 + 0.1 * sim.step))

def linear_step_size_schedule(sim):
    """
    Step size schedule that decreases linearly with steps.

    Parameters:
    - step: Current step number.

    Returns:
    - Updated step size.
    """
    return np.maximum(0.0001, 1 - 1 / 10000 * sim.step)


def sqrt_step_size_schedule(sim):
    """
    Step size schedule that decreases as the square root of temperature.

    Parameters:
    - step: Current step number.

    Returns:
    - Updated step size.
    """
    return np.sqrt(sim.T) * rand.rand()


def sqrt_const_step_size_schedule(sim):
    """
    Step size schedule that decreases as the square root of temperature.

    Parameters:
    - step: Current step number.

    Returns:
    - Updated step size.
    """
    return np.sqrt(sim.T) 



def random_step_direction(sim, particle_index):
    return 2 * np.pi * rand.rand()

# def force_step_direction(sim, particle_index):
#     particle = sim.particle_locations[particle_index]

#     diff = particle - sim.particle_locations

#     F = diff / (np.linalg.norm(diff, axis=1) **3).reshape(-1,1)
#     F[particle_index] = 0
#     F = np.sum(F, axis=0)
#     theta = np.atan2(F[1],  F[0])
#     # theta = np.pi - theta

#     return theta

def force_step_direction(sim, particle_index):
    particle = sim.particle_locations[particle_index]
    diff = particle - sim.particle_locations

    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    norms = np.linalg.norm(diff, axis=1)
    norms = np.maximum(norms, epsilon)  # Ensure no zero norms
    norms = norms ** 3

    F = diff / norms[:, None]
    F[particle_index] = 0  # Exclude self-force
    F = np.sum(F, axis=0)

    theta = np.arctan2(F[1], F[0])
    return theta



class CircleParticleSim:
    """
    Simulation of particles constrained to a circular boundary.
    
    Attributes:
    - N: Number of particles.
    - T: Current temperature.
    - particle_locations: Array of particle coordinates.
    - distance_matrix: Pairwise distances between particles.
    - cooling_schedule: Function to update temperature.
    - step_size_schedule: Function to update step size.
    """

    def __init__(
            self,
            N,
            initial_temperature=10,
            steps=10000,
            seed=42,
            cooling_schedule=paper_cooling_schedule,
            step_size_schedule=random_step_size_schedule,
            random_step_likelihood = 0.2,
            extra_args = None
            ) -> None:
        """
        Initialize the simulation with given parameters.

        Parameters:
        - N: Number of particles.
        - initial_temperature: Initial temperature of the system.
        - steps: Number of steps for the simulation.
        - seed: Random seed for reproducibility.
        - cooling_schedule: Function to update temperature.
        - step_size_schedule: Function to update step size.
        - random_step_likelihood: probability of taking a random step 
        (instead of a step in the direction of all forces working on the particle)
        - extra_args dictionary with named arguments accessed by the member functions
        """

        # Set random seed
        # rand.seed(seed)

        # Simulation parameters
        self.N = N
        self.T = initial_temperature
        self.extra_args = extra_args
        self.initial_locations()
        self.initial_energy()
        self.cooling_schedule = cooling_schedule
        self.step_size_schedule = step_size_schedule
        self.random_step_likelihood = random_step_likelihood

        # print('initial energy', self.E)

        # Initialize statistics data structures
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
        """
        Plot the current positions of particles in the circular boundary.
        """
        thetas = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(thetas), np.sin(thetas), linestyle=':', color='gray')
        plt.scatter(self.particle_locations[:, 0], self.particle_locations[:, 1])
        plt.grid()
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.show()

    def initial_locations(self):
        """
        Initialize random positions of particles within the circle.
        """
        r = rand.rand(self.N)
        theta = 2 * np.pi * rand.rand(self.N)
        self.particle_locations = np.dstack((r * np.cos(theta), r * np.sin(theta))).reshape(-1, 2)

    def initial_energy(self):
        """
        Calculate the initial energy of the system based on pairwise distances.
        """
        self.distance_matrix = scipy.spatial.distance_matrix(self.particle_locations, self.particle_locations)
        np.fill_diagonal(self.distance_matrix, np.inf)  # Avoid division by zero
        self.E = np.sum(1 / self.distance_matrix)

    def step_consequences(self, particle_index, new_location):
        """
        Calculate energy change and distances for a proposed move.

        Parameters:
        - particle_index: Index of the particle to move.
        - new_location: Proposed new location for the particle.

        Returns:
        - Energy change due to the move.
        - Updated distance matrix for the particle.
        """
        energy_contribution = 2 * np.sum(1 / self.distance_matrix[particle_index, :])
        new_distances = scipy.spatial.distance_matrix(new_location[None, :], self.particle_locations).reshape(-1)
        new_distances[particle_index] = np.inf  # Ignore self-distance
        new_energy_contribution = 2 * np.sum(1 / new_distances)
        return (new_energy_contribution - energy_contribution), new_distances

    def single_move(self):
        """
        Perform a single move in the simulation, updating the system's state.
        """
        i = rand.randint(self.N)  # Select a random particle
        particle_location = self.particle_locations[i]

        # Propose a move
        r = self.step_size_schedule(self)
        theta = 0
        if rand.rand() < self.random_step_likelihood:
            theta = random_step_direction(self, i)
        else:
            theta = force_step_direction(self, i)
        step = [r * np.cos(theta), r * np.sin(theta)]
        new_location = particle_location + step

        # Ensure the particle stays within the circular boundary
        if np.linalg.norm(new_location) > 1:
            new_location /= np.linalg.norm(new_location)

        # Calculate energy change and decide whether to accept the move
        dE, new_distances = self.step_consequences(i, new_location)
        acceptance = min(1, np.exp(-dE / self.T))
        if rand.rand() < acceptance:
            self.particle_locations[i] = new_location
            self.E += dE
            self.distance_matrix[i, :] = new_distances
            self.distance_matrix[:, i] = new_distances

        # Update statistics
        self.energy_values[self.step] = self.E
        self.temp_values[self.step] = self.T

    def run_simulation(self, steps):
        """
        Run the simulation for a given number of steps.

        Parameters:
        - steps: Number of steps to run the simulation.

        Returns:
        - Array of energy values over time.
        """
        energies_over_time = []
        temp_over_time = []
        for step in range(steps):
            self.step = step
            self.single_move()
            self.T = self.cooling_schedule(self)
            energies_over_time.append(self.E)
            temp_over_time.append(self.T)
        return np.array(energies_over_time), np.array(temp_over_time)

# Plotting functions
def evaluate_multiple_runs(N, cooling_schedule, steps=10000, num_runs=10):
    """
    Run the simulation multiple times and collect statistics.

    Parameters:
    - N: Number of particles.
    - cooling_schedule: Cooling schedule function.
    - steps: Number of steps per run.
    - num_runs: Number of simulation runs.

    Returns:
    - Mean energy and standard deviation of energy over time.
    """
    all_energy_values = np.zeros((num_runs, steps))
    all_temperature_values = np.zeros((num_runs, steps))

    for run in range(num_runs):
        sim = CircleParticleSim(N, cooling_schedule=cooling_schedule)
        energies, temperatures = sim.run_simulation(steps)
        all_energy_values[run] = energies
        all_temperature_values[run] = temperatures
    
    mean_energy = np.mean(all_energy_values, axis=0)
    std_energy = np.std(all_energy_values, axis=0)
    mean_temperatures = np.mean(all_temperature_values, axis=0)


    return mean_energy, std_energy, mean_temperatures, all_energy_values

def plot_shadow(mean_energy, std_energy):
    """
    Plot the mean energy with standard deviation as a shadow.

    Parameters:
    - mean_energy: Mean energy values over time.
    - std_energy: Standard deviation of energy values over time.
    """
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
    num_runs = 3
    schedules = [
    # log_cooling_schedule,
    # basic_cooling_schedule,
    paper_cooling_schedule,
    exponential_cooling_schedule,
    # linear_cooling_schedule,
    # quadratic_cooling_schedule,
    sigmoid_cooling_schedule,
    inverse_sqrt_cooling_schedule,
    cosine_annealing_cooling_schedule,
    # stepwise_cooling_schedule,
    ]

 

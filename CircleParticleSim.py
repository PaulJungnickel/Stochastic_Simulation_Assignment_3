import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

class CircleParticleSim:
    
    def __init__(self, N, seed=42) -> None:
        
        self.N = N
        self.particlePositions = np.zeros([N, 2])

        
        r = rand.rand(N)
        theta = 2*np.pi * rand.rand(N)
        
        self.particlePositions = np.dstack((r*np.cos(theta), r*np.sin(theta))).reshape(-1,2)

        self.plot_positions()
        
        for i in range(100):
            self.single_move()
        
        self.plot_positions()
          
          
    def plot_positions(self):
        
        plt.scatter(self.particlePositions[:,0], self.particlePositions[:,1])
        plt.grid()
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.show()
        
            
            
    def single_move(self):
        
        i = rand.randint(self.N)
        
        particle = self.particlePositions[i]
        
        r = rand.rand()
        theta = 2*np.pi * rand.rand()
        
        print(i, particle)
        particle += [r*np.cos(theta), r*np.sin(theta)]
        
        print(i, particle)
        
        if np.linalg.norm(particle)>1:
            particle = particle / np.linalg.norm(particle)
            
        self.particlePositions[i] = particle
        print(i, particle)
    
    
if __name__ == '__main__':
    
    sim = CircleParticleSim(30)

    
#learn to play with vectors
import numpy as np
import matplotlib.pyplot as plt


# to make it repeatable
# how to make a function that can plot given a list of vectors
# lets' use tuples to represent a vector

def plot_vectors(vecs):
    # for different color vectors
    colors = ['r', 'b', 'g', 'o', 'y']
    i = 0
    for vec in vecs:
        plt.quiver(vec[0], vec[1], vec[2], vec[3], scale_units='xy', angles='xy', scale=1, color=colors[i%len(colors)])
        i+= 1
        
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.show()


#plot a vector
plt.quiver(0,0,3,4, scale_units='xy', angles='xy', scale=1, color='r')

#plot another vector
plt.quiver(0,0,-3,4, scale_units='xy', angles='xy', scale=1, color='g')

# or we can have a function and do it this way to make it repeatable
plot_vectors([(0,0,1,2), (1,2,3,5)])
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()

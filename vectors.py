# learn to play with vectors
import numpy as np
import matplotlib.pyplot as plt


# to make it repeatable
# how to make a function that can plot given a list of vectors
# lets' use tuples to represent a vector

def plot_vectors(vecs):
    # for different color vectors
    colors = ['r', 'b', 'g', 'y']
    i = 0
    for vec in vecs:
        plt.quiver(vec[0], vec[1], vec[2], vec[3], scale_units='xy',
                   angles='xy', scale=1, color=colors[i % len(colors)])
        i += 1

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()
    return


def add_vectors(vecs):
    # do nothing
    # first convert the vectors to numpy arrays so that we can do placewise addition
    vecs = [np.asarray(vecs[0]), np.asarray(vecs[1]),
            np.asarray(vecs[2]), np.asarray(vecs[3])]
    # if you now do an addition of two vectors then we will get a placewise addition
    plot_vectors([vecs[0], vecs[1], vecs[0]+vecs[1]])

    return


def dot_product(myVec):
    print("processign doct product")
    a = np.asarray(myVec[0])
    b = np.asarray(myVec[1])
    # vec a . vec b  =  mod a.mod b. cos 0
    # dot product is computed
    a_dot_b = np.dot(a, b)
    # we get a scalar output, placewise multiplication
    print(a_dot_b)

    # another way, let's visualize it now.
    # a_b  = mod a. cos 0
    # a_b = mod a (a_dot_b / mod a . mod b)
    # or a_b = a_dot_b / mod b
    a_b = np.dot(a, b)/np.linalg.norm(b)
    print(a_b)  # a scalar value
    # to find a vector with this scalar value
    # vec a on b = = a_b . unit vector b
    # how to plot this vector, it needs a direction, it is in the same direction as b
    # so find a unit vector in the direction of b
    # which is vector b/ mag of b
    #
    vec_a_b = (a_b/np.linalg.norm(b)) * b
    plot_vectors([a, b, vec_a_b])
    return


myVec = [(0, 0, 3, 4), (0, 0, -3, 4), (0, 0, -3, -2), (0, 0, 4, -1)]

# plot a vector
plt.quiver(0, 0, 3, 4, scale_units='xy',
           angles='xy', scale=1, color='r')

# plot another vector
plt.quiver(0, 0, -3, 4, scale_units='xy',
           angles='xy', scale=1, color='g')

# or we can have a function and do it this way to make it repeatable
# plot_vectors([(0, 0, 1, 2), (1, 2, 3, 5)])
# plot_vectors([myVec[0], myVec[3]])
add_vectors(myVec)

# try dot product
dot_product([[0, 0, 3, 4], [0, 0, 4, -1]])

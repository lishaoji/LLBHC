
# coding: utf-8

# In[1]:

# require necessary packages
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, vectorize, float64, int64


# In[2]:

class node():
    """
    The class for node.
    Initialization requires two child branches, which may be None for end node, 
    the data points, and the index number of cluter.
    When initialized, the node also keeps track of the number of total nodes,
    both internal and end, within its structure as n.
    
    For example:
    new_node = node(None, None, np.repeat(1,4), 1)
    """
    
    # initialization 
    def __init__(self, left, right, data, cnum):
        # set up the left and right branches
        self.l = left
        self.r = right
        
        # track of number of total nodes using recursion
        if(left == None and right == None):
            self.n = 1
        else:
            self.n = left.n + right.n
            
        # save data points and number of cluster
        self.data = data
        self.cluster = cnum


# In[3]:

@jit(float64(float64[:,:], int64[:]))

def p_hyp1(dataset, a):
    """
    Function to calculate the posterior probability.
    The input requires data, which is either a vector
    or a 2D numpy array, and an alpha value, which is
    a double
    """
    
    # extract the number of features and the total number of data
    
    # If the data is a vector, do the following
    if (len(dataset.shape) == 1):
        N = 1
        k = dataset.shape[0]
        # part I
        p1 = 1
        comp = special.gamma(np.sum(dataset)+1) / np.prod(special.gamma(dataset+1))
        p1 = p1 * comp
        
        # part II
        # iterate to calculate the probability
        p2 = p1 * special.gamma(np.sum(a)) / special.gamma(np.sum(dataset) + np.sum(a))
        for j in range(k):
            comp = special.gamma(a[j] + np.sum(dataset[j])) / special.gamma(a[j])
            p2 = p2 * comp
    # if the data is not vector, do the following
    else:
        N = dataset.shape[0]
        k = dataset.shape[1]
    
        # part I
        p1 = 1
        for i in range(N):
            comp = special.gamma(np.sum(dataset[i, :])+1) / np.prod(special.gamma(dataset[i, :]+1))
            p1 = p1 * comp
        
        # part II
        # iterate to calculate the probability
        p2 = p1 * special.gamma(np.sum(a)) / special.gamma(np.sum(dataset) + np.sum(a))
        for j in range(k):
            comp = special.gamma(a[j] + np.sum(dataset[:, j])) / special.gamma(a[j])
            p2 = p2 * comp

    return p2


# In[4]:

def get_d(node, a):
    """
    Recursive function to calculate the 'd' value for each node
    
    """
    if node.l == None and node.r == None:
        return a
    else:
        return a*special.gamma(node.n) + get_d(node.l, a)*get_d(node.r, a)


# In[5]:

def get_pi(node, a):
    """
    The function to calculate the weight for each node (pi_k).
    It uses d and the gamma function.
    The inputs are a node object and a double
    """
    dk = get_d(node, a)
    pi_k = a*special.gamma(node.n)/dk
    return pi_k


# In[ ]:

def get_dk(node, a):
    """
    The Recursive function to calculate the posterior probability for
    each node given a subtree (Ti).
    The inputs are a node and a double
    """
    post = p_hyp1(node.data, np.repeat(a, node.data.shape[1]))
    pi = get_pi(node, a)
    if node.l == None and node.r == None:
        return  pi * post
    else:
        return  pi * post + (1-pi) * get_dk(node.l, a) * get_dk(node.r, a)


# In[ ]:

def bhc(data, a=1, r_thres=0.5):
    """
    The Baysian Hierarchical Clustering algorithm.
    It is described in the paper collaborated by
    Dr. Katherine Heller and Dr. Zoubin Ghahramani
    in 2005.
    """
    
    # Initialize a node_list tracking the nodes to be merged
    # and a node_list_copy to track the cluster number of each
    # node. The initial value of those two lists are each data
    # points with its own cluster number.
    node_list = []
    node_list_copy = []
    for i in range(data.shape[0]):
        node_list.append(node(None, None, np.array([data[i,:]]), i))
        node_list_copy.append(node(None, None, np.array([data[i,:]]), i))
    
    # Cluster number, default value equals the number of data points
    c = data.shape[0]
    
    # Iterate to merge nodes. Note that BHC is a greedy algorithm, which means
    # If no tow nodes can be merged, the loop stops automatically
    while c > 1:
        # Indicate whether to break the while loop
        flag = False
        
        for i in range(len(node_list)):
            for j in range(i+1, len(node_list)):
                # Create a new data by row-binding the datasets in the two nodes
                newdata = np.concatenate((node_list[i].data, node_list[j].data), axis = 0)
                
                # Create a new node based on the new data
                # Set the cluster number of the new node to
                # the minimum of the two nodes combined
                node_new = node(node_list[i], node_list[j], newdata, 
                                min(node_list[i].cluster,node_list[j].cluster))
                
                # Calculate the probability of the hypothesis being true
                pi_k = get_pi(node_new, a)
                
                # Calculate the posterior probability of data given hypothesis
                p_hyp = p_hyp1(node_new.data, np.repeat(a, data.shape[1]))
                
                # Calculate the posterior probability of data given subtree
                p_dk = get_dk(node_new, a)
                
                # Calculate the probability of the merged hypothesis
                rk = pi_k * p_hyp / p_dk
                
                # If the probability of the merged hypothesis is greater
                # than the threshold we set, merge the two nodes, reset
                # their cluster number in node_list_copy and remove the
                # two nodes from nodes_list.
                
                # Note that since it's a greedy algorithm, we break the
                # double for loop if the nodes are merged and continue
                # on finding the next two nodes to merge.
                if rk >= r_thres:
                    for k in range(len(node_list_copy)):
                        entry = node_list_copy[k].cluster
                        if entry == node_list[i].cluster or entry == node_list[j].cluster:
                            node_list_copy[k].cluster = min(node_list[i].cluster,node_list[j].cluster)
                    node_list =  node_list[:i] + node_list[(i+1):j] + node_list[(j+1):]
                    node_list = [node_new] + node_list
                    
                    c = c - 1
                    flag = True
                    break
            if flag == True:
                break
        
        if flag == False:
            c = 1        

    return node_list, node_list_copy
    


# In[ ]:




# In[6]:

get_ipython().system('jupyter nbconvert --to script __init__.ipynb')


# In[ ]:




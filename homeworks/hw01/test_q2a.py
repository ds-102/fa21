import numpy as np
from scipy import stats

unif_val = np.random.uniform(0, 1, 50000)  # large sample from U(0, 1)
norm_val = np.random.normal(3, 1, 50000)   # large sample from N(3, 1)
alternative_distr = 1 - stats.norm.cdf(norm_val)   # corresponding p-values if H_0: N(0, 1)


def check_1(array, pi_0):
    # Inputs: array - array of p-values, pi_0 - fraction of null p-values
    # Output: error if array doesn't seem correct

    # mixture of p-values from null and alternative
    mixture_distr = np.random.choice(np.concatenate(
        (unif_val[0 : int(pi_0*len(unif_val))],
        alternative_distr[int(pi_0*len(alternative_distr)) : len(alternative_distr)])),
                                    50000, replace = False)
    # test if first half is balanced
    t1 = stats.kstest(array[0 : int(0.5*len(array))], mixture_distr)[1] 
    # test if second half is balanced
    t2 = stats.kstest(array[int(0.5*len(array)) + 1 : len(array)], mixture_distr)[1] 
    assert((t1 > 0.05) & (t2 > 0.05))  
    return('The array of p-values looks good!')
    #return(t1, t2)
    

def check_2(array, pi_0):
    # Inputs: array - array of p-values, pi_0 - fraction of null p-values
    # Output: error if array doesn't seem correct
    
    # test if first block of p-val are from null
    t1 = stats.kstest(array[0 : int(pi_0*len(array))], 'uniform')[1] 
    # test if second block of p-val are from alternative
    t2 = stats.kstest(array[int(pi_0*len(array)) + 1 : len(array)], alternative_distr)[1] 
    assert((t1 > 0.05) & (t2 > 0.05))
    return('The array of p-values looks good!')
    #return(t1, t2)


def check_3(array, pi_0):
    # Inputs: array - array of p-values, pi_0 - fraction of null p-values
    # Output: error if array doesn't seem correct
    
    # test if first block of p-val are from alternative
    t1 = stats.kstest(array[0 : len(array) - int(pi_0*len(array))], alternative_distr)[1]
    # test if second block of p-val are from null
    t2 = stats.kstest(array[len(array) - int(pi_0*len(array)) + 1 : len(array)], 'uniform')[1]
    assert((t1 > 0.05) & (t2 > 0.05))
    return('The array of p-values looks good!')
    #return(t1, t2)
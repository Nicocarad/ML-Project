import numpy



def mcol(v):
    return v.reshape((v.size, 1))


def mrow(array):
    return array.reshape((1,array.size))



def weighted_bounds(C, LTR, prior):
    bounds = numpy.zeros(LTR.shape[0])
    emp = numpy.sum(LTR == 1) / LTR.shape[0]
    
    bounds[LTR == 1] = C * prior[1] / emp
    bounds[LTR == 0] = C * prior[0] / (1 - emp)
    
    return list(zip(numpy.zeros(LTR.shape[0]), bounds))




def compute_lagrangian_wrapper(H_hat):
    def compute_lagrangian(alpha):
        ones = numpy.ones_like(alpha)
        term1 = H_hat @ alpha
        term2 = alpha @ term1
        J_hat_D = 0.5 * term2 - alpha @ ones
        L_hat_D_gradient = term1 - ones
        
        return J_hat_D, numpy.ravel(L_hat_D_gradient)

    return compute_lagrangian


import numpy



def kfold(model,k,D,L,eff_prior=None,seed=4):
    

    SPost_partial = []
    folds = []
     
     # Create a list with indices of the Label vector in a random order
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    
    Label = L[idx] # randomize the vector of Real_labels in the same way

    
    fold_size = D.shape[1] // k

    # Divide indices in k-folds
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds.append(idx[start:end])
    

    # If the number of samples is not divisible by K, add the leavings samples in the last fold
    if D.shape[1] % k != 0:
        folds[-1] = numpy.concatenate((folds[-1], idx[k * fold_size:]))
        
    # Perform Cross validation
    for i in range(k):
        # Choose the i-th fold as validation fold
        validation_indices = folds[i]
        DTE = D[:,validation_indices] 
        LTE = L[validation_indices]
        # Use the leaving folds as Training Set
        train_indices = numpy.concatenate([folds[j] for j in range(k) if j != i])
        DTR = D[:,train_indices] 
        LTR = L[train_indices]
        # Append in the list the Scores (posterior probabilities) for the samples of the training fold
        if eff_prior is None:
            
           Spost = model(DTR, LTR, DTE, LTE)
        else:
           Spost = model(DTR, LTR, DTE, LTE,eff_prior)
        
       
        SPost_partial.append(Spost)
       
          
    S = numpy.hstack(SPost_partial) 
      
    
    return S,Label




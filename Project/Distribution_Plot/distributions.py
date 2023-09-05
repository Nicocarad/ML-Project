import numpy;
import matplotlib.pyplot as plt



def mcol(v):
    return v.reshape((v.size, 1)) # takes a row vector and emit a column vector

def load(fname): 
    DList = []
    labelsList = []
    hLabels = {
        '0': 0,
        '1': 1
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12] # list of string ( the first 4 string)
                attrs = mcol(numpy.array([float(i) for i in attrs])) # list compr
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except: #if the file contains an empty line at the end
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]


    hFea = {
        0: 'Feature 1',
        1: 'Feature 2',
        2: 'Feature 3',
        3: 'Feature 4',
        4: 'Feature 5',
        5: 'Feature 6',
        6: 'Feature 7',
        7: 'Feature 8',
        8: 'Feature 9',
        9: 'Feature 10',
        10: 'Feature 11',
        11: 'Feature 12'
                                                                          
        }

    for dIdx in range(12):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Male')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Female')

        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    
    







if __name__ == '__main__':



    D, L = load('Dataset/Train.txt')
    plot_hist(D, L)


  
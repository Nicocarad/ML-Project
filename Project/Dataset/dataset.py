import numpy;


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
if __name__ == '__main__':



    D1, L1 = load('Train.txt')
    D2, L2 = load('Test.txt')
    print(L1.size)
    print(D1.size)
    
    num_label_0 = numpy.count_nonzero(L1 == 0)
    num_label_1 = numpy.count_nonzero(L1 == 1)
    
    print("(Train) Number of samples with label 0:", num_label_0)
    print("(Train) Number of samples with label 1:", num_label_1)
    
    print(L2.size)
    print(D2.size)
    
    num_label_0 = numpy.count_nonzero(L2 == 0)
    num_label_1 = numpy.count_nonzero(L2 == 1)
    
    print(" (Test) Number of samples with label 0:", num_label_0)
    print(" (Test) Number of samples with label 1:", num_label_1)

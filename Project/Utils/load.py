import numpy
from Utils.utils import * 


def load(fname):
    DList = []
    labelsList = []
    hLabels = {"0": 0, "1": 1}

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(",")[0:12]  # list of string ( the first 4 string)
                attrs = mcol(numpy.array([float(i) for i in attrs]))  # list compr
                name = line.split(",")[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:  # if the file contains an empty line at the end
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

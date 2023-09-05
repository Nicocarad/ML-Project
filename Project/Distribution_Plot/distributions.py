import numpy
import matplotlib.pyplot as plt


def mcol(v):
    return v.reshape((v.size, 1))


def load(fname):
    DList = []
    labelsList = []
    hLabels = {"0": 0, "1": 1}

    with open(fname) as f:
        for line in f:
            try:
                attrs = [float(i) for i in line.split(",")[0:12]]
                attrs = mcol(numpy.array(attrs))
                name = line.split(",")[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def plot_centered_hist(D, L, centered):
    if centered == "true":
        mean_D = numpy.mean(
            D, axis=1, keepdims=True
        )  # compute the mean along each feature
        D_centered = D - mean_D
    else:
        D_centered = D

    D0 = D_centered[:, L == 0]
    D1 = D_centered[:, L == 1]

    hFea = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
        6: "Feature 7",
        7: "Feature 8",
        8: "Feature 9",
        9: "Feature 10",
        10: "Feature 11",
        11: "Feature 12",
    }

    for dIdx, h in hFea.items():
        plt.figure()
        plt.xlabel(h, fontsize = 10, fontweight='bold')
        plt.ylabel("Probability Density", fontsize = 10, fontweight='bold')
        plt.hist(D0[dIdx, :], bins=80, density=True, alpha=0.4, label="Male", linewidth= 0.3, edgecolor='black')
        plt.hist(D1[dIdx, :], bins=80, density=True, alpha=0.4, label="Female", linewidth=0.3, edgecolor='black' ,color = "red")

        plt.legend()
        plt.tight_layout()
        plt.savefig("Plot/hist_%d.pdf" % dIdx)


if __name__ == "__main__":
    D, L = load("Train.txt")
    plot_centered_hist(D, L, "true")

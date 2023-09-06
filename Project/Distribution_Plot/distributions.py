import numpy
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.stats as sp


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
        plt.xlabel(h, fontsize=10, fontweight="bold")
        plt.ylabel("Probability Density", fontsize=10, fontweight="bold")
        plt.hist(
            D0[dIdx, :],
            bins=80,
            density=True,
            alpha=0.4,
            label="Male",
            linewidth=0.3,
            edgecolor="black",
        )
        plt.hist(
            D1[dIdx, :],
            bins=80,
            density=True,
            alpha=0.4,
            label="Female",
            linewidth=0.3,
            edgecolor="black",
            color="red",
        )

        plt.legend()
        plt.tight_layout()
        plt.savefig("Distributions/hist_%d.pdf" % dIdx)


def plot_scatter(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

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

    for dIdx1 in range(12):
        for dIdx2 in range(12):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label="Male")
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label="Female")

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            plt.savefig("Scatter/scatter_%d_%d.pdf" % (dIdx1, dIdx2))


def Sw_c(D_c):
    Sw_c = 0
    nc = D_c.shape[1]
    mu_c = mcol(D_c.mean(1))
    DC = D_c - mu_c
    Sw_c = numpy.dot(DC, DC.T) / nc
    return Sw_c


def SbSw(matrix, label):
    Sb = 0  # initialize the between class cov. matrix
    Sw = 0  # initialize the within class cov. matrix
    mu = mcol(matrix.mean(1))  # dataset mean
    N = matrix.shape[1]
    for i in range(
        label.max() + 1
    ):  # in "label" there are only 0,1,2 element so the max will be 2
        D_c = matrix[
            :, label == i
        ]  # filter the matrix data according to the label (0,1,2)
        nc = D_c.shape[1]  # number of sample in class "c"
        mu_c = mcol(
            D_c.mean(1)
        )  # calc a column vector containing the mean of the attributes (sepal-length, petal-width ...) for one class at a time
        Sb = Sb + nc * numpy.dot((mu_c - mu), (mu_c - mu).T)
        Sw = Sw + nc * Sw_c(
            D_c
        )  # calculate the within covariance matrix as a weighted sum of the cov matrix of the classes
    Sb = Sb / N
    Sw = Sw / N

    return Sb, Sw


def LDA1(matrix, label, m):
    Sb, Sw = SbSw(matrix, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]  # reverse the eigenvectors and then retrive the first m
    return W


def plot_LDA_hist(D, L, m):
    W1 = LDA1(D, L, m)  # m = 2 because at most n_classes - 1
    y1 = numpy.dot(W1.T, D)

    D0 = y1[:, L == 0]
    D1 = y1[:, L == 1]

    plt.figure()
    plt.xlabel("LDA Direction")
    plt.hist(D0[0], bins=70, density=True, alpha=0.7, label="Male", edgecolor="black")
    plt.hist(
        D1[0],
        bins=70,
        density=True,
        alpha=0.7,
        label="Female",
        edgecolor="black",
        color="red",
    )
    plt.legend()
    plt.savefig("LDA/lda_hist.pdf")


def plot_heatmap(D, L, cmap_name, filename):
    D = D[:, L]
    heatmap = numpy.zeros((D.shape[0], D.shape[0]))
    
    for f1 in range(D.shape[0]):
        for f2 in range(f1+1):
            coef = abs(sp.stats.pearsonr(D[f1,:], D[f2,:])[0])
            heatmap[f1][f2] = coef
            heatmap[f2][f1] = coef
    
    plt.figure()
    plt.xticks(numpy.arange(D.shape[0]), numpy.arange(1, D.shape[0] + 1))
    plt.yticks(numpy.arange(D.shape[0]), numpy.arange(1, D.shape[0] + 1))
    plt.suptitle("Heatmap of Pearson Correlation")
    plt.imshow(heatmap, cmap=cmap_name)
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def plot_heatmaps_dataset(D):
    cmap_name = "Greys"
    filename = "Heat/correlation_all.png"
    plot_heatmap(D, range(D.shape[1]), cmap_name, filename)

def plot_heatmaps_male(D, L):
    cmap_name = "Blues"
    filename = "Heat/correlation_male.png"
    plot_heatmap(D, L == 0, cmap_name, filename)

def plot_heatmaps_female(D, L):
    cmap_name = "Reds"
    filename = "Heat/correlation_female.png"
    plot_heatmap(D, L == 1, cmap_name, filename)
    
def createCenteredCov(DC):
    C = 0
    for i in range(DC.shape[1]):
        C += numpy.dot(DC[:, i:i+1], DC[:, i:i+1].T)
    
    C /= float(DC.shape[1])
    return C
    
def centerData(D):
    mu = D.mean(1)
    DC = D - mcol(mu)
    return DC

def PCA_plot(D):
    
    N = D.shape[1] 
    mu = mcol(D.mean(1)) 
    DC = D - mu  
    C = numpy.dot(DC, DC.T)/N
    eigenvalues = numpy.linalg.eigh(C)[0]
    
    # Ordina i valori propri in ordine decrescente
    eigenvalues = eigenvalues[::-1]
    
    # Calcola la varianza spiegata per ogni componente principale
    explained_variance = eigenvalues / numpy.sum(eigenvalues)
    y_min, y_max = plt.ylim()
    y_values = numpy.linspace(y_min, y_max, 40)
    plt.yticks(y_values)
    plt.xlim(0,11)
    # Creare un grafico della varianza spiegata
    plt.plot(numpy.cumsum(explained_variance), c='red', marker='.')
    plt.xlabel('Number of components')
    plt.ylabel('Fraction of explained variance')
    plt.grid()
    plt.savefig("PCA/exp_var")

if __name__ == "__main__":
    D, L = load("Train.txt")
    # plot_centered_hist(D, L, "true")
    # plot_scatter(D,L)
    # plot_LDA_hist(D,L,1)
    # plot_heatmaps_dataset(D)
    # plot_heatmaps_male(D,L)
    # plot_heatmaps_female(D,L)
    PCA_plot(D)

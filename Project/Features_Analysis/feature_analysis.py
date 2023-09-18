import numpy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from Preprocessing.LDA import *


def plot_centered_hist(D, L, centered):
    if centered:
        mean_D = numpy.mean(D, axis=1, keepdims=True)
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

    fig, axs = plt.subplots(len(hFea), 1, figsize=(6, 10))

    for dIdx, h in hFea.items():
        ax = axs[dIdx]
        ax.set_xlabel(h, fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")

        ax.hist(
            D0[dIdx, :],
            bins=80,
            density=True,
            alpha=0.4,
            label="Male",
            linewidth=0.3,
            edgecolor="black",
        )
        ax.hist(
            D1[dIdx, :],
            bins=80,
            density=True,
            alpha=0.4,
            label="Female",
            linewidth=0.3,
            edgecolor="black",
            color="red",
        )

        ax.legend()

    plt.tight_layout()
    plt.savefig("Features_Analysis/Histograms/hist_%d.pdf" % dIdx)


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

    plt.figure()

    for dIdx1 in range(12):
        for dIdx2 in range(dIdx1 + 1, 12):
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label="Male")
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label="Female")
            plt.legend()
            plt.tight_layout()

            with plt.style.context("ggplot"):
                plt.savefig(
                    f"Features_Analysis/Scatter_plots/scatter_{dIdx1}_{dIdx2}.pdf"
                )

            plt.clf()

    plt.close()


def plot_LDA_hist(D, L, m):
    W1 = LDA1(D, L, m)
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
    plt.savefig("Features_Analysis/LDA/lda_hist.pdf")


def plot_heatmap(D, L, cmap_name, filename):
    D = D[:, L]
    heatmap = numpy.zeros((D.shape[0], D.shape[0]))

    for i in range(D.shape[0]):
        for j in range(i + 1):
            coef = abs(pearsonr(D[i, :], D[j, :])[0])
            heatmap[i][j] = coef
            heatmap[j][i] = coef

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap=cmap_name)

    ax.set_xticks(numpy.arange(D.shape[0]))
    ax.set_yticks(numpy.arange(D.shape[0]))
    ax.set_xticklabels(numpy.arange(1, D.shape[0] + 1))
    ax.set_yticklabels(numpy.arange(1, D.shape[0] + 1))

    ax.set_title("Heatmap of Pearson Correlation")
    fig.colorbar(im)

    plt.savefig(filename)
    plt.close(fig)


def plot_heatmaps_dataset(D):
    cmap_name = "Greys"
    filename = "Features_Analysis/Heatmap/correlation_all.png"
    plot_heatmap(D, range(D.shape[1]), cmap_name, filename)


def plot_heatmaps_male(D, L):
    cmap_name = "Blues"
    filename = "Features_Analysis/Heatmap/correlation_male.png"
    plot_heatmap(D, L == 0, cmap_name, filename)


def plot_heatmaps_female(D, L):
    cmap_name = "Reds"
    filename = "Features_Analysis/Heatmap/correlation_female.png"
    plot_heatmap(D, L == 1, cmap_name, filename)


def PCA_plot(D):
    N = D.shape[1]
    mu = numpy.mean(D, axis=1, keepdims=True)
    DC = D - mu
    C = numpy.dot(DC, DC.T) / N
    s = numpy.linalg.eigh(C)[0]

    s = s[::-1]

    explained_variance = s / numpy.sum(s)

    fig, ax = plt.subplots()
    ax.set_yticks(numpy.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 40))
    ax.set_xlim(0, 11)

    ax.plot(numpy.cumsum(explained_variance), c="red", marker=".")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Fraction of explained variance")
    ax.grid(True)
    fig.savefig("Features_Analysis/PCA/exp_var")

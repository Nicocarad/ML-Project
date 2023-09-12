from Utils.Kfold import *
from Utils.PCA import *
from Utils.DCF import min_DCF
from Training.GMM.gmm import *
from Utils.Znorm import *
import matplotlib.pyplot as plt





def test_gmm(D,L):
    
    gmm = GMM_TiedDiagonal(4)
    SPost, Label = kfold(gmm, 5, D, L, None)
    res = min_DCF(0.5, 1, 1, Label, SPost)
    print("min_dcf", res)
    
    
    

def GMM_plot_diff_component(D, L):
    min_dcf_values = []
    min_dcf_values_znorm = []
    
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    D = znorm(D)
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_znorm.append(res)    
    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW")
    plt.bar(x_axis + 0.25, min_dcf_values_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label="Znorm")
    

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Std_GMM_RAW+znorm.pdf")
    

    
    
    
    
    
def GMM_plot_diff_component_PCA(D, L, m):
    min_dcf_values_pca = []
    min_dcf_values = []
    
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    D = PCA(D,m)
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_pca.append(res)    
    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM + PCA ")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW")
    plt.bar(x_axis + 0.25, min_dcf_values_pca, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label="PCA m=" +str(m))

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Std_GMM_RAW+PCA"+str(m)+".pdf")
    
    
    
    
    
    
    
def GMM_Tied_plot_diff_component(D, L):
    min_dcf_values = []
    min_dcf_values_znorm = []
    
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    D = znorm(D)
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_znorm.append(res)    
    
    plt.figure()
    plt.xlabel("Tied_GMM components")
    plt.ylabel("minDCF")
    plt.title("Tied GMM")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW")
    plt.bar(x_axis + 0.25, min_dcf_values_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label="Znorm")
    

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Tied_GMM_RAW+znorm.pdf")
    
    
    
    
    
    
def GMM_Tied_plot_diff_component_PCA(D, L, m):
    min_dcf_values_pca = []
    min_dcf_values = []
    
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    D = PCA(D,m)
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_pca.append(res)    
    
    plt.figure()
    plt.xlabel("Tied_GMM components")
    plt.ylabel("minDCF")
    plt.title("Tied GMM + PCA ")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW")
    plt.bar(x_axis + 0.25, min_dcf_values_pca, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label="PCA m = " + str(m))

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Tied_GMM_RAW+PCA" + str(m) + ".pdf")
    
    
    
    
    
    
def GMM_Diagonal_plot_diff_component(D, L):
    min_dcf_values = []
    min_dcf_values_znorm = []
    
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    D = znorm(D)
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_znorm.append(res)    
    
    plt.figure()
    plt.xlabel("Diagonal_GMM components")
    plt.ylabel("minDCF")
    plt.title("Diagonal GMM")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW")
    plt.bar(x_axis + 0.25, min_dcf_values_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label="Znorm")
    

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Diagonal_GMM_RAW+znorm.pdf")
    
    
    
    
def GMM_Diagonal_plot_diff_component_PCA(D, L):
    min_dcf_values_pca = []
    min_dcf_values = []
    
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)

        min_dcf_values.append(res)
        
    D = PCA(D,11)
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)

        min_dcf_values_pca.append(res)    
    
    plt.figure()
    plt.xlabel("Tied_GMM components")
    plt.ylabel("minDCF")
    plt.title("Diagonal GMM + PCA ")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW")
    plt.bar(x_axis + 0.25, min_dcf_values_pca, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label="PCA m=11")

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Diagonal_GMM_RAW+PCA11.pdf")
    
    
    
    
def GMM_TiedDiagonal_plot_diff_component(D, L):
    min_dcf_values = []
    min_dcf_values_znorm = []
    
    for i in range(5):
        gmm = GMM_TiedDiagonal(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)

        min_dcf_values.append(res)
        
    D = znorm(D)
    for i in range(5):
        gmm = GMM_TiedDiagonal(i)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)

        min_dcf_values_znorm.append(res)    
    
    plt.figure()
    plt.xlabel("Tied Diagonal_GMM components")
    plt.ylabel("minDCF")
    plt.title("Tied Diagonal GMM")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW")
    plt.bar(x_axis + 0.25, min_dcf_values_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Orange",
            label="Znorm")
    

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Tied_Diagonal_GMM_RAW+znorm.pdf")
    
    
    
    
def GMM_diff_priors(D,L):
    
    for i in range(1,3):
        for pi in [0.5, 0.1, 0.9]:
            gmm = GMM(i)
            SPost, Label = kfold(gmm, 5, D, L, None)
            res = min_DCF(pi, 1, 1, Label, SPost)
            print("GMM min_DCF pi = ", pi, str(2**i) + " components: ", round(res,3))
    
    for pi in [0.5, 0.1, 0.9]:
        gmm = GMM_Tied(3)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print("Tied_GMM min_DCF pi = ", pi, str(2**3) + " components : ", round(res,3))
        
    
    D = PCA(D,11)
    
    for pi in [0.5, 0.1, 0.9]:
        gmm = GMM(2)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print("min_DCF pi = ", pi, str(2**i) + " components + PCA(11): ", round(res,3))
    
    
    
def GMM_diff_priors_zscore(D,L):

    D = znorm(D)
    for i in range(1,3):
        for pi in [0.5, 0.1, 0.9]:
            gmm = GMM(i)
            SPost, Label = kfold(gmm, 5, D, L, None)
            res = min_DCF(pi, 1, 1, Label, SPost)
            print("GMM min_DCF pi = ", pi, str(2**i) + " components: + znorm", round(res,3))
    
    for pi in [0.5, 0.1, 0.9]:
        gmm = GMM_Tied(3)
        SPost, Label = kfold(gmm, 5, D, L, None)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print("Tied_GMM min_DCF pi = ", pi, str(2**3) + " components : + znorm", round(res,3))
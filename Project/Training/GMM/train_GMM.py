from Utils.Kfold import *
from Utils.PCA import *
from Utils.DCF import min_DCF
from Training.GMM.gmm import *
from Utils.Znorm import *
import matplotlib.pyplot as plt





def test_gmm(D,L):
    
    gmm = GMM_TiedDiagonal(4)
    SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
    res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
    print("min_dcf",res_3)
    
    
    


def GMM_plot_diff_component(D, L):
    min_dcf_values = []
    min_dcf_values_znorm = []
    
    for i in range(5):
        gmm = GMM(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values.append(res_3)
        
    D = znorm(D)
    for i in range(5):
        gmm = GMM(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values_znorm.append(res_3)    
    
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
    

    
    
    
    
    
def GMM_plot_diff_component_PCA(D, L):
    min_dcf_values_pca = []
    min_dcf_values = []
    
    for i in range(5):
        gmm = GMM(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values.append(res_3)
        
    D = PCA(D,10)
    for i in range(5):
        gmm = GMM(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values_pca.append(res_3)    
    
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
            label="PCA m=10")

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Std_GMM_RAW+PCA10.pdf")
    
    
    
    
    
    
    
def GMM_Tied_plot_diff_component(D, L):
    min_dcf_values = []
    min_dcf_values_znorm = []
    
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values.append(res_3)
        
    D = znorm(D)
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values_znorm.append(res_3)    
    
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
    
    
    
    
    
    
def GMM_Tied_plot_diff_component_PCA(D, L):
    min_dcf_values_pca = []
    min_dcf_values = []
    
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values.append(res_3)
        
    D = PCA(D,10)
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values_pca.append(res_3)    
    
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
            label="PCA m=10")

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Tied_GMM_RAW+PCA10.pdf")
    
    
    
    
    
    
def GMM_Diagonal_plot_diff_component(D, L):
    min_dcf_values = []
    min_dcf_values_znorm = []
    
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values.append(res_3)
        
    D = znorm(D)
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values_znorm.append(res_3)    
    
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
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values.append(res_3)
        
    D = PCA(D,11)
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        print("min_dcf",res_3)
        min_dcf_values_pca.append(res_3)    
    
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
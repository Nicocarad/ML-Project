from Utils.Kfold import *
from Utils.PCA import *
from Utils.DCF import min_DCF
from Training.GMM.gmm import *
from Utils.Znorm import *
import matplotlib.pyplot as plt





def GMM_plot_diff_component(DTR, LTR, DTE, LTE):
    min_dcf_values = []
    min_dcf_values_znorm = []
    min_dcf_values_eval = []
    min_dcf_values_eval_znorm = []
    
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    for i in range(5):
        gmm = GMM(i)
        gmm.train(DTR,LTR,DTE,LTE)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval.append(res)
        
    DTR = znorm(DTR)
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_znorm.append(res)    
        
    DTE = znorm(DTE)
    for i in range(5):
        gmm = GMM(i)
        gmm.train(DTR,LTR,DTE,LTE)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval_znorm.append(res)
    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(x_axis + 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW (VAL)")
    plt.bar(x_axis + 0.25, min_dcf_values_eval, width=0.25, linewidth=1.0, edgecolor='black', color="Red", ,
            label="RAW (EVAL)")
    plt.bar(x_axis + 0.50, min_dcf_values_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Blue",
            label="Znorm (VAL)")
    plt.bar(x_axis + 0.75, min_dcf_values_eval_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Blue",
            label="Znorm (EVAL)")
    

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Training/GMM/Plot/Std_GMM_RAW+znorm.pdf")
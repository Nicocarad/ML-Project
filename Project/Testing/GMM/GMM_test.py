from Utils.Kfold import *
from Preprocessing.PCA import *
from Metrics.DCF import min_DCF
from Models.GMM.gmm import *
from Preprocessing.Znorm import *
import matplotlib.pyplot as plt
from Calibration.calibration import *





def GMM_plot_diff_component_eval(DTR, LTR, DTE, LTE):
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
        gmm.train(DTR,LTR,DTE,LTE,None)
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
        gmm.train(DTR,LTR,DTE,LTE,None)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval_znorm.append(res)
    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM")
    iterations = range(5)
    x_axis = np.arange(len(iterations)) * 1.25
    bounds = np.array(iterations)
    plt.bar(x_axis - 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW (VAL)")
    plt.bar(x_axis + 0.25, min_dcf_values_eval, width=0.25, linewidth=1.0, edgecolor='black', color="Red", hatch=".",
            label="RAW (EVAL)")
    plt.bar(x_axis + 0.50, min_dcf_values_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Blue",
            label="Znorm (VAL)")
    plt.bar(x_axis + 0.75, min_dcf_values_eval_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Blue", hatch=".",
            label="Znorm (EVAL)")
    

    plt.xticks([r*1.25 + 0.375 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Testing\GMM\Plot\Std_GMM_val_eval+znorm.pdf")
    
    
    
    
def GMM_plot_diff_component_eval_PCA(DTR, LTR, DTE, LTE):
    min_dcf_values = []
    min_dcf_values_pca = []
    min_dcf_values_eval = []
    min_dcf_values_eval_pca = []
    
    DTR, DTE = PCA(DTR,11,DTE)
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    for i in range(5):
        gmm = GMM(i)
        gmm.train(DTR,LTR,DTE,LTE,None)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval.append(res)
        
    
    for i in range(5):
        gmm = GMM(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_pca.append(res)    
        
   
    for i in range(5):
        gmm = GMM(i)
        gmm.train(DTR,LTR,DTE,LTE,None)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval_pca.append(res)
    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM + PCA")
    iterations = range(5)
    x_axis = np.arange(len(iterations)) * 1.25
    bounds = np.array(iterations)
    plt.bar(x_axis - 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW (VAL)")
    plt.bar(x_axis + 0.25, min_dcf_values_eval, width=0.25, linewidth=1.0, edgecolor='black', color="Red", hatch=".",
            label="RAW (EVAL)")
    plt.bar(x_axis + 0.50, min_dcf_values_pca, width=0.25, linewidth=1.0, edgecolor='black', color="Blue",
            label="RAW + PCA11 (VAL)")
    plt.bar(x_axis + 0.75, min_dcf_values_eval_pca, width=0.25, linewidth=1.0, edgecolor='black', color="Blue", hatch=".",
            label="RAW + PCA11 (EVAL)")
    

    plt.xticks([r*1.25 + 0.375 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Testing\GMM\Plot\Std_GMM_val_eval+PCA11.pdf")
    
    
    
    
    
    
def GMM_Tied_plot_diff_component_eval_PCA(DTR, LTR, DTE, LTE):
    min_dcf_values = []
    min_dcf_values_pca = []
    min_dcf_values_eval = []
    min_dcf_values_eval_pca = []
    
    DTR, DTE = PCA(DTR,11,DTE)
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    for i in range(5):
        gmm = GMM_Tied(i)
        gmm.train(DTR,LTR,DTE,LTE,None)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval.append(res)
        
    
    for i in range(5):
        gmm = GMM_Tied(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_pca.append(res)    
        
   
    for i in range(5):
        gmm = GMM_Tied(i)
        gmm.train(DTR,LTR,DTE,LTE,None)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval_pca.append(res)
    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("Tied GMM + PCA")
    iterations = range(5)
    x_axis = np.arange(len(iterations)) * 1.25
    bounds = np.array(iterations)
    plt.bar(x_axis - 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW (VAL)")
    plt.bar(x_axis + 0.25, min_dcf_values_eval, width=0.25, linewidth=1.0, edgecolor='black', color="Red", hatch=".",
            label="RAW (EVAL)")
    plt.bar(x_axis + 0.50, min_dcf_values_pca, width=0.25, linewidth=1.0, edgecolor='black', color="Blue",
            label="RAW + PCA11 (VAL)")
    plt.bar(x_axis + 0.75, min_dcf_values_eval_pca, width=0.25, linewidth=1.0, edgecolor='black', color="Blue", hatch=".",
            label="RAW + PCA11 (EVAL)")
    

    plt.xticks([r*1.25 + 0.375 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Testing\GMM\Plot\Tied_GMM_val_eval+PCA11.pdf")
    
    
    
    
def GMM_Diagonal_plot_diff_component_eval_znorm(DTR, LTR, DTE, LTE):
    min_dcf_values = []
    min_dcf_values_znorm = []
    min_dcf_values_eval = []
    min_dcf_values_eval_znorm = []
    
    
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
    for i in range(5):
        gmm = GMM_Diagonal(i)
        gmm.train(DTR,LTR,DTE,LTE,None)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval.append(res)
        
    DTR = znorm(DTR)
    DTE = znorm(DTE)
    for i in range(5):
        gmm = GMM_Diagonal(i)
        SPost, Label = kfold(gmm, 5, DTR, LTR, None)
        res = min_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values_znorm.append(res)    
        
   
    for i in range(5):
        gmm = GMM_Diagonal(i)
        gmm.train(DTR,LTR,DTE,LTE,None)
        gmm.compute_scores()
        scores = gmm.scores
        res = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_values_eval_znorm.append(res)
    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("Diagonal GMM")
    iterations = range(5)
    x_axis = np.arange(len(iterations)) * 1.25
    bounds = np.array(iterations)
    plt.bar(x_axis - 0.00, min_dcf_values, width=0.25, linewidth=1.0, edgecolor='black', color="Red",
            label="RAW (VAL)")
    plt.bar(x_axis + 0.25, min_dcf_values_eval, width=0.25, linewidth=1.0, edgecolor='black', color="Red", hatch=".",
            label="RAW (EVAL)")
    plt.bar(x_axis + 0.50, min_dcf_values_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Blue",
            label="Znorm (VAL)")
    plt.bar(x_axis + 0.75, min_dcf_values_eval_znorm, width=0.25, linewidth=1.0, edgecolor='black', color="Blue", hatch=".",
            label="Znorm (EVAL)")
    

    plt.xticks([r*1.25 + 0.375 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    
    plt.savefig("Testing\GMM\Plot\Diagonal_GMM_val_eval+znorm.pdf")
    
    
    
    
    
def GMM_test_compare(DTR,LTR,DTE,LTE):
    
    gmm = GMM(2)
    gmm.train(DTR,LTR,DTE,LTE,None)
    gmm.compute_scores()
    scores = gmm.scores
    llr_cal,Label_cal = calibration(scores,LTE,0.5)
    res = min_DCF(0.5,1,1,Label_cal,llr_cal)
    print("GMM 4 components min_DCF", res)
    
    DTR_pca,DTE_pca = PCA(DTR,11,DTE)
    gmm = GMM(2)
    gmm.train(DTR_pca,LTR,DTE_pca,LTE,None)
    gmm.compute_scores()
    scores = gmm.scores
    llr_cal,Label_cal = calibration(scores,LTE,0.5)
    res = min_DCF(0.5,1,1,Label_cal,llr_cal)
    print("GMM 4 components + PCA11 min_DCF", res)
    
    DTR_pca,DTE_pca = PCA(DTR,11,DTE)
    gmm = GMM_Tied(2)
    gmm.train(DTR_pca,LTR,DTE_pca,LTE,None)
    gmm.compute_scores()
    scores = gmm.scores
    llr_cal,Label_cal = calibration(scores,LTE,0.5)
    res = min_DCF(0.5,1,1,LTE,scores)
    print("GMM_Tied 4 components + PCA11 min_DCF", res)
    
    
    
    
    
def GMM_test_best(DTR, DTE, LTR, LTE):
    DTR, DTE = PCA(DTR, 11, DTE)
    iterations = 2
    gmm = GMM(iterations)
    gmm.train(DTR, LTR, DTE, LTE, None)
    gmm.compute_scores()
    scores = gmm.scores

    return scores, LTE
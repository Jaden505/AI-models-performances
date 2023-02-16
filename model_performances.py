import matplotlib.pyplot as plt
import numpy as np

class ModelPerformances:
    def __init__(self, dtc_eval, rfc_eval, xgb_eval, knn_eval):
        self.dtc_eval = dtc_eval
        self.rfc_eval = rfc_eval
        self.xgb_eval = xgb_eval
        self.knn_eval = knn_eval

    def create_fig(self):
        # Intitialize figure with two plots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

        fig.set_figheight(7)
        fig.set_figwidth(14)
        fig.set_facecolor('white')

        return fig, ax1, ax2

    def bar_plot(self, ax1):
        # First plot
        ## set bar size
        barWidth = 0.2
        dtc_score = [self.dtc_eval['acc'], self.dtc_eval['prec'], self.dtc_eval['rec'], self.dtc_eval['f1'], self.dtc_eval['kappa']]
        rf_score = [self.rfc_eval['acc'], self.rfc_eval['prec'], self.rfc_eval['rec'], self.rfc_eval['f1'], self.rfc_eval['kappa']]
        nb_score = [self.xgb_eval['acc'], self.xgb_eval['prec'], self.xgb_eval['rec'], self.xgb_eval['f1'], self.xgb_eval['kappa']]
        knn_score = [self.knn_eval['acc'], self.knn_eval['prec'], self.knn_eval['rec'], self.knn_eval['f1'], self.knn_eval['kappa']]

        ## Set position of bar on X axis
        r1 = np.arange(len(dtc_score))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]

        ## Make the plot
        ax1.bar(r1, dtc_score, width=barWidth, edgecolor='white', label='Decision Tree')
        ax1.bar(r2, rf_score, width=barWidth, edgecolor='white', label='Random Forest')
        ax1.bar(r3, nb_score, width=barWidth, edgecolor='white', label='Naive Bayes')
        ax1.bar(r4, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')

        ## Configure x and y axis
        ax1.set_xlabel('Metrics', fontweight='bold')
        labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
        ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(dtc_score))], )
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_ylim(0, 1)

        ## Create legend & title
        ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
        ax1.legend()

    def roc_plot(self, ax2):
        # Second plot
        ## Comparing ROC Curve
        ax2.plot(self.dtc_eval['fpr'], self.dtc_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(self.dtc_eval['auc']))
        ax2.plot(self.rfc_eval['fpr'], self.rfc_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(self.rfc_eval['auc']))
        ax2.plot(self.xgb_eval['fpr'], self.xgb_eval['tpr'], label='Naive Bayes, auc = {:0.5f}'.format(self.xgb_eval['auc']))
        ax2.plot(self.knn_eval['fpr'], self.knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(self.knn_eval['auc']))

        ## Configure x and y axis
        ax2.set_xlabel('False Positive Rate', fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontweight='bold')

        ## Create legend & title
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc=4)


    def display(self):
        plt.show()
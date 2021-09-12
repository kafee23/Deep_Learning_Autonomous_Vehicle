# -*- coding: utf-8 -*-
"""

"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
import string
import numpy as np


class Model_Evaluation:
    def __init__(self, name):
        self.name = name

    def ROC_Curve_Generator(self, df, no_of_output, offset):
            

        for x in range(no_of_output):

            plt.figure()            
            # plot no skill roc curve
            plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
            # calculate roc curve for model
            fpr, tpr, _ = roc_curve(df.iloc[:,x], df.iloc[:, x+offset])
            # plot model roc curve
            plt.plot(fpr, tpr, marker='.', label='ROC_DNN_'+df.columns[x])
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend()
            # show the plot
            plt.show()

    def ROC_Curve_Generator_Subplot(self, df, no_of_output, offset):
            
        fig,a =  plt.subplots(1,3,squeeze=False, figsize=(18,4))
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        
        for x in range(no_of_output):
                      
            # plot no skill roc curve
            a[0, x].plot([0, 1], [0, 1], linestyle='--', label='No_Skill_' + df.columns[x])
            # calculate roc curve for model
            fpr, tpr, _ = roc_curve(df.iloc[:,x], df.iloc[:, x+offset])
            # plot model roc curve
            a[0, x].plot(fpr, tpr, marker='.', label='ROC_'+df.columns[x])
            # axis labels
            a[0, x].set_xlabel('False Positive Rate', fontsize = 15)
            a[0, x].set_ylabel('True Positive Rate', fontsize = 15)
            # show the legend
            a[0, x].legend(fontsize = 15)
            a[0, x].text(0.6, 0.4, "AUC = " + "{:.2f}".format(round(roc_auc_score(df.iloc[:,x], df.iloc[:,x+(offset*2)]), 2)), transform=a[0, x].transAxes, fontsize=14, verticalalignment='top', bbox=props)
            a[0, x].text(0.5, -0.3, "("+string.ascii_lowercase[x]+")", transform=a[0, x].transAxes, size=20)

        filename = 'ROC_AUC_Curve_'+str(no_of_output)+'.png'
        plt.savefig(filename, format="png", bbox_inches='tight', figsize=(9, 11))

        # show the plot
        plt.show()
        
    def ROC_Curve_Generator_Subplot_Comparison(self, df, df_SVM, df_LSTM, no_of_output, offset):
            
        fig,a =  plt.subplots(1,3,squeeze=False, figsize=(18,4))
        
        for x in range(no_of_output):
                      
            # plot no skill roc curve
            #a[0, x].plot([0, 1], [0, 1], linestyle='--', label='No_Skill_' + df.columns[x])
            # calculate roc curve for model
            fpr, tpr, _ = roc_curve(df.iloc[:,x], df.iloc[:, x+offset])
            # plot model roc curve
            a[0, x].plot(fpr, tpr, marker='.', label='ROC_'+df.columns[x]+'_DNN')
            
            # calculate roc curve for model
            fpr_SVC, tpr_SVC, _ = roc_curve(df_SVM.iloc[:,x], df_SVM.iloc[:, x+offset])
            # plot model roc curve
            a[0, x].plot(fpr_SVC, tpr_SVC, marker='.', label='ROC_'+df.columns[x]+'_SVM')
            
            # calculate roc curve for model
            fpr_LSTM, tpr_LSTM, _ = roc_curve(df_LSTM.iloc[:,x], df_LSTM.iloc[:, x+offset])
            # plot model roc curve
            a[0, x].plot(fpr_LSTM, tpr_LSTM, marker='.', label='ROC_'+df.columns[x]+'_LSTM')
            
            
            # axis labels
            a[0, x].set_xlabel('False Positive Rate', fontsize = 15)
            a[0, x].set_ylabel('True Positive Rate', fontsize = 15)
            # show the legend
            a[0, x].legend(fontsize = 12)
            a[0, x].text(0.55, 0.65, "AUC_DNN = " + "{:.2f}".format(round(roc_auc_score(df.iloc[:,x], df.iloc[:,x+(offset*2)]), 2)), transform=a[0, x].transAxes, fontsize=14, verticalalignment='top')#, bbox=props)
            a[0, x].text(0.55, 0.55, "AUC_SVM = " + "{:.2f}".format(round(roc_auc_score(df_SVM.iloc[:,x], df_SVM.iloc[:,x+(offset*2)]), 2)), transform=a[0, x].transAxes, fontsize=14, verticalalignment='top')#, bbox=props)
            a[0, x].text(0.55, 0.45, "AUC_LSTM = " + "{:.2f}".format(round(roc_auc_score(df_LSTM.iloc[:,x], df_LSTM.iloc[:,x+(offset*2)]), 2)), transform=a[0, x].transAxes, fontsize=14, verticalalignment='top')#, bbox=props)
            
            a[0, x].text(0.5, -0.3, "("+string.ascii_lowercase[x]+")", transform=a[0, x].transAxes, size=20)

        filename = 'ROC_AUC_Curve_'+str(no_of_output)+'.png'
        plt.savefig(filename, format="png", bbox_inches='tight', figsize=(9, 11))

        # show the plot
        plt.show()

    def PR_Curve_Generator(self, df, no_of_output, offset, no_skill_df, nsd_offset):
        for x in range(no_of_output):
            print(f'Average Precision Score for {df.columns[x]} is {average_precision_score(df.iloc[:,x], df.iloc[:, x+offset])}')
            
            plt.figure()

            y = no_skill_df.iloc[:, x + nsd_offset]
            no_skill = len(y[y==1]) / len(y)
            # plot the no skill precision-recall curve
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

            precision, recall, _ = precision_recall_curve(df.iloc[:,x], df.iloc[:, x+offset])
            # plot the model precision-recall curve
            plt.plot(recall, precision, marker='.', label='PR_DNN_'+df.columns[x])
            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # show the legend
            plt.legend()
            # show the plot
            plt.show()
    
    def PR_Curve_Generator_Subplot(self, df, no_of_output, offset, no_skill_df, nsd_offset):
        fig,a =  plt.subplots(1,3,squeeze=False, figsize=(18,4))
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        for x in range(no_of_output):
            print(f'Average Precision Score for {df.columns[x]} is {average_precision_score(df.iloc[:,x], df.iloc[:, x+offset])}')

            y = no_skill_df.iloc[:, x + nsd_offset]
            no_skill = len(y[y==1]) / len(y)
            # plot the no skill precision-recall curve
            a[0, x].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

            precision, recall, _ = precision_recall_curve(df.iloc[:,x], df.iloc[:, x+offset])
            # plot the model precision-recall curve
            a[0, x].plot(recall, precision, marker='.', label='PR_DNN_'+df.columns[x])
            # axis labels
            a[0, x].set_xlabel('Recall', fontsize = 15)
            a[0, x].set_ylabel('Precision', fontsize = 15)
            # show the legend
            a[0, x].legend(fontsize = 13)
            
            a[0, x].text(0.1, 0.3, "Average Precision = " + "{:.2f}".format(round(average_precision_score(df.iloc[:,x], df.iloc[:, x+offset]), 2)), transform=a[0, x].transAxes, fontsize=14, verticalalignment='top', bbox=props)
            a[0, x].text(0.5, -0.3, "("+string.ascii_lowercase[x]+")", transform=a[0, x].transAxes, size=20)

        filename = 'PR_Curve_'+str(no_of_output)+'.png'
        plt.savefig(filename, format="png", bbox_inches='tight', figsize=(9, 11))

        # show the plot
        plt.show() 
        
    def PR_Curve_Generator_Subplot_Comparison(self, df, df_SVM, df_LSTM, no_of_output, offset, no_skill_df, nsd_offset):
        fig,a =  plt.subplots(1,3,squeeze=False, figsize=(18,4))
       
        for x in range(no_of_output):
            print(f'Average Precision Score for {df.columns[x]} is {average_precision_score(df.iloc[:,x], df.iloc[:, x+offset])}')

            precision, recall, _ = precision_recall_curve(df.iloc[:,x], df.iloc[:, x+offset])
            # plot the model precision-recall curve
            a[0, x].plot(recall, precision, marker='.', label='PR_'+df.columns[x]+'_DNN')
            
            precision_SVC, recall_SVC, _ = precision_recall_curve(df_SVM.iloc[:,x], df_SVM.iloc[:, x+offset])
            a[0, x].plot(recall_SVC, precision_SVC, marker='.', label='PR_'+df.columns[x]+'_SVM')
            
            precision_LSTM, recall_LSTM, _ = precision_recall_curve(df_LSTM.iloc[:,x], df_LSTM.iloc[:, x+offset])
            a[0, x].plot(recall_LSTM, precision_LSTM, marker='.', label='PR_'+df.columns[x]+'_LSTM')
            
            
            # axis labels
            a[0, x].set_xlabel('Recall', fontsize = 15)
            a[0, x].set_ylabel('Precision', fontsize = 15)
            # show the legend
            a[0, x].legend(fontsize = 12, loc = 'lower right')
            
            a[0, x].set_xlim(0.01, 1.0)
            a[0, x].set_ylim(-0.1, 1.0)
        
            a[0, x].text(0.05, 0.58, "Average Precision (DNN) = " + "{:.2f}".format(round(average_precision_score(df.iloc[:,x], df.iloc[:, x+offset]), 2)), transform=a[0, x].transAxes, fontsize=12, verticalalignment='top')#bbox=props)
            a[0, x].text(0.05, 0.49, "Average Precision (SVM) = " + "{:.2f}".format(round(average_precision_score(df_SVM.iloc[:,x], df_SVM.iloc[:, x+offset]), 2)), transform=a[0, x].transAxes, fontsize=12, verticalalignment='top')#, bbox=props)
            a[0, x].text(0.05, 0.40, "Average Precision (LSTM) = " + "{:.2f}".format(round(average_precision_score(df_LSTM.iloc[:,x], df_LSTM.iloc[:, x+offset]), 2)), transform=a[0, x].transAxes, fontsize=12, verticalalignment='top')#, bbox=props)
            
            a[0, x].text(0.5, -0.3, "("+string.ascii_lowercase[x]+")", transform=a[0, x].transAxes, size=20)

        filename = 'PR_Curve_'+str(no_of_output)+'.png'
        plt.savefig(filename, format="png", bbox_inches='tight', figsize=(9, 11))

        # show the plot
        plt.show()             
    def metrics_printer(self, df, no_of_output, offset):
        for x in range(no_of_output):
            print(f'Accuracy Score for {df.columns[x]} is {accuracy_score(df.iloc[:,x], df.iloc[:,x+offset])}')
            print(f'Precision Score for {df.columns[x]} is {precision_score(df.iloc[:,x], df.iloc[:,x+offset])}')
            print(f'Recall Score for {df.columns[x]} is {recall_score(df.iloc[:,x], df.iloc[:,x+offset])}')
            print(f'F1 Score for {df.columns[x]} is {f1_score(df.iloc[:,x], df.iloc[:,x+offset])}')
            print(f'Cohen Kappa Score for {df.columns[x]} is {cohen_kappa_score(df.iloc[:,x], df.iloc[:,x+offset])}')
            print(f'ROC AUC Score for {df.columns[x]} is {roc_auc_score(df.iloc[:,x], df.iloc[:,x+offset])}')
            print(f'Confusion Matrix for {df.columns[x]} is {confusion_matrix(df.iloc[:,x], df.iloc[:,x+offset])}')    
            
   
    def metrics_file_writer(self, file_name, df, no_of_output, offset):
        with open(file_name, 'w') as f: 
            for x in range(no_of_output):
                print(f'Accuracy Score for {df.columns[x]} is {accuracy_score(df.iloc[:,x], df.iloc[:,x+offset])}', file = f)
                print(f'Precision Score for {df.columns[x]} is {precision_score(df.iloc[:,x], df.iloc[:,x+offset])}', file = f)
                print(f'Recall Score for {df.columns[x]} is {recall_score(df.iloc[:,x], df.iloc[:,x+offset])}', file = f)
                print(f'F1 Score for {df.columns[x]} is {f1_score(df.iloc[:,x], df.iloc[:,x+offset])}', file = f)
                print(f'Cohen Kappa Score for {df.columns[x]} is {cohen_kappa_score(df.iloc[:,x], df.iloc[:,x+offset])}', file = f)
                print(f'ROC AUC Score for {df.columns[x]} is {roc_auc_score(df.iloc[:,x], df.iloc[:,x+offset])}', file = f)
                print(f'Confusion Matrix for {df.columns[x]} is {confusion_matrix(df.iloc[:,x], df.iloc[:,x+offset])}', file = f)
    
    def figure_generator_single_output(self, df, no_of_output, offset, no_skill_df, nsd_offset):
        fig,a =  plt.subplots(1, 2, squeeze=False, figsize=(18,4))
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        
        x = 0
                      
        # plot no skill roc curve
        a[0, 0].plot([0, 1], [0, 1], linestyle='--', label='No_Skill_' + df.columns[x])
        # calculate roc curve for model
        fpr, tpr, _ = roc_curve(df.iloc[:,x], df.iloc[:, x+offset])
        # plot model roc curve
        a[0, 0].plot(fpr, tpr, marker='.', label='ROC_'+df.columns[x])
        # axis labels
        a[0, 0].set_xlabel('False Positive Rate', fontsize = 15)
        a[0, 0].set_ylabel('True Positive Rate', fontsize = 15)
        # show the legend
        a[0, 0].legend(fontsize = 15)
        a[0, 0].text(0.6, 0.4, "AUC = " + "{:.2f}".format(round(roc_auc_score(df.iloc[:,x], df.iloc[:,x+(offset*2)]), 2)), transform=a[0, 0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        a[0, 0].text(0.5, -0.3, "("+string.ascii_lowercase[x]+")", transform=a[0, 0].transAxes, size=20)
        a[0, 0].set_xlim(-0.004, 1.0)
        a[0, 0].set_ylim(-0.01, 1.01)
        
        print(f'Average Precision Score for {df.columns[x]} is {average_precision_score(df.iloc[:,x], df.iloc[:, x+offset])}')
     
        y = no_skill_df.iloc[:, x + nsd_offset]
        no_skill = len(y[y==1]) / len(y)
        # plot the no skill precision-recall curve
        a[0, 1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

        precision, recall, _ = precision_recall_curve(df.iloc[:,x], df.iloc[:, x+offset])
        # plot the model precision-recall curve
        a[0, 1].plot(recall, precision, marker='.', label='PR_DNN_'+df.columns[x])
        # axis labels
        a[0, 1].set_xlabel('Recall', fontsize = 15)
        a[0, 1].set_ylabel('Precision', fontsize = 15)
        # show the legend
        a[0, 1].legend(fontsize = 13)
        
        a[0, 1].set_xlim(0.0, 1.0)
        a[0, 1].set_ylim(0.4, 1.01)
    
        a[0, 1].text(0.1, 0.3, "Average Precision = " + "{:.2f}".format(round(average_precision_score(df.iloc[:,x], df.iloc[:, x+offset]), 2)), transform=a[0, 1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        a[0, 1].text(0.5, -0.3, "("+string.ascii_lowercase[1]+")", transform=a[0, 1].transAxes, size=20)

        filename = 'ROC_PR_Curve_'+str(no_of_output)+'.png'
        plt.savefig(filename, format="png", bbox_inches='tight', figsize=(9, 11))

        # show the plot
        plt.show() 
        
    def figure_generator_single_output_comparison(self, df, df_SVM, df_LSTM, no_of_output, offset, no_skill_df, nsd_offset):
        fig,a =  plt.subplots(1, 2, squeeze=False, figsize=(18,4))
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        
        x = 0

        fpr, tpr, _ = roc_curve(df.iloc[:,x], df.iloc[:, x+offset])
        # plot model roc curve
        a[0, 0].plot(fpr, tpr, marker='.', label='ROC_DNN')
        # axis labels
        
        fpr_SVM, tpr_SVM, _ = roc_curve(df_SVM.iloc[:,x], df_SVM.iloc[:, x+offset])
        # plot model roc curve
        a[0, 0].plot(fpr_SVM, tpr_SVM, marker='.', label='ROC_SVM')
        
        
        fpr_LSTM, tpr_LSTM, _ = roc_curve(df_LSTM.iloc[:,x], df_LSTM.iloc[:, x+offset])
        # plot model roc curve
        a[0, 0].plot(fpr_LSTM, tpr_LSTM, marker='.', label='ROC_LSTM')
        
        # axis labels
        
        
        a[0, 0].set_xlabel('False Positive Rate', fontsize = 15)
        a[0, 0].set_ylabel('True Positive Rate', fontsize = 15)
        # show the legend
        a[0, 0].legend(fontsize = 15)
        a[0, 0].text(0.6, 0.5, "AUC_DNN = " + "{:.2f}".format(round(roc_auc_score(df.iloc[:,x], df.iloc[:,x+(offset*2)]), 2)), transform=a[0, 0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        a[0, 0].text(0.6, 0.4, "AUC_SVM = " + "{:.2f}".format(round(roc_auc_score(df_SVM.iloc[:,x], df_SVM.iloc[:,x+(offset*2)]), 2)), transform=a[0, 0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        a[0, 0].text(0.6, 0.3, "AUC_LSTM = " + "{:.2f}".format(round(roc_auc_score(df_LSTM.iloc[:,x], df_LSTM.iloc[:,x+(offset*2)]), 2)), transform=a[0, 0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        
        
        a[0, 0].text(0.5, -0.3, "("+string.ascii_lowercase[x]+")", transform=a[0, 0].transAxes, size=20)
        
        a[0, 0].set_xlim(-0.004, 1.0)
        a[0, 0].set_ylim(-0.01, 1.01)
        
        print(f'Average Precision Score for {df.columns[x]} is {average_precision_score(df.iloc[:,x], df.iloc[:, x+offset])}')
     
        precision, recall, _ = precision_recall_curve(df.iloc[:,x], df.iloc[:, x+offset])
        # plot the model precision-recall curve
        a[0, 1].plot(recall, precision, marker='.', label='PR_DNN')
        
        precision_SVM, recall_SVM, _ = precision_recall_curve(df_SVM.iloc[:,x], df_SVM.iloc[:, x+offset])
        # plot the model precision-recall curve
        a[0, 1].plot(recall_SVM, precision_SVM, marker='.', label='PR_SVM')  
        
        precision_LSTM, recall_LSTM, _ = precision_recall_curve(df_LSTM.iloc[:,x], df_LSTM.iloc[:, x+offset])
        # plot the model precision-recall curve
        a[0, 1].plot(recall_LSTM, precision_LSTM, marker='.', label='PR_LSTM')  
        
        # axis labels
        a[0, 1].set_xlabel('Recall', fontsize = 15)
        a[0, 1].set_ylabel('Precision', fontsize = 15)
        # show the legend
        a[0, 1].legend(fontsize = 13)
        
        a[0, 1].set_xlim(0.05, 1.0)
        a[0, 1].set_ylim(0.0, 1.01)
    
        a[0, 1].text(0.4, 0.4, "Average Precision (DNN) = " + "{:.2f}".format(round(average_precision_score(df.iloc[:,x], df.iloc[:, x+offset]), 2)), transform=a[0, 1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        a[0, 1].text(0.4, 0.3, "Average Precision (SVM) = " + "{:.2f}".format(round(average_precision_score(df_SVM.iloc[:,x], df_SVM.iloc[:, x+offset]), 2)), transform=a[0, 1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        a[0, 1].text(0.4, 0.2, "Average Precision (LSTM) = " + "{:.2f}".format(round(average_precision_score(df_LSTM.iloc[:,x], df_LSTM.iloc[:, x+offset]), 2)), transform=a[0, 1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        
        a[0, 1].text(0.5, -0.3, "("+string.ascii_lowercase[1]+")", transform=a[0, 1].transAxes, size=20)

        filename = 'ROC_PR_Curve_'+str(no_of_output)+'.png'
        plt.savefig(filename, format="png", bbox_inches='tight', figsize=(9, 11))

        # show the plot
        plt.show() 
    
    def box_plot_generator(self, df, columns, df_car, columns_car):        
               
        fig, axes = plt.subplots(1, 4, squeeze=False, figsize=(24,4))
        flierprops = dict(marker='o', markerfacecolor="0.5", markersize=3,
                  linestyle='none', markeredgecolor="0.5")
   
        ax = (df[columns]).boxplot(by = columns[0], ax=axes.flatten()[0], fontsize = 15, grid = False, widths=0.3, flierprops = flierprops)
        ax.axhline(y=0.5, color="0.5", linestyle='--', alpha=0.5)
        ax.set_xlabel('Non-Trustworthy (0) and Trustworthy(1)', fontsize = 12)
        ax.set_ylabel('Predicted Trustworthiness Scores', fontsize = 12)
        ax.set_title('')
        fig = np.asarray(ax).reshape(-1)[0].get_figure()
        fig.suptitle('', fontsize = 1)
        ax.text(0.5, -0.3, "("+string.ascii_lowercase[0]+")", transform=ax.transAxes, size=20)
                
        ax_C = (1 - df_car[[columns_car[0], columns_car[3]]]).boxplot(by = columns_car[0], ax=axes.flatten()[1], fontsize = 15, grid = False, widths=0.3, flierprops = flierprops)
        ax_C.axhline(y=0.5, color="0.5", linestyle='--', alpha=0.5)
        ax_C.set_xlabel('Non-Trustworthy (0) and Trustworthy(1)', fontsize = 12)
        ax_C.set_ylabel('Predicted Trustworthiness Scores', fontsize = 12)
        ax_C.set_title('')
        fig = np.asarray(ax_C).reshape(-1)[0].get_figure()
        fig.suptitle('', fontsize = 1)
        ax_C.text(0.5, -0.3, "("+string.ascii_lowercase[1]+")", transform=ax_C.transAxes, size=20)
        
        ax_L = (1 - df_car[[columns_car[1], columns_car[4]]]).boxplot(by = columns_car[1], ax=axes.flatten()[2], fontsize = 15, grid = False, widths=0.3, flierprops = flierprops)
        ax_L.axhline(y=0.5, color="0.5", linestyle='--', alpha=0.5)
        ax_L.set_xlabel('Non-Trustworthy (0) and Trustworthy(1)', fontsize = 12)
        ax_L.set_ylabel('Predicted Trustworthiness Scores', fontsize = 12)
        ax_L.set_title('')
        fig = np.asarray(ax_L).reshape(-1)[0].get_figure()
        fig.suptitle('', fontsize = 1)
        ax_L.text(0.5, -0.3, "("+string.ascii_lowercase[2]+")", transform=ax_L.transAxes, size=20)
        
        ax_R = (1 - df_car[[columns_car[2], columns_car[5]]]).boxplot(by = columns_car[2], ax=axes.flatten()[3], fontsize = 15, grid = False, widths=0.3, flierprops = flierprops)
        ax_R.axhline(y=0.5, color="0.5", linestyle='--', alpha=0.5)
        ax_R.set_xlabel('Non-Trustworthy (0) and Trustworthy(1)', fontsize = 12)
        ax_R.set_ylabel('Predicted Trustworthiness Scores', fontsize = 12)
        ax_R.set_title('')
        fig = np.asarray(ax_R).reshape(-1)[0].get_figure()
        fig.suptitle('', fontsize = 1)
        ax_R.text(0.5, -0.3, "("+string.ascii_lowercase[3]+")", transform=ax_R.transAxes, size=20)
        
        filename = 'Box_Plot_All.png'
        plt.savefig(filename, format="png", bbox_inches='tight', figsize=(9, 11))
        
        plt.show()
        
        
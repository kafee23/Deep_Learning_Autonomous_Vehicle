# -*- coding: utf-8 -*-
"""

"""
# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import Deep_Models as DM
import Model_Evaluation as ME


filename = "Data_Combined_Manually.xlsx"
filename_car = "Data_Set_Car.xlsx"
metrics_file = "Metrics_File.txt"
filename_car_processed = "Data_Set_Car_Processed.xlsx"
filename_MOC_car = "Predictions_Multi_Class_Car.xlsx"
metrics_file_car = "Metrics_File_Car.txt"
filename_analysis = "Predictions_Multi_Class_Car_PR_ROC.xlsx"
filename_analysis_DNN = "Predictions_PR_ROC.xlsx"
filename_analysis_Keras = "Predictions_PR_ROC_Keras.xlsx"
filename_BC = "Predictions_Binary_Class.xlsx"

features = ['Flowrate', 'Speed', 'Phasetime']
data_car_columns = ['Vehicle no', 'Time 1', 'Time 2 (T1+30 sec)', 'Speed 1', 'Speed 2', 'Distance 1', 'Distance 2', 'Camera', 'Lidar', 'Radar']
target_car = ['C', 'L', 'R']
target =   ['Two_Class']
sheet_name = ['Trustworthy_Data', 'Untrustworthy_Data', 'Untrustworthy_MO']
hidden_units_spec = [1024, 512, 256]
dropout_spec = [0.5, 0.4, 0.3]


def read_data(filename, features, target, sheet_name):
    df = pd.read_excel(filename, sheet_name = sheet_name)
    df = df[features + target]
    
    return df

def prepare_data(df1, df2, Shuffle = True):
    df = df1.append(df2).reset_index(drop = True)
    if Shuffle == True:
        df = df.reindex(np.random.permutation(df.index))  # need to shuffle the data to get a mix of both classes    
    
    return df

def read_data_car(filename_car, data_car_columns, sheet_name = 'data_car'):    
    # Reading the data related to individual cars' speed and distance.
    df = pd.read_excel(filename_car, sheet_name = sheet_name)
    df = df[data_car_columns]
    
    columns = ['A1', 'D1', 'SD1', 'A2', 'D2', 'SD2', 'A3', 'RS3', 'D3', 'SD3', 'A4', 'RS4', 'D4', 'SD4',
               'A5', 'RS5', 'D5', 'SD5', 'A6', 'RS6', 'D6', 'SD6', 'A7', 'RS7', 'D7', 'SD7', 'A8', 'RS8', 'D8', 'SD8',
               'C', 'L', 'R']
    data = pd.DataFrame([], columns = columns)
    
    
    frame = []
    for x in list(df.index):
        if df.iloc[x, :]['Vehicle no'] == 9:
            data = data.append(pd.DataFrame([frame], columns = columns), ignore_index = True)
            frame = []
            continue
        frame.append((df.iloc[x, :]['Speed 1'] - df.iloc[x, :]['Speed 2'])/0.5)
        
        if df.iloc[x, :]['Vehicle no'] != 1 and df.iloc[x, :]['Vehicle no'] != 2:
            rs1 = (df.iloc[x, :]['Speed 1'] - df.iloc[x-2, :]['Speed 1'])
            rs2 = (df.iloc[x, :]['Speed 2'] - df.iloc[x-2, :]['Speed 2'])
            frame.append((rs1 + rs2)/2)
        
        frame.append((df.iloc[x, :]['Distance 1'] - df.iloc[x, :]['Distance 2']))
        frame.append((df.iloc[x, :]['Speed 1'] - df.iloc[x, :]['Speed 2']) * (df.iloc[x, :]['Distance 1'] - df.iloc[x, :]['Distance 2']))
        
        if df.iloc[x, :]['Vehicle no'] == 8:
            frame.extend(list(df.iloc[x, :][['Camera', 'Lidar', 'Radar']]))
        
        
    data['C'] = data['C'].astype(int)
    data['L'] = data['L'].astype(int)
    data['R'] = data['R'].astype(int)
    
    data['CLR'] = (4*data['C'] + 2*data['L'] + data['R']) - 1
    
    data.to_excel(filename_car_processed, index = False)
    return df, data


def keras_MOClassifier_car(train_data, test_data, target_columns, feature_columns):

    x_train = train_data[feature_columns]
    y_train = train_data[target_columns]

    input_dict = {
            'hidden_layers' : hidden_units_spec,
            'input_dm' : len(feature_columns),
            'activation_fn': ['relu', 'relu', 'relu', 'sigmoid'],  # should be sigmoid at the end according to https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
            'k_fold': 0,
            'loss' : 'binary_crossentropy',  # treat each output as a binary classifier - thus ensure to trigger the output indipendantly from others
            'optimizer' : 'adam',
            'model_type':'classifier',
            'no_of_output' : len(target_columns),
            'metrics' : ['accuracy'],
            'dropout_spec': dropout_spec,
            'sample_weight' : None
         } 
    Deep_Model = DM.Deep_Models("Deep Neural Network - Multi-output Classifier")
    print(Deep_Model.get_name())
    MOClassifier = Deep_Model.DNN_Models(x_train, y_train, **input_dict)
    
    x_test = test_data[feature_columns]
    y_test = test_data[target_columns].reset_index(drop = True)

        
    if len(list(y_test.columns)) > 1:
        predicted_probability = MOClassifier.predict_proba(x_test)
    else: # for single class classifier, we need one output column
        predicted_probability = MOClassifier.predict(x_test)
    
    
    df_pred = pd.DataFrame([], columns = list(x + '_Pred' for x in list(y_test.columns)))
   
 #   pdb.set_trace()
    for pred in predicted_probability:
        df_row = pd.DataFrame([pred], columns = df_pred.columns)
        df_pred = pd.concat([df_pred, df_row], ignore_index = True, axis = 0)  

    df_class = df_pred.gt(0.5).astype(int)
    df_class.columns = list(x + '_Pred_Class' for x in list(y_test.columns))
        
    df = pd.concat([y_test, df_pred, df_class], axis = 1)
    
    df.to_excel(filename_MOC_car, index = False)
    
    return MOClassifier, df

def keras_BClassifier(train_data, test_data, target):
    x_train = train_data[features]
    y_train = train_data[target]
    
    input_dict = {
                'hidden_layers' : hidden_units_spec,
                'input_dm' : 3,
                'activation_fn': ['relu', 'relu', 'relu', 'sigmoid'],  # should be sigmoid at the end according to https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
                'k_fold': 0,
                'loss' : 'binary_crossentropy',  # treat each output as a binary classifier - thus ensure to trigger the output indipendantly from others
                'optimizer' : 'adam',
                'model_type':'classifier',
                'no_of_output' : 1,
                'metrics' : ['accuracy'],
                'dropout_spec': dropout_spec,
                'class_weight': {0: 50.,
                                 1: 1.},
                'sample_weight' : None
             } 
    
    Deep_Model = DM.Deep_Models("Deep Neural Network - Binary Classifier")
    print(Deep_Model.get_name())
    BClassifier = Deep_Model.DNN_Models_Class_Weight(x_train, y_train, **input_dict)

    x_test = test_data[features]
    y_test = test_data[target].reset_index(drop = True)
    
    predicted_class = BClassifier.predict(x_test)
    predicted_probability = BClassifier.predict_proba(x_test)

    df = pd.concat([y_test,  pd.DataFrame(predicted_probability[:,1], columns = ['Prediction']), pd.DataFrame(predicted_class, columns = ['Predicted_Class'])], axis = 1)
    df.to_excel(filename_BC, index = False)

    return BClassifier, df

def Keras_classifier_all_data():
    #Keep
    df_tr = read_data(filename, features, target, sheet_name[0])    # Read the trustworthy data
    df_utr = read_data(filename, features, target, sheet_name[1])    # Read the untrustworthy data
    df_combined = prepare_data(df_tr, df_utr, True)                 # combining the two data set
    
    # divide the dataset in to train, validate and test - training, validating and predicting.
    
    train_df, test_df = np.split(df_combined.sample(frac=1), [int(.8*len(df_combined))])    
    
    classifier, df = keras_BClassifier(train_df, test_df, target)
    
    df.columns = ['Target', 'Prediction', 'Class']

    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier_Car")

    Model_Eval.metrics_printer(df, len(target), 2)
    Model_Eval.ROC_Curve_Generator(df, len(target), 1) # Ofset to the real probability not the class level, see the df and the generated excel file

    Model_Eval.metrics_file_writer(metrics_file_car, df, len(target), 2)
    
    Model_Eval.PR_Curve_Generator(df, len(target), 1, df_combined, list(df_combined.columns).index('Two_Class')) # adjust to 3 when phasetime will be added as feature set
    
        
    return classifier, test_df, df


def Keras_multiclass_cassifier_car():
    data_excel, data = read_data_car(filename_car, data_car_columns, sheet_name = 'data_car')

    train_data, test_data = np.split(data.sample(frac=1), [int(0.7*len(data))])
    
    feature_columns = [x for x in list(data.columns) if x not in target_car]
    
    MOClassifier, df = keras_MOClassifier_car(train_data, test_data, target_car, feature_columns)
    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier_Car")
    
    Model_Eval.metrics_printer(df, len(target_car), 6)
    Model_Eval.ROC_Curve_Generator(df, len(target_car), 3) # Ofset to the real probability not the class level, see the df and the generated excel file

    Model_Eval.metrics_file_writer(metrics_file_car, df, len(target_car), 6)
    Model_Eval.PR_Curve_Generator(df, len(target_car), 3, data, list(data.columns).index('C')) # adjust to 3 when phasetime will be added as feature set
    
    return MOClassifier, data_excel, data, test_data, df

def final_result_analysis():
    print('Processing Final Results')
    data_excel, data = read_data_car(filename_car, data_car_columns, sheet_name = 'data_car')
    df_final = pd.read_excel(filename_analysis)
    columns = ['C', 'L', 'R', 'C_Pred', 'L_Pred', 'R_Pred', 'C_Pred_Class', 'L_Pred_Class', 'R_Pred_Class']        
    
    df_final = df_final[columns]    
    Model_Eval = ME.Model_Evaluation("Keras_Multiclass_Classifier_Car")
    
    Model_Eval.metrics_printer(df_final, len(target_car), 6)
    Model_Eval.ROC_Curve_Generator_Subplot(df_final, len(target_car), 3) # Ofset to the real probability not the class level, see the df and the generated excel file

    Model_Eval.metrics_file_writer(metrics_file_car, df_final, len(target_car), 6)
    
    Model_Eval.PR_Curve_Generator_Subplot(df_final, len(target_car), 3, data, list(data.columns).index('C')) # adjust to 3 when phasetime will be added as feature set
    
def final_result_analysis_DNN_Keras(Model):
    print('Processing Final Results DNN all data - trust, non-trust classification')
    
    if Model == 'Keras':
        filename_two_class = filename_analysis_Keras
    elif Model == 'DNN':
        filename_two_class = filename_analysis_DNN
    else:
        print('Wrong Method Name')
        exit()
    
    df_tr = read_data(filename, features, target, sheet_name[0])    # Read the trustworthy data
    df_utr = read_data(filename, features, target, sheet_name[1])    # Read the untrustworthy data
    df_combined = prepare_data(df_tr, df_utr, True)   
    
    df_final = pd.read_excel(filename_two_class)
    columns = ['Target', 'Prediction', 'Class']        
    
    df_final = df_final[columns]
    
    Model_Eval = ME.Model_Evaluation("Evaluating DNN classifier for trus vs non-trust")
    Model_Eval.metrics_printer(df_final, len(target), 2)
    Model_Eval.ROC_Curve_Generator(df_final, len(target), 1) # Ofset to the real probability not the class level, see the df and the generated excel file

    Model_Eval.metrics_file_writer(metrics_file_car, df_final, len(target), 2)
    Model_Eval.PR_Curve_Generator(df_final, len(target), 1, df_combined, list(df_combined.columns).index('Two_Class')) # adjust to 3 when phasetime will be added as feature set
    Model_Eval.figure_generator_single_output(df_final, len(target), 1, df_combined, list(df_combined.columns).index('Two_Class'))

def box_plot_analysis():
    df_final = pd.read_excel(filename_analysis_Keras)
    
    df_final_car = pd.read_excel(filename_analysis)
    columns = ['C', 'L', 'R', 'C_Pred', 'L_Pred', 'R_Pred', 'C_Pred_Class', 'L_Pred_Class', 'R_Pred_Class']        
    
    df_final_car = df_final_car[columns]
    
    Model_Eval = ME.Model_Evaluation("Evaluating DNN classifier for trus vs non-trust")
   
    param = ["Target", "Prediction"]
    
    Model_Eval.box_plot_generator(df_final, param, df_final_car, columns)    
    
if __name__ == '__main__':
    
    MOClassifier, data_excel, data_processed, test_df_MOC, df = Keras_multiclass_cassifier_car()
    
    # Keras_classifier_all_data()

    
    # final_result_analysis()
 
    # final_result_analysis_DNN_Keras('Keras')

    # box_plot_analysis()

    
    
    


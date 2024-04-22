import time

import ML_v3_Tree as ML_Tree
import ML_v3_LR as ML_LR
import my_function.useful_fun as myfunction
import pandas as pd
import pickle
import numpy as np
from param_Select_main_func import LR_Param_Select

def my_rounding(Recall, Speciality, AUC, G_mean, Accuracy, Precision, F1_score, MCC):
    Recall=int(Recall*1000+0.5)*0.001
    Speciality=int(Speciality*1000+0.5)*0.001
    AUC=int(AUC*1000+0.5)*0.001
    G_mean=int(G_mean*1000+0.5)*0.001
    Accuracy=int(Accuracy*1000+0.5)*0.001
    Precision=int(Precision*1000+0.5)*0.001
    F1_score=int(F1_score*1000+0.5)*0.001
    MCC=int(MCC*1000+0.5)*0.001
    return Recall, Speciality, AUC, G_mean, Accuracy, Precision, F1_score, MCC


if __name__ == "__main__":
    """"""
    # LR_Param_Select()
    localtime = time.asctime(time.localtime(time.time()))
    print("localtime: ", localtime)


    xlsx_path =  "data/csv_xslx/ICD3_group_disease_names.xlsx"
    xlsxpath=  "data/csv_xslx/AA_MULTI_DESC_TENSOR.xlsx"

    dic_flag_ftr_subset={48: 'BL', 49: 'Baseline+CI', 50: 'BL+疾病组', 51: 'BL+ CI+疾病组', 52: 'BL+network', 53: 'BL+network+CI', 54: 'BL+network+疾病组',
                         55: 'BL+network+CI+疾病组', 56: 'Baseline+history', 57: 'BL+CI+history', 58: 'Baseline+疾病组+history', 59: 'BL+CI+疾病组',
                         60: 'BL+history+network', 61: 'BL+CI+history+network', 62: 'BL+疾病组+history+network', 63: 'BL+CI+疾病组+history+network',
                         64:'BL+cmbdty_attri*3',65:'BL+ntw*3+cmbdty_atti*3', 66: 'BL+CI+疾病组+history+ntw*3+cmbdty_atti*3', 67: 'BL+CI+history+network*6',
                         68: 'BL+CI+history+network*3+weighted_similarity_history_cost',69:"基础特征+网络特征*3+weighted_similarity_history_cost",
                         70:"基础特征+ECI_onehot+CCI_onehot+网络特征+weighted_similarity_history_cost",71:"Baseline + hcp1",72:"Baseline + hcp1 + shortest_distance",
                         73:"Baseline + hcp1 + shortest_distance + EC",74:"Baseline + hcp1 + shortest_distance + EC + CH",75:"Baseline",
                         76:"Baseline+shortest_distance",77:"Baseline + EC",78:"Baseline + CP ",79:"Baseline + network(HCP,SD,CP)",
                         80:"Baseline + hcp1 + EC + CP",81:"Baseline + EC + shortest_distance + CP",82:"Baseline + hcp1 + CP",83:"Baseline + shortest_distance + CP",
                         84:"Baseline + CI + network(HCP,SD,CP)",85:"Baseline + history + network(HCP,SD,CP)",86:"Baseline + CI + history + network(HCP,SD,CP)",89:"Baseline + CI + history + network(HCP,SD,CP) SMOTENC k=5",
                         90:"Baseline + CI + history + network(HCP,SD,CP) random over-sample",91:"random under-sample",92:"SMOTE",93:"SMOTENN",94:"NCR",95:"none"}
    # [75,49,56,57,79,84,85,86]
    # 处理基础特征  要用LR的话，还要对基础特征进行处理
    results_evlt = []
    print("开始训练network_select_main_fun----------")
    pastt=time.time()
    for quantile_flag in [0.9]:
        df,columns_name = ML_Tree.read_data_feature_extract(xlsx_path,xlsxpath,quantile_flag)   #220疾病组的columns_name
        # # 添加ECI和CCI特征
        df,lst_nm_onehot_cci,lst_nm_apper_time_cci,lst_nm_score_cci,lst_nm_onehot_eci,lst_nm_apper_time_eci,lst_nm_score_eci= myfunction.add_ftr_ECI_CCI(df)
        #将eci_cci特征的名字装入一个list中，传入data_preprcss_and_split（）函数中
        lst_name_of_cci_eci=[lst_nm_onehot_cci,lst_nm_apper_time_cci,lst_nm_score_cci,lst_nm_onehot_eci,lst_nm_apper_time_eci,lst_nm_score_eci]
        # lst_name_of_cci_eci = [[], [], [], [], [], []]

        dir = "data_wo_threshold"
        prevalence_path =dir+ "/med_data_mtx_dic/dic_disease_prevalence_rate.pkl"

        for layer_index in [-1]:
            for netwk_type in ["CC"]:
                gml_path = dir + "/gml_dir/"+netwk_type+"_all_modularity_"+str(layer_index)+"layer.gml"

                # 处理网络特征,划分训练集和测试集  （因为网络特征和OR特征要用训练集数据计算，所以，先划分训练集和测试集，再提取网络特征）
                X_train_raw, X_test_raw, y_train_raw, y_test_raw, numOfModule, EC_onehot_name = ML_Tree.add_network_features(gml_path, df, prevalence_path,quantile_flag)

                X_train_raw, X_test_raw, y_train_raw, y_test_raw = ML_Tree.ftr_preprcss_standard(X_train_raw, X_test_raw,y_train_raw, y_test_raw,EC_onehot_name)
                X_train_raw, X_test_raw, y_train_raw, y_test_raw = ML_Tree.ftr_preprcss_standard_cmbdty_attri(X_train_raw, X_test_raw,y_train_raw, y_test_raw)


                for model in ['LightGBM']:
                    flag_lst = [75,49,56,57,79,84,85,86]
                    for i in flag_lst:
                        X_train_ftr, y_train_folds, X_test_ftr, y_test_folds = ML_Tree.data_preprcss_and_split(i, X_train_raw,
                                                                                                       X_test_raw, y_train_raw,
                                                                                                       y_test_raw, numOfModule,
                                                                                                       columns_name,
                                                                                                       lst_name_of_cci_eci)
                        if quantile_flag==0.95:
                            pos_weight_lst=[19]
                        elif quantile_flag==0.9:
                            pos_weight_lst=[9]
                        elif quantile_flag==0.8:
                            pos_weight_lst=[4]
                        for pos_weight in pos_weight_lst:

                            if model == 'XGBoost':
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob = ML_Tree.xgb_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds,pos_weight)
                            elif model == 'LightGBM':
                                if i in [89,90,91,92,93,94,95]:
                                    class_w=None
                                else:
                                    class_w='balanced'
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob = ML_Tree.lightGBM_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds,class_w)
                            elif model=='LR':
                                if i in [89,90,91,92,93,94,95]:
                                    class_w=None
                                else:
                                    class_w='balanced'
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob=ML_Tree.LR_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds,class_w)
                            elif model=='RF':
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname, y_pred_prob = ML_Tree.RF_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds)
                            elif model == 'DT':
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname, y_pred_prob = ML_Tree.DT_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds)
                            elif model == 'SVM':
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob = ML_Tree.SVM_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds)
                            elif model == 'KNN':
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob = ML_Tree.KNN_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds)
                            elif model == 'ANN':
                                x_train_df=pd.concat([X_train_ftr,y_train_folds],axis=1)
                                X_train_ftr, y_train_folds=myfunction.my_random_undersample(x_train_df)  # 随机下采样
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob = ML_Tree.ANN_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds)
                            elif model == 'ANN_unbalance':
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob = ML_Tree.ANN_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds)
                            elif model == 'balanceRF':
                                y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname, y_pred_prob = ML_Tree.balanceRF_prdct(
                                    X_train_ftr, y_train_folds, X_test_ftr, y_test_folds)

                            # 混淆矩阵,各评估指标
                            print("花费阈值是：",quantile_flag)
                            print("pos_weight的值为:",pos_weight)
                            print("-----------读取的是文件  " + gml_path + "   ------------------")
                            print("featrue_set_flag=", i)
                            print("测试集上的表现")
                            Recall, Speciality, AUC, G_mean, Accuracy, Precision, F1_score, MCC = ML_Tree.evaluation_index(y_test_folds,y_pred_rf,y_pred_prob)

                            #将结果指标写到csv中
                            Recall, Speciality, AUC, G_mean, Accuracy, Precision, F1_score, MCC = my_rounding(Recall, Speciality, AUC, G_mean, Accuracy, Precision, F1_score, MCC)
                            results_evlt.append([i,model,quantile_flag,dic_flag_ftr_subset[i],Recall, Speciality, AUC, G_mean, Accuracy, Precision, F1_score, MCC,str(pos_weight)])

                            if model=='XGBoost' or model=='LightGBM' or model=='RF' or model=='DT' or model=='balanceRF':
                                # 特征重要性
                                ftr_imptnt = ML_Tree.evaluation_feature_importantce(feature_importance, X_columnsname)
                                #特征重要性柱状图
                                # ML_Tree.importance_figure(feature_importance, X_columnsname, 40,columns_name,lst_nm_onehot_eci,lst_nm_onehot_cci,EC_onehot_name)

    results_evlt_df=pd.DataFrame(results_evlt,columns=['flag','model','quantile', 'ftr_subset', 'Sensitivity', 'Specificity', 'AUC', 'G-mean', 'Accuracy', 'precision', 'F1-score', 'MCC','pos_weight'])
    results_evlt_df.to_csv(dir+"/results/"+"_result01.csv")
    print("训练和预测耗时：", (time.time() - pastt) / 60)






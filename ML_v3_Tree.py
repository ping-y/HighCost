import cx_Oracle
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder, MinMaxScaler, StandardScaler
import my_function.useful_fun as myfunction
import pickle
import networkx as nx
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split, cross_val_predict,GridSearchCV
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer,confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
import xgboost as xgb
# from sklearn.model_selection
# from imblearn.under_sampling import NeighbourhoodCleaningRule
from scipy import stats
import my_function.myplot as myplt
import lightgbm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTENC,RandomOverSampler,SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler,NeighbourhoodCleaningRule
from imblearn.ensemble import BalancedRandomForestClassifier


def read_data_feature_extract(xlsx_path,xlsxpath,quantile_flag):
    """从数据库读入基础特征，处理基础特征
    quantile_flag=0.95 : 95分位值
    quantile_flag=0.90 : 90分位值
    quantile_flag=0.85 : 85分位值"""

    f2 = open("data/feature_related/df_Histry9_clearned4.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    df=df.reset_index(drop=True)
    print("入院确诊有IHD的住院记录数：", df.shape[0])

    print("--------------处理基础特征中-----------------")
    # 处理基本特征
    # 处理一下性别特征
    sex = pd.get_dummies(df['XB'], drop_first=True)
    sex.columns = [('XB_' + str(i)) for i in range(sex.shape[1])]
    df = pd.concat([df, sex], axis=1)
    df.drop(columns='XB', axis=1, inplace=True)  # 性别特征的取值：0，1

    # 处理标签（二分类）
    # 住院费用分类
    # 先按照95%以上为high cost
    zyfy = df['ZFY']
    # quantile_value = zyfy.quantile(0.95)
    quantile_value= zyfy.quantile(quantile_flag)

    print(quantile_value)  # 40747.12
    df['zfy_label'] = df['ZFY'].apply(lambda x: 1 if x >= quantile_value else 0)  # 高于95为1，低于95为0
    # print(df.head(10))
    # df.drop(columns='ZFY', axis=1, inplace=True)

    # 处理时间   # 后面处理成onehot?
    # print(df['RY_DATE'])
    df['RY_DATE_'] = df['RY_DATE'].astype(str)
    df['RY_Month'] = df['RY_DATE_'].apply(lambda x: x[5:7]).astype(int)
    df['RY_day'] = df['RY_DATE_'].apply(lambda x: x[8:10]).astype(int)
    # df.drop(columns='RY_DATE', axis=1, inplace=True)
    # print(df['RY_Month'])
    # print(df['RY_day'])

    # 处理入院情况 （存在空值）空值用0替换   ! 暂不考虑，空值太多
    # print(df['RYQK'] )

    # 处理行政区域编号
    lbl = LabelEncoder()
    region_ndarry = lbl.fit_transform(df['XZZ_XZQH2'])
    df['XZZ_XZQH2'] = pd.DataFrame(region_ndarry)
    # df.drop(columns='XZZ_XZQH2', axis=1, inplace=True)

    # 处理入院途径  onehot
    lb_rytj = LabelBinarizer()
    rytj_ndarry = lb_rytj.fit_transform(df['RYTJ'])
    df= pd.concat([df, pd.DataFrame(rytj_ndarry,columns=['RYTJ'+str(i) for i in range(rytj_ndarry.shape[1])])], axis=1)
    # df.drop(columns='RYTJ', axis=1, inplace=True)

    # 年龄'NL', 'YYDJ_J', 'YYDJ_D'归一化 fit训练集
    yydjj_ndarry = lbl.fit_transform(df['YYDJ_J'])
    df['YYDJ_J'] = pd.DataFrame(yydjj_ndarry)
    # df.drop(columns='YYDJ_J', axis=1, inplace=True)
    print('YYDJ_J')
    print(list(lbl.classes_))
    yydjd_ndarry = lbl.fit_transform(df['YYDJ_D'])
    df['YYDJ_D'] = pd.DataFrame(yydjd_ndarry)
    print('YYDJ_D')
    print(list(lbl.classes_))
    # df.drop(columns='YYDJ_D', axis=1, inplace=True)

    # 添加特征——疾病所属的ICD10章
    # df = myfunction.chapter_dic(xlsx_path, df)
    # print("添加ICD编码后特征的维数：",df.shape[1])

    # 添加疾病组特征：
    df,columns_name = myfunction.disease_group220_dic(xlsxpath,df)
    print("添加疾病组后特征的维数：",df.shape[1])
    columns_name=[]

    # df[['NL']].hist(bins=20, figsize=(20, 15))
    # plt.show()

    return df,columns_name


def network_feature(gml_path, df, dic_omiga, dic_hcp, dic_hcp_only, dic_OR, dic_hcp1):
    """处理网络特征
         _df :特征df"""
    print("---------------开始计算网络特征：-----------------")
    pastt = time.time()
    # ### 计算社区内特征向量中心性；全网介数中心性，接近中心性，强度中心性指标
    #  对于急性病和流行率小于1%的慢性病，认为他们的特征向量中心度为0
    dic_EC, numOfModule, dic_module = myfunction.Compute_EC(gml_path)  # 得到各个网络中的节点的特征向量中心度;社区个数，疾病所属社区字典
    print("numOfModule", numOfModule)

    betweenness_centrality, closeness_centrality, degree_centrality, dic_strength_centrality = myfunction.Compute_Centrality(
        gml_path)
    # 以上四个中心度变量为字典
    # print("len(betweenness_centrality)", len(betweenness_centrality))

    EC_list = []
    BC_list = []
    CC_list = []
    SC_list = []
    omiga_list = []
    omiga_max_list = []
    hcp_list = []
    hcp_sum_list = []
    hcp1_max_list=[]
    # hcp_only_sum_list = []
    # hcp_only_max_list=[]
    OR_list = []
    OR_max_list = []

    one_hot_list = []
    EC_one_hot_lists = []
    for index, i in enumerate(tqdm(df["ry_diseases_"])):
        EC_value = 0
        BC_value = 0
        CC_value = 0
        SC_value = 0
        omiga_value = 0
        omiga_max = 0
        hcp_value = 0
        hcp_sum_value = 0
        hcp1_max=0
        hcp_only_value = 0
        hcp_only_sum_value = 0
        OR_value = 0
        OR_max = 0

        zero_list = [0 for mi in range(numOfModule)]

        EC_onehot_list = [0 for mindex in range(numOfModule)]

        for j in i:
            if j in dic_EC:
                EC_value += dic_EC[j]

                EC_onehot_list[dic_module[j]] += dic_EC[j]

            if j in betweenness_centrality:
                BC_value += betweenness_centrality[j]
            if j in closeness_centrality:
                CC_value += closeness_centrality[j]
            if j in dic_strength_centrality:
                SC_value += dic_strength_centrality[j]
            if j in dic_omiga:
                omiga_value += dic_omiga[j]
                if dic_omiga[j] > omiga_max:
                    omiga_max = dic_omiga[j]
            if j in dic_hcp:
                hcp_sum_value += dic_hcp[j]
                if dic_hcp[j] > hcp_value:
                    hcp_value = dic_hcp[j]
            if j in dic_hcp1:
                # hcp_sum_value += dic_hcp[j]
                if dic_hcp1[j] > hcp1_max:
                    hcp1_max = dic_hcp1[j]
            # if j in dic_hcp_only:
            #     hcp_only_sum_value+=dic_hcp_only[j]
            #     if dic_hcp_only[j] > hcp_only_value:
            #         hcp_only_value = dic_hcp_only[j]
            if j in dic_module:
                zero_list[dic_module[j]] = 1
            if j in dic_OR:
                OR_value += dic_OR[j][0]
                if dic_OR[j][0] > OR_max:
                    OR_max = dic_OR[j][0]  # dic_OR[i] = [OR, ci_low, ci_high]

        one_hot_list.append(zero_list)

        EC_list.append(EC_value)
        BC_list.append(BC_value)
        CC_list.append(CC_value)
        SC_list.append(SC_value)
        omiga_list.append(omiga_value)
        omiga_max_list.append(omiga_max)
        hcp_list.append(hcp_value)
        hcp_sum_list.append(hcp_sum_value)
        hcp1_max_list.append(hcp1_max)
        # hcp_only_max_list.append(hcp_only_value)
        # hcp_only_sum_list.append(hcp_only_sum_value)
        OR_list.append(OR_value)
        OR_max_list.append(OR_max)
        EC_one_hot_lists.append(EC_onehot_list)

    EC_onehot_name = ['EC_Community_' + str(i) for i in range(numOfModule)]

    df_cpt_onehot = pd.DataFrame(one_hot_list, columns=['Louvain_Community_' + str(i) for i in range(numOfModule)])
    df_EC_onehot = pd.DataFrame(EC_one_hot_lists, columns=['EC_Community_' + str(i) for i in range(numOfModule)])

    df_EC = pd.DataFrame(EC_list, columns=['Eigenvector_centrality_sum'])
    df_BC = pd.DataFrame(BC_list, columns=['betweenness_centrality_sum'])
    df_CC = pd.DataFrame(CC_list, columns=['closeness_centrality_sum'])
    df_SC = pd.DataFrame(SC_list, columns=['strength_centrality_sum'])
    df_omiga = pd.DataFrame(omiga_list, columns=['omiga_sum'])
    df_omiga_max = pd.DataFrame(omiga_max_list, columns=['omiga_max'])
    df_hcp = pd.DataFrame(hcp_list, columns=['hcp_max'])
    df_hcp_sum = pd.DataFrame(hcp_sum_list, columns=['hcp_sum'])
    df_hcp1_max = pd.DataFrame(hcp1_max_list, columns=['hcp1_max'])
    # df_hcp_only = pd.DataFrame(hcp_only_max_list, columns=['hcp_only_max'])
    # df_hcp_sum_only = pd.DataFrame(hcp_only_sum_list, columns=['hcp_only_sum'])
    df_OR = pd.DataFrame(OR_list, columns=['OR_sum'])
    df_OR_max = pd.DataFrame(OR_max_list, columns=['OR_max'])

    df = pd.concat(
        [df, df_EC, df_BC, df_CC, df_SC, df_omiga, df_hcp, df_hcp_sum, df_OR, df_OR_max, df_cpt_onehot, df_EC_onehot,df_omiga_max,df_hcp1_max],
        axis=1)
    del one_hot_list
    print("----------------计算网络特征耗时：---", (time.time() - pastt) / 60)
    return df, numOfModule, EC_onehot_name


def compute_omiga_cmbdty(gml_path,df,dic_omiga):
    """ 添加网络特征：omiga_multi_cmbdty """
    print("计算共病网络特征omiga_multi_cmbdty中-----------")
    pastt=time.time()
    mygraph = nx.read_gml(gml_path)
    node_list = sorted(list(mygraph.nodes))

    df_related=df[node_list]
    for node_name in node_list:
        if  node_name in dic_omiga:
            df_related[node_name]=df_related[node_name].apply(lambda x:x*dic_omiga[node_name])
        else:
            print("网络节点不在dic_omiga中################")
            df_related[node_name] = df_related[node_name].apply(lambda x: 0)

    df_related['omiga_multi_cmbdty'] = df_related.apply(lambda x: x.sum(), axis=1)

    df=pd.concat([df,df_related['omiga_multi_cmbdty'].to_frame()],axis=1)
    print("计算共病网络特征omiga_multi_cmbdty耗时：",(time.time()-pastt)/60)
    return df



def add_network_features(gml_path, df,prevalence_path,quantile_flag):
    """
    划分训练集和测试集，处理网络特征
    """
    # 划分训练集和测试集 ：分层随机划分原则，4：1
    # X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(df, df['zfy_label'], test_size=0.2,
    #                                                                     random_state=42, shuffle=True,
    #                                                                     stratify=df['zfy_label'])

    #按时序划分
    X_train_raw, X_test_raw, y_train_raw, y_test_raw=myfunction.time_series_split(df)

    print("训练集数量：",X_train_raw.shape[0])
    print("测试集数量：", X_test_raw.shape[0])
    # 添加网络特征
    df_cydis_label = pd.concat([X_train_raw['cy_diseases_'], X_train_raw['zfy_label']], axis=1)
    # print("df_cydis_label.shape[0]", df_cydis_label.shape[0])  #138882
    dic_omiga, hcp, hcp_only,hcp1  = myfunction.compute_OMIGA(df_cydis_label, gml_path,prevalence_path)  # dic_omiga,hcp :两个字典

    #观察hcp和流行率之间的关系
    # x_coef, intrcpt=myplt.hcp_prevalence("data/med_data_mtx_dic/dic_disease_prevalence_rate.pkl", hcp_only)
    if quantile_flag==0.95:
        gml_path_OR="data/gml_dir/distance_OR_Graph_all_95_time_series.gml"
    elif quantile_flag==0.9:
        gml_path_OR="data/gml_dir/distance_OR_Graph_all_9_time_series.gml"
    elif quantile_flag==0.8:
        gml_path_OR="data/gml_dir/distance_OR_Graph_all_8_time_series.gml"
    dic_shortest_path,max_dist,min_dist=myfunction.find_shortest_path(gml_path_OR)
    # dic_OR_greater_1, dic_OR_less_1=myfunction.compute_OR_comobidity_cost(X_train_raw)

    # dic_tmp=dic_shortest_path.copy()
    # dic_tmp['max']=max_dist
    # pickle.dump(hcp1, open("data_model/dic_hcp1.pkl", "wb"))
    # pickle.dump(dic_tmp, open("data_model/dic_shortest_path.pkl", "wb"))

    dic_OR = myfunction.compute_OR(df_cydis_label)
    df_all_sample = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)
    df_all_sample, numOfModule , EC_onehot_name = network_feature(gml_path, df_all_sample, dic_omiga, hcp, hcp_only, dic_OR,hcp1)

    df_all_sample=myfunction.network_feature_shortest_dist(df_all_sample, dic_shortest_path, max_dist)

    df_all_sample=compute_omiga_cmbdty(gml_path, df_all_sample, dic_omiga)  # 添加网络特征 omiga_multi_cmbdty

    # df_all_sample =myfunction.network_feature_OR_comobidity_cost(df_all_sample, dic_OR_greater_1, dic_OR_less_1)

    len_train = X_train_raw.shape[0]
    X_train_raw = df_all_sample.loc[[i for i in range(len_train)], :]
    X_test_raw = df_all_sample.loc[[i for i in range(len_train, df_all_sample.shape[0])], :].reset_index(drop=True)
    print("X_train_raw.shape[0]", X_train_raw.shape[0])
    print("X_test_raw.shape[0]", X_test_raw.shape[0])
    # print("X_train_raw.columns", X_train_raw.columns)
    # print("X_train_raw.head", X_train_raw.head().values.tolist())

    # X_test_hierarchy = X_test_raw[['XB_0', 'NL', 'ry_diseases_','zfy_label']]
    # X_test_hierarchy = pd.concat([X_test_hierarchy, y_test_raw], axis=1)
    # f0 = open("data/temporary/X_test_hierarchy.pkl", 'wb')   #X_test_for_hierarchy
    # pickle.dump(X_test_hierarchy, f0)
    # f0.close()

    return X_train_raw, X_test_raw, X_train_raw['zfy_label'], X_test_raw['zfy_label'], numOfModule,EC_onehot_name


def ftr_preprcss_standard( X_train_folds, X_test_folds, y_train_folds, y_test_folds,EC_onehot_name):
    # 年龄'NL', 'YYDJ_J', 'YYDJ_D'标准化 fit训练集
    # X_train_folds[['NL']].hist(bins=20,figsize=(20,15))
    # plt.show()

    # boxcox
    # hcp_max_df=X_train_folds[~(X_train_folds['hcp_max']==0)]['hcp_max']
    # xt, lmbda = stats.boxcox(hcp_max_df)
    # print("lmbda=",lmbda)
    # if lmbda==0:
    #     X_train_folds['hcp_max'] = X_train_folds['hcp_max'].apply(lambda x:np.log(x) if x!=0 else 0)
    #     X_test_folds['hcp_max'] = X_test_folds['hcp_max'].apply(lambda x: np.log(x) if x != 0 else 0)
    # elif lmbda>0:
    #     X_train_folds['hcp_max'] = X_train_folds['hcp_max'].apply(lambda x: (x**lmbda-1)/lmbda if x != 0 else 0)
    #     X_test_folds['hcp_max'] = X_test_folds['hcp_max'].apply(lambda x: (x**lmbda-1)/lmbda if x != 0 else 0)
    # else:
    #     print("lmbda小于0#######################################")

    # X_train_folds['hcp_max'] = X_train_folds['hcp_max'].apply(lambda x: np.log(x+1))
    # X_test_folds['hcp_max'] = X_test_folds['hcp_max'].apply(lambda x: np.log(x+1))

    # X_train_folds[['NL', 'betweenness_centrality_sum', 'strength_centrality_sum', 'Eigenvector_centrality_sum',
    #                'closeness_centrality_sum', 'hcp_sum', 'hcp_max', 'OR_sum', 'omiga_sum']].hist(bins=20,
    #                                                                                               figsize=(20, 15))
    # plt.show()

    # 标准化
    #
    # ,'hcp_div_prev_sum','hcp_div_prev_max','hcp_sub_prev_sum','hcp_sub_prev_max'
    # stdard_ftr_name=['XZZ_XZQH2','YYDJ_D','YYDJ_J','weighted_average_cost_past_3y','last_time_cost_per_day','min_zfy_last_3y','std_zfy_last_3y','interval_thistimery_pasttimecy','shortest_distance','OR_max','omiga_max','los_last_time','RY_Month','zfy_last_time', 'mean_zfy_last_3y', 'med_zfy_last_3y',
    #                                         'max_zfy_last_3y', 'zyci_last_3y', 'NL', 'betweenness_centrality_sum',
    #                                         'strength_centrality_sum', 'closeness_centrality_sum', 'hcp_sum', 'hcp_max','hcp1_max', 'OR_sum', 'omiga_sum']
    stdard_ftr_name=['XZZ_XZQH2','YYDJ_D','YYDJ_J','weighted_average_cost_past_3y','last_time_cost_per_day','min_zfy_last_3y','std_zfy_last_3y',
                     'interval_thistimery_pasttimecy','shortest_distance','los_last_time','RY_Month','zfy_last_time', 'med_zfy_last_3y',
                                            'max_zfy_last_3y', 'zyci_last_3y', 'NL', 'hcp1_max']

    # stdard_ftr_name.extend(EC_onehot_name)
    stdrd_scale = StandardScaler().fit(X_train_folds[stdard_ftr_name])
    X_train_folds[stdard_ftr_name] = stdrd_scale.transform(X_train_folds[stdard_ftr_name])
    X_test_folds[stdard_ftr_name] = stdrd_scale.transform(X_test_folds[stdard_ftr_name])
    # X_train_folds[['NL','betweenness_centrality_sum', 'strength_centrality_sum', 'Eigenvector_centrality_sum',
    #                      'closeness_centrality_sum', 'hcp_sum', 'hcp_max','OR_sum','omiga_sum']].hist(bins=20,figsize=(20,15))
    # plt.show()

    f0 = open("data/temporary/stdrd_scale.pkl", 'wb')
    pickle.dump(stdrd_scale, f0)
    f0.close()
    f0 = open("data/temporary/stdard_ftr_name_1.pkl", 'wb')
    pickle.dump(stdard_ftr_name, f0)
    f0.close()

    return X_train_folds, X_test_folds, y_train_folds, y_test_folds


def ftr_preprcss_standard_cmbdty_attri( X_train_folds, X_test_folds, y_train_folds, y_test_folds):
    # 标准化网络特征：cmbdty_attri_sum这一类的网络特征

    f2 = open("data_wo_threshold/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'rb')
    dic_cmbdty_attribution = pickle.load(f2)
    f2.close()
    lst_columns = list(dic_cmbdty_attribution.keys())

    # stdard_ftr_name=["cmbdty_attri_sum","omiga_multi_cmbdty","weighted_similarity_history_cost"]
    # stdard_ftr_name.extend(lst_columns)
    stdard_ftr_name =lst_columns     # 要标准化的列（列名组成的列表）
    stdrd_scale = StandardScaler().fit(X_train_folds[stdard_ftr_name])   #
    X_train_folds[stdard_ftr_name] = stdrd_scale.transform(X_train_folds[stdard_ftr_name])
    X_test_folds[stdard_ftr_name] = stdrd_scale.transform(X_test_folds[stdard_ftr_name])

    f0 = open("data/temporary/stdrd_scale_2.pkl", 'wb')
    pickle.dump(stdrd_scale, f0)
    f0.close()
    # f0 = open("data/temporary/lst_cmbdty_propensity_name.pkl", 'wb')
    # pickle.dump(lst_columns, f0)
    # f0.close()

    return X_train_folds, X_test_folds, y_train_folds, y_test_folds


def data_preprcss_and_split(ftr_type, X_train_folds, X_test_folds, y_train_folds, y_test_folds, numOfModule,columns_name,lst_name_of_cci_eci):
    """
    :param ftr_type:  0：只纳入基础特征；1：纳入基础特征和网络特征,......
    :param X_train_folds:  a df
    :param X_test_folds:   a df
    :param y_train_folds: a df
    :param y_test_folds: a df
    :return:
    """
    # print(np.array(X_train_folds.head(10)))
    # ECI和CCI相关特征的名字列表，下面分特征组时需要用到
    print("y_test_fold:",y_test_folds.shape)
    lst_nm_onehot_cci, lst_nm_apper_time_cci, lst_nm_score_cci, lst_nm_onehot_eci, lst_nm_apper_time_eci, lst_nm_score_eci=lst_name_of_cci_eci

    columns_EC_onehot = ['EC_Community_' + str(i) for i in range(numOfModule)]
    columns_Louvain_onehot = ['Louvain_Community_' + str(i) for i in range(numOfModule)]
    columns_ICDchapter_onehot = ['ICD_chapter_' + str(i) for i in range(22)]

    column_base2 = ['NL', 'XB_0', 'zfy_label', 'RY_Month', 'RYTJ0', 'RYTJ1', 'RYTJ2', 'RYTJ3','YYDJ_J', 'YYDJ_D','XZZ_XZQH2']
    column_base3 = ['NL', 'XB_0', 'zfy_label', 'RY_Month', 'RYTJ', 'YYDJ_J', 'YYDJ_D','XZZ_XZQH2']
    column_chapter_5last = ['ICD_chapter_8', 'ICD_chapter_14', 'ICD_chapter_15', 'ICD_chapter_19', 'ICD_chapter_21']
    column_1_chapter = [i for i in columns_ICDchapter_onehot if
                        i not in ['ICD_chapter_8', 'ICD_chapter_14', 'ICD_chapter_15', 'ICD_chapter_19', 'ICD_chapter_21']]
    history_column=['los_last_time', 'zfy_last_time', 'weighted_average_cost_past_3y', 'med_zfy_last_3y',
                             'max_zfy_last_3y',
                             'zyci_last_3y', 'last_time_cost_per_day', 'min_zfy_last_3y', 'std_zfy_last_3y',
                             'interval_thistimery_pasttimecy']

    f2 = open("data_wo_threshold/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'rb')
    dic_cmbdty_attribution = pickle.load(f2)
    f2.close()
    lst_columns_comobidity = list(dic_cmbdty_attribution.keys())



    if ftr_type == 0:
        # 基础特征(含ICD编码)
        column_base2.extend(column_chapter_5last)
        column_base2.extend(column_1_chapter)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -1:
        # 基础特征（不含ICD编码)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 1:
        # 基础特征+ICD章节onehot+删除了几个最不重要的章节
        column_1 = column_base2
        column_1.extend(column_1_chapter)
        X_train_ftr = X_train_folds[column_1]
        X_test_ftr = X_test_folds[[i for i in column_1 if i not in ['zfy_label']]]
    elif ftr_type == -2:
        # 基础特征+疾病组213不降维
        column_base2.extend(columns_name)
        X_train_ftr = X_train_folds[column_base2]

        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == -3:
        # 基础特征+疾病组降维LDA-40
        X_train_folds, X_test_folds,columns_name2=myfunction.PCA_DR(X_train_folds, y_train_folds, X_test_folds, columns_name, 40)

        column_base2.extend(columns_name2)
        X_train_ftr = X_train_folds[column_base2]

        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -4:
        # 基础特征+疾病组降维LDA-30
        X_train_folds, X_test_folds,columns_name2=myfunction.PCA_DR(X_train_folds, y_train_folds, X_test_folds, columns_name, 30)

        column_base2.extend(columns_name2)
        X_train_ftr = X_train_folds[column_base2]

        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -5:
        # 基础特征+疾病组降维LDA-20
        X_train_folds, X_test_folds,columns_name2=myfunction.PCA_DR(X_train_folds, y_train_folds, X_test_folds, columns_name, 20)

        column_base2.extend(columns_name2)
        X_train_ftr = X_train_folds[column_base2]

        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == -6:
        # 基础特征+疾病组降维LDA-10
        X_train_folds, X_test_folds,columns_name2=myfunction.PCA_DR(X_train_folds, y_train_folds, X_test_folds, columns_name, 10)

        column_base2.extend(columns_name2)
        X_train_ftr = X_train_folds[column_base2]

        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == -7:
        # 基础特征+疾病组降维LDA-5
        X_train_folds, X_test_folds,columns_name2=myfunction.PCA_DR(X_train_folds, y_train_folds, X_test_folds, columns_name, 5)

        column_base2.extend(columns_name2)
        X_train_ftr = X_train_folds[column_base2]

        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -8:
        # 基础特征+疾病组降维LDA-5
        X_train_folds, X_test_folds,columns_name2=myfunction.LDA_DR(X_train_folds, y_train_folds, X_test_folds, columns_name, 1)

        column_base2.extend(columns_name2)
        X_train_ftr = X_train_folds[column_base2]

        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    # -9到-14 ：ECI和CCI相关共病特征组
    elif ftr_type == -9:
        # 基础特征+CCI_onehot
        column_base2.extend(lst_nm_onehot_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -10:
        # 基础特征+CCI_score
        column_base2.extend(lst_nm_score_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -11:
        # 基础特征+CCI_apper_time
        column_base2.extend(lst_nm_apper_time_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -12:
        # 基础特征+ECI_onehot
        column_base2.extend(lst_nm_onehot_eci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -13:
        # 基础特征+ECI_score
        column_base2.extend(lst_nm_score_eci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -14:
        # 基础特征+ECI_apper_time
        column_base2.extend(lst_nm_apper_time_eci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == -15:
        # 基础特征+ECI_onehot+CCI_onehot
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -16:
        # 基础特征+ECI_score+CCI_score
        column_base2.extend(lst_nm_score_eci)
        column_base2.extend(lst_nm_score_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -17:
        # 基础特征+ECI_apper_time+CCI_apper_time
        column_base2.extend(lst_nm_apper_time_eci)
        column_base2.extend(lst_nm_apper_time_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == -18:
        # 基础特征+ECI_apper_time+CCI_apper_time+网络特征
        column_base2.extend(lst_nm_apper_time_eci)
        column_base2.extend(lst_nm_apper_time_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == -19:
        # 基础特征+ECI_onehot+CCI_onehot+网络特征
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == 2:
        # 基础特征+高花费风险特征
        column_2 = ['OR_sum', 'omiga_sum']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]



    elif ftr_type == 3:
        # 基础特征+网络特征MembershipScore(m)
        column_3 = []
        column_3.extend(column_base2)
        column_3.extend(columns_EC_onehot)
        # column_3.extend(columns_Louvain_onehot)
        # column_3.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'Eigenvector_centrality_sum',
        #                  'closeness_centrality_sum', 'hcp_sum', 'hcp_max'])
        X_train_ftr = X_train_folds[column_3]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]

    elif ftr_type == 4:
        # 基础特征+网络特征：中心度指标
        column_3 = []
        column_3.extend(column_base2)
        # column_3.extend(columns_EC_onehot)
        # column_3.extend(columns_Louvain_onehot)
        # column_3.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'Eigenvector_centrality_sum','closeness_centrality_sum'])
        column_3.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'closeness_centrality_sum'])
        X_train_ftr = X_train_folds[column_3]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]

    elif ftr_type == 5:
        # 基础特征+网络特征hcp_sum，hcp_max
        column_3 = []
        column_3.extend(column_base2)
        # column_3.extend(columns_EC_onehot)
        # column_3.extend(columns_Louvain_onehot)
        column_3.extend(['hcp_sum', 'hcp_max'])
        X_train_ftr = X_train_folds[column_3]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]

    elif ftr_type == 6:
        # 基础特征+网络特征columns_Louvain_onehot
        column_3 = []
        column_3.extend(column_base2)
        # column_3.extend(columns_EC_onehot)
        column_3.extend(columns_Louvain_onehot)
        # column_3.extend(['hcp_sum', 'hcp_max'])
        X_train_ftr = X_train_folds[column_3]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]
    elif ftr_type == 7:
        # 基础特征+历史特征
        column_3 = []
        column_3.extend(column_base2)
        # column_3.extend(columns_EC_onehot)
        # column_3.extend(columns_Louvain_onehot)
        column_3.extend(['zfy_last_time', 'mean_zfy_last_3y', 'med_zfy_last_3y', 'max_zfy_last_3y', 'zyci_last_3y'])
        X_train_ftr = X_train_folds[column_3]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]
    elif ftr_type == 8:
        # 基础特征+all network特征
        column_3 = []
        column_3.extend(column_base2)
        column_3.extend(columns_EC_onehot)
        column_3.extend(columns_Louvain_onehot)
        column_3.extend(['hcp_sum', 'hcp_max'])
        column_3.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'closeness_centrality_sum'])

        X_train_ftr = X_train_folds[column_3]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]

    elif ftr_type == 9:
        # 基础特征+all network特征+高花费特征
        column_3 = []
        column_3.extend(column_base2)
        column_3.extend(columns_EC_onehot)
        column_3.extend(columns_Louvain_onehot)
        column_3.extend(['hcp_sum', 'hcp_max'])
        column_3.extend(['OR_sum', 'omiga_sum'])
        column_3.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'closeness_centrality_sum'])

        # column_3.extend(['los_last_time', 'zfy_last_time', 'mean_zfy_last_3y', 'med_zfy_last_3y', 'max_zfy_last_3y',
        #                  'zyci_last_3y'])
        X_train_ftr = X_train_folds[column_3]

        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        # X_train_ftr=myfunction.NCR_imblearn(X_train_ftr)  #ncr
        # X_train_ftr=myfunction.my_up_sample(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]

    elif ftr_type == 10:
        # 基础特征+all network特征+高花费特征+历史特征
        column_3 = []
        column_3.extend(column_base2)
        column_3.extend(columns_EC_onehot)
        column_3.extend(columns_Louvain_onehot)
        column_3.extend(['hcp_sum', 'hcp_max'])
        column_3.extend(['OR_sum', 'omiga_sum'])
        column_3.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'closeness_centrality_sum'])

        column_3.extend(['los_last_time', 'zfy_last_time', 'mean_zfy_last_3y', 'med_zfy_last_3y', 'max_zfy_last_3y',
                         'zyci_last_3y'])
        X_train_ftr = X_train_folds[column_3]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr,"data/ncr/flag_10_ncr.pkl")
        X_test_ftr = X_test_folds[[i for i in column_3 if i not in ['zfy_label']]]

    elif ftr_type == 17:
        # 基础特征+OR特征max
        column_2 = ['OR_max']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 18:
        # 基础特征+OR特征sum
        column_2 = ['OR_sum']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 19:
        # 基础特征+Omiga特征max
        column_2 = ['omiga_max']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 20:
        # 基础特征+Omiga特征sum
        column_2 = ['omiga_sum']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 21:
        # 基础特征+Omiga_max+omiga_sum
        column_2 = ['omiga_max','omiga_sum']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 22:
        # 基础特征+OR特征max+OR_sum
        column_2 = ['OR_sum','OR_max']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 23:
        # 基础特征+OR特征max+OR_sum
        column_2 = ['OR_sum','omiga_max']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 24:
        # 基础特征+OR特征max+OR_sum
        column_2 = ['OR_sum','omiga_max','OR_max','omiga_sum']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]


    elif ftr_type == 25:
        # 基础特征+hcp1_max
        column_2 = ['hcp1_max']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 26:
        # 基础特征+hcp_max
        column_2 = ['hcp_max']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 27:
        # 基础特征 + hcp1_max + EC
        column_2 = ['hcp1_max']
        column_2.extend(columns_EC_onehot)
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]


    elif ftr_type == 28:
        # 基础特征 + fast_unfolding_onehot
        column_2 = []
        column_2.extend(columns_Louvain_onehot)
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 29:
        # 基础特征 + fast_unfolding_onehot+EC+hcp1_max
        column_2 = ['hcp1_max']
        column_2.extend(columns_Louvain_onehot)
        column_2.extend(columns_EC_onehot)
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]

    elif ftr_type == 30:
        # 基础特征 + 中心度
        column_2 = []
        column_2.extend(column_base2)
        column_2.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'closeness_centrality_sum'])
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 31:
        # 基础特征 + 中心度+EC+hcp1_max
        column_2 = ['hcp1_max']
        column_2.extend(columns_EC_onehot)
        column_2.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'closeness_centrality_sum'])
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 32:
        # 基础特征 + 中心度betweenness_centrality_sum+EC+hcp1_max
        column_2 = ['hcp1_max']
        column_2.extend(columns_EC_onehot)
        column_2.extend(['betweenness_centrality_sum'])
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]


    elif ftr_type == 33:
        # 基础特征+shortest_distance
        column_2 = ['shortest_distance']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 34:
        # 基础特征+shortest_distance+EC+HCP!
        column_2 = ['shortest_distance','hcp1_max']
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 35:
        # 基础特征+OR_sum+EC+HCP!
        column_2 = ['OR_sum','hcp1_max']
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 36:
        # 基础特征+OR_sum+EC+HCP!
        column_2 = ['OR_max', 'hcp1_max']
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]

    elif ftr_type == 37:
        # 基础特征+'comobidity_num_OR_G1','comobidity_num_OR_L1'
        column_2 = ['comobidity_num_OR_G1','comobidity_num_OR_L1']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 38:
        # 基础特征+'comobidity_num_OR_G1','comobidity_num_OR_L1','shortest_distance','hcp1_max',EC
        column_2 = ['comobidity_num_OR_G1','comobidity_num_OR_L1','shortest_distance','hcp1_max']
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]

    elif ftr_type == 39:
        # 基础特征+历史特征all
        column_2 = ['los_last_time','zfy_last_time', 'mean_zfy_last_3y', 'med_zfy_last_3y', 'max_zfy_last_3y', 'zyci_last_3y','weighted_average_cost_past_3y','last_time_cost_per_day','min_zfy_last_3y','std_zfy_last_3y','interval_thistimery_pasttimecy']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 40:
        # 基础特征+历史特征,无mweight_average_cost_past_3y
        column_2 =['los_last_time','zfy_last_time', 'mean_zfy_last_3y', 'med_zfy_last_3y', 'max_zfy_last_3y', 'zyci_last_3y','last_time_cost_per_day','min_zfy_last_3y','std_zfy_last_3y','interval_thistimery_pasttimecy']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 41:
        # 基础特征+历史特征,无mean_average_cost_past_3y
        column_2 = ['los_last_time', 'zfy_last_time', 'weighted_average_cost_past_3y_wtDJ_y', 'med_zfy_last_3y', 'max_zfy_last_3y',
                    'zyci_last_3y', 'last_time_cost_per_day', 'min_zfy_last_3y', 'std_zfy_last_3y',
                    'interval_thistimery_pasttimecy']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 42:
        # 基础特征+3个历史特征
        column_2 = ['zyci_last_3y','weighted_average_cost_past_3y','interval_thistimery_pasttimecy']
        column_2.extend(column_base2)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 43:
        # 基础特征+shortest_distance+EC+HCP!+中心度
        column_2 = ['shortest_distance','hcp1_max']
        column_2.extend(['betweenness_centrality_sum', 'strength_centrality_sum', 'closeness_centrality_sum'])
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 44:
        # 基础特征+shortest_distance+EC+HCP!+中心度
        column_2 = ['shortest_distance','hcp1_max']
        column_2.extend(['OR_sum', 'omiga_sum'])
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]

    elif ftr_type == 45:
        # 基础特征+
        column_2 = ['shortest_distance','hcp1_max']
        column_2.extend(lst_nm_onehot_eci)
        column_2.extend(lst_nm_onehot_cci)
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        column_2.extend(columns_name)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]
    elif ftr_type == 46:
        # 基础特征+
        column_2 = ['shortest_distance','hcp1_max','OR_sum']
        column_2.extend(['los_last_time', 'zfy_last_time', 'weighted_average_cost_past_3y', 'med_zfy_last_3y',
                    'max_zfy_last_3y',
                    'zyci_last_3y', 'last_time_cost_per_day', 'min_zfy_last_3y', 'std_zfy_last_3y',
                    'interval_thistimery_pasttimecy'])
        column_2.extend(lst_nm_onehot_eci)
        column_2.extend(lst_nm_onehot_cci)
        column_2.extend(column_base2)
        column_2.extend(columns_EC_onehot)
        column_2.extend(columns_name)
        X_train_ftr = X_train_folds[column_2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule(X_train_ftr)
        X_test_ftr = X_test_folds[[i for i in column_2 if i not in ['zfy_label']]]


    elif ftr_type == 47:
        # 基础特征（不含ICD编码),不含医院级别和医院等级
        column_base_base=['NL', 'XB_0', 'zfy_label', 'RY_Month', 'RYTJ', 'XZZ_XZQH2']
        X_train_ftr = X_train_folds[column_base_base]
        X_test_ftr = X_test_folds[[i for i in column_base_base if i not in ['zfy_label']]]
    elif ftr_type == 48:
        # 基础特征（不含ICD编码)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 49:
        # 基础特征+ECI_onehot+CCI_onehot
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 50:
        # 基础特征+疾病组213不降维
        column_base2.extend(columns_name)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 51:
        # 基础特征+疾病组213不降维+CI
        column_base2.extend(columns_name)
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]


    elif ftr_type == 52:
        # 基础特征+网络特征
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 53:
        # 基础特征+ECI_onehot+CCI_onehot+网络特征
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 54:
        # 基础特征+213疾病组+网络特征
        column_base2.extend(columns_name)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 55:
        # 基础特征+213疾病组+网络特征+CI
        column_base2.extend(columns_name)
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]


    elif ftr_type == 56:
        # 基础特征+history
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 57:
        # 基础特征+ECI_onehot+CCI_onehot+history
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 58:
        # 基础特征+疾病组213不降维+history
        column_base2.extend(columns_name)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 59:
        # 基础特征+疾病组213不降维+CI+history
        column_base2.extend(columns_name)
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]


    elif ftr_type == 60:
        # 基础特征+网络特征+history
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 61:
        # 基础特征+ECI_onehot+CCI_onehot+网络特征+history
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 62:
        # 基础特征+213疾病组+网络特征+history
        column_base2.extend(columns_name)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 63:
        # 基础特征+213疾病组+网络特征+history+CI
        column_base2.extend(columns_name)
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        # X_train_ftr = myfunction.Neighbor_clearn_rule2(X_train_ftr, "data/ncr/flag_63_ncr.pkl",columns_EC_onehot)
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]



    elif ftr_type == 64:
        # 基础特征+网络特征cmbidty_attri*3
        column_base2.extend(["cmbdty_attri_sum", "omiga_multi_cmbdty"])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 65:
        # 基础特征+网络特征cmbidty_attri*3+hcp+shortest_path+EC_onehot
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(["cmbdty_attri_sum", "omiga_multi_cmbdty"])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 66:
        # 基础特征+213疾病组+网络特征+history+CI+网络特征cmbdty_attri*3
        column_base2.extend(columns_name)
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(history_column)
        column_base2.extend(["cmbdty_attri_sum", "omiga_multi_cmbdty"])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 67:
        # 基础特征+网络特征+history+CI+网络特征cmbdty_attri*3
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(history_column)
        column_base2.extend(["cmbdty_attri_sum", "omiga_multi_cmbdty"])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == 68:
        # 基础特征+ECI_onehot+CCI_onehot+网络特征+history+weighted_similarity_history_cost
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(["weighted_similarity_history_cost"])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == 69:
        # 基础特征+网络特征*3+weighted_similarity_history_cost
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(["weighted_similarity_history_cost"])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 70:
        # 基础特征+ECI_onehot+CCI_onehot+网络特征+weighted_similarity_history_cost
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(["weighted_similarity_history_cost"])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    # 71: , 72: , 73: , 74:
    elif ftr_type == 71:
        # "Baseline + hcp1"
        column_base2.extend(['hcp1_max'])
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 72:
        # "Baseline + hcp1 + shortest_distance"
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 73:
        # "Baseline + hcp1 + shortest_distance + EC"
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 74:
        # "Baseline + hcp1 + shortest_distance + EC + CH"
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(columns_EC_onehot)
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    # 75: "Baseline", 76: "Baseline+shortest_distance"
    elif ftr_type == 75:
        # "Baseline"
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 76:
        # "Baseline +shortest_distance"
        column_base2.extend(['shortest_distance'])
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 77:
        # "Baseline + EC "
        column_base2.extend(columns_EC_onehot)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 78:
        # "Baseline + CH "
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 79:
        # "Baseline + hcp1 + shortest_distance + CH"
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 80:
        # "Baseline + hcp1 + EC + CH"
        column_base2.extend(['hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 81:
        # "Baseline + EC + shortest_distance + CH"
        column_base2.extend(['shortest_distance'])
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
        # 82: "Baseline + hcp1 + CH", 83: "Baseline + shortest_distance + CH"
    elif ftr_type == 82:
        # "Baseline + hcp1 + CH"
        column_base2.extend(['hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 83:
        # "Baseline +shortest_distance + CH"
        column_base2.extend(['shortest_distance'])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == 84:
        # "Baseline + CI + network(HCP,SD,CP)"
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 85:
        # "Baseline + history + network(HCP,SD,CP)"
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        column_base2.extend(history_column)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
    elif ftr_type == 86:
        # "Baseline + CI + history + network(HCP,SD,CP)"

        # dic_ftr_groups_name = {"baseline": [i for i in column_base2 if i not in ['zfy_label']]}


        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(history_column)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

        # dic_ftr_groups_name['eci']=lst_nm_onehot_eci
        # dic_ftr_groups_name['cci']=lst_nm_onehot_cci
        # dic_ftr_groups_name['history']=history_column
        # dic_ftr_groups_name['comorbidity']=lst_columns_comobidity
        # dic_ftr_groups_name['two_netw']=['shortest_distance', 'hcp1_max']
        # pickle.dump(dic_ftr_groups_name, open("data_model/dic_ftr_groups_name.pkl", "wb"))


    elif ftr_type == 87 or ftr_type == 88 or ftr_type == 89:
        # "Baseline + CI + history + network(HCP,SD,CP)"  SMOTENC
        if ftr_type==87:
            k=3
        elif ftr_type==88:
            k=4
        elif ftr_type==89:
            k=5

        column_base3.extend(lst_nm_onehot_eci)
        column_base3.extend(lst_nm_onehot_cci)
        column_base3.extend(history_column)
        column_base3.extend(['shortest_distance', 'hcp1_max'])
        column_base3.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base3]

        y_train_folds = X_train_ftr['zfy_label']
        X_train_ftr.drop(columns='zfy_label', axis=1, inplace=True)

        categorical_ftrs = ['RYTJ','XB_0']
        categorical_ftrs.extend(lst_nm_onehot_cci)
        categorical_ftrs.extend(lst_nm_onehot_eci)

        idx_lst = []
        for i in categorical_ftrs:
            idx = list(X_train_ftr.columns).index(i)
            idx_lst.append(idx)
            # print(idx)
        print("smotenc---------------")
        pastt=time.time()
        sm = SMOTENC(random_state=42, categorical_features=idx_lst, k_neighbors=k,n_jobs=-1)
        X_train_ftr, y_train_folds = sm.fit_resample(X_train_ftr, y_train_folds)
        X_train_ftr = pd.concat([X_train_ftr, y_train_folds.to_frame()], axis=1)
        print("smotenc time:",(time.time()-pastt)/60)
        print("X_train_ftr.shape:", X_train_ftr.shape)
        print("X_train_ftr.:", list(X_train_ftr.columns))
        print("y_train_folds.shape:", y_train_folds.shape)
        # print("y_train_folds.:", list(X_train_ftr.columns))

        lb_rytj = LabelBinarizer()
        lb_rytj.fit([1,2,3,9])
        rytj_ndarry = lb_rytj.transform(X_train_ftr['RYTJ'])
        X_train_ftr = pd.concat([X_train_ftr, pd.DataFrame(rytj_ndarry, columns=['RYTJ' + str(i) for i in range(rytj_ndarry.shape[1])])], axis=1)
        X_train_ftr.drop(columns='RYTJ', axis=1, inplace=True)
        print("X_train_ftr.shape:",X_train_ftr.shape)
        print("X_train_ftr.:", list(X_train_ftr.columns))

        # y_test_folds=X_test_folds['zfy_label']
        # X_test_ftr = X_test_folds[[i for i in column_base3 if i not in ['zfy_label']]]
        X_test_ftr = X_test_folds[column_base3].reset_index(drop=True)
        y_test_folds = X_test_folds['zfy_label']
        X_test_ftr.drop(columns='zfy_label', axis=1, inplace=True)

        rytj_ndarry = lb_rytj.transform(X_test_ftr['RYTJ'])
        X_test_ftr = pd.concat([X_test_ftr, pd.DataFrame(rytj_ndarry, columns=['RYTJ' + str(i) for i in range(rytj_ndarry.shape[1])])],axis=1)
        X_test_ftr.drop(columns='RYTJ', axis=1, inplace=True)
        print("X_test_ftr.shape:", X_test_ftr.shape)
        print("X_test_ftr.:", list(X_test_ftr.columns))

    elif ftr_type == 90 or ftr_type == 91 or ftr_type == 92 or ftr_type == 93 or ftr_type == 94:
        # "Baseline + CI + history + network(HCP,SD,CP)"  Random oversampling
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(history_column)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]

        y_train_folds = X_train_ftr['zfy_label']
        X_train_ftr.drop(columns='zfy_label', axis=1, inplace=True)

        print("random over sample--------------")
        pastt=time.time()
        if ftr_type == 90:
            sm = RandomOverSampler(random_state=42)
        elif ftr_type == 91:
            sm=RandomUnderSampler(random_state=42)
        elif ftr_type == 92:
            sm=SMOTE(random_state=42,n_jobs=-1)
        elif ftr_type == 93:
            sm=SMOTEENN(random_state=42,n_jobs=-1)
        elif ftr_type == 94:
            sm=NeighbourhoodCleaningRule(n_jobs=-1)
        X_train_ftr, y_train_folds = sm.fit_resample(X_train_ftr, y_train_folds)
        X_train_ftr = pd.concat([X_train_ftr, y_train_folds.to_frame()], axis=1)
        print("random over sample time:",(time.time()-pastt)/60)
        print("X_train_ftr.sum():", X_train_ftr['zfy_label'].sum())
        print("X_train_ftr.shape:", X_train_ftr.shape)
        print("X_train_ftr.:", list(X_train_ftr.columns))
        print("y_train_folds.shape:", y_train_folds.shape)
        # print("y_train_folds.:", list(X_train_ftr.columns))

        # y_test_folds=X_test_folds['zfy_label']
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]
        print("X_test_ftr.shape:", X_test_ftr.shape)
        print("X_test_ftr.:", list(X_test_ftr.columns))

    elif ftr_type == 95:
        # "Baseline + CI + history + network(HCP,SD,CP)"
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(history_column)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(lst_columns_comobidity)
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == 96:
        # "Baseline + CI + history + network(HCP,SD,CP)"

        # dic_ftr_groups_name = {"baseline": [i for i in column_base2 if i not in ['zfy_label']]}

        # 相较于86， 少了comorbidity propensity 网络特征，加了16_144vector
        column_base2.extend(lst_nm_onehot_eci)
        column_base2.extend(lst_nm_onehot_cci)
        column_base2.extend(history_column)
        column_base2.extend(['shortest_distance', 'hcp1_max'])
        column_base2.extend(['16_144vector'])
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]

    elif ftr_type == 97:
        column_base2 = ['zfy_label']
        # column_base2.extend(["hcp1_vector_" + str(i) for i in range(16)])
        # column_base2.extend(["dist_vector_" + str(i) for i in range(16)])

        column_base2.extend(['16_145vector'])
        X_train_ftr = X_train_folds[column_base2]
        X_test_ftr = X_test_folds[[i for i in column_base2 if i not in ['zfy_label']]]


    # 过采样
    # X_train_ftr_1=X_train_ftr[X_train_ftr['zfy_label']==1]
    # for i in range(18):
    #     X_train_ftr=pd.concat([X_train_ftr,X_train_ftr_1],axis=0)
    # print("X_train_ftr.shape[0]",X_train_ftr.shape[0])
    # X_train_ftr=X_train_ftr.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    # y_train=X_train_ftr['zfy_label']

    y_train_folds = X_train_ftr['zfy_label']
    X_train_ftr.drop(columns='zfy_label', axis=1, inplace=True)

    #实际住院费用
    # x_test_cost=X_test_folds[['zfy_label','ZFY']]
    # f0 = open("data_wo_threshold/results/x_test_actual_cost.pkl", 'wb')
    # pickle.dump(x_test_cost, f0)
    # f0.close()

    # 要调参
    # rnd_clf=LogisticRegression(random_state=0,n_jobs=-1,verbose=1,class_weight='balanced',max_iter=1000)
    # print("SVM:----------------")
    # pastt=time.time()
    # rnd_clf=SVC(gamma='auto',class_weight='balanced',random_state=0,verbose=1)
    # print("SVM耗时：--------------",(time.time()-pastt)/60)
    # rnd_clf=RandomForestClassifier(random_state=42,n_estimators=2000,class_weight='balanced',n_jobs=-1,max_depth=10,verbose=1)
    # rnd_clf = RandomForestClassifier(random_state=42, n_estimators=2000, n_jobs=-1, max_depth=7, verbose=1, max_samples=0.05)

    # 找thresholds   应该在不平衡的训练集上找到一个稳定不变的阈值运用于测试，而不是每次训练的时候都重新找一个阈值
    # cv_split=StratifiedKFold(n_splits=5, random_state=420, shuffle=True)
    # y_scores=cross_val_predict(rnd_clf,X_train_ftr, y_train,cv=cv_split,method='predict_proba')
    # # print(y_scores)  #[[0.31387426 0.68612574]...
    # # print(type(y_scores))
    # # print(y_scores.shape)
    # y_scores_df=pd.DataFrame(y_scores,columns=['0','1'])
    # precisions,recalls,threshold=precision_recall_curve(y_train,y_scores_df['1'])
    # print("len(threshold)",len(threshold))  #159789
    # # plot_precision_recall_vs_threshold(precisions, recalls, threshold)
    # # for thres_idx in range(len(threshold)):
    # precisions=np.array(precisions)
    # recalls=np.array(recalls)
    # subtract=abs(precisions-recalls)
    # min_idx=np.argmin(subtract)
    # thrshld=threshold[min_idx]
    # print("thrshld",thrshld)

    # 测试集若均衡
    # X_test_ftr_1 = X_test_ftr[X_test_ftr['zfy_label'] == 0]
    # X_test_ftr_1=X_test_ftr_1.sample(n=1736,random_state=42,axis=0)
    # X_test_ftr_2=X_test_ftr[X_test_ftr['zfy_label'] == 1]
    # X_test_ftr=pd.concat([X_test_ftr_1,X_test_ftr_2],axis=0).reset_index(drop=True)
    # y_test_folds=X_test_ftr['zfy_label']
    # X_test_ftr.drop(columns='zfy_label', axis=1, inplace=True)

    # 测试
    # rnd_clf.fit(X_train_ftr, y_train_folds)
    # y_pred_rf_train=rnd_clf.predict(X_train_ftr)
    # y_pred_rf=rnd_clf.predict(X_test_ftr)
    # 测试，输出为概率
    # y_pred_rf = rnd_clf.predict_proba(X_test_ftr)[:, 1]
    # print("type(y_pred_rf)",type(y_pred_rf))   # 一维ndarray
    # y_pred_rf_df=pd.DataFrame(y_pred_rf,columns=['prob_of_1'])
    # y_pred_rf_df['pred_label']=y_pred_rf_df['prob_of_1'].apply(lambda x:1 if x>=thrshld else 0)

    # 与特征重要性相关的两个变量：
    # X_columnsname=X_train_ftr.columns.values
    # feature_importance=rnd_clf.feature_importances_     # ndarray of shape (n_features,)

    # return y_test_folds,y_pred_rf,feature_importance,X_columnsname
    # return y_test_folds,y_pred_rf_df['pred_label'],feature_importance,X_columnsname
    return X_train_ftr, y_train_folds, X_test_ftr, y_test_folds


def my_Speciality_func(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    Speciality = m[0][0] / (m[0][0] + m[0][1])
    return Speciality


def my_MCC_func(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    list1=[[m[0][0],m[0][1]],[m[1][0],m[1][1]]]
    MCC=myfunction.compute_MCC(list1)
    return MCC


def my_Gmean_func(y_true, y_pred):
    Recall = recall_score(y_true, y_pred)
    Speciality = my_Speciality_func(y_true, y_pred)
    g_mean=np.sqrt(Recall*Speciality)
    return g_mean


def cv_paraSelect_LR(X_train_ftr, y_train_folds):
    """交叉验证"""
    # rnd_clf = LogisticRegression(random_state=42,n_jobs=-1,verbose=1,class_weight='balanced',penalty='l2',solver='lbfgs',max_iter=100,C=1)
    rnd_clf = LogisticRegression(random_state=42, n_jobs=-1, verbose=1, class_weight='balanced',penalty='l2',max_iter=150)
    # param_grid1 = [{'penalty': ['l1','l2']}]
    param_grid2=[{'C':[1,0.9,0.8],'max_iter':range(100, 301, 50)}]
    # param_grid3 = [{'solver': ['newton-cg','lbfgs','liblinear','sag','saga’]}]
    param_grid3 = [{'solver': ['newton-cg','lbfgs','liblinear','sag','saga']}]
    # 6: {'C': 0.8, 'class_weight': 'balanced', 'max_iter': 100}
    # param_grid = [{'class_weight': [{0: 2 * (10 / 19), 1: 10},{0: 1.8 * (10 / 19), 1: 10}, {0: 1.6 * (10 / 19), 1: 10},{0: 1.4 * (10 / 19), 1: 10},{0: 1.2 * (10 / 19), 1: 10},'balanced', {0: 0.8 * (10 / 19), 1: 10},
    #                                 {0: 0.6 * (10 / 19), 1: 10},{0: 0.4 * (10 / 19), 1: 10},{0: 0.2 * (10 / 19), 1: 10}],'max_iter':[500],'C': [0.8]}]
    param_grid=param_grid3

    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        },
                             n_jobs=-1,
                             refit='AUC',
                             return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_

    print(grid_search.best_params_)
    print()

    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))


def cv_paraSelect_RF(X_train_ftr, y_train_folds):
    """交叉验证"""
    rnd_clf = RandomForestClassifier(random_state=42,verbose=1,bootstrap=True,class_weight="balanced",n_estimators=350,max_depth=19,min_samples_split=90,min_samples_leaf=10,max_features=21)

    param_grid1=[ {'n_estimators':[250,300,350,400]}]
    # param_grid2 = [{'max_depth': [3,5,7,9,11,13,15]}]
    param_grid2={'max_depth': range(3, 22, 2), 'min_samples_split': range(50, 201, 20)}  #{'max_depth': 19, 'min_samples_split': 90}
    param_grid3 = {'min_samples_split': range(80, 101,10), 'min_samples_leaf': range(10, 51, 10)}
    param_grid4 = {'max_features': range(15, 22, 2)}

    # param_grid_calss_weight = [{'class_weight': [{0: 2 * (10 / 19), 1: 10},{0: 1.8 * (10 / 19), 1: 10}, {0: 1.6 * (10 / 19), 1: 10},{0: 1.4 * (10 / 19), 1: 10},{0: 1.2 * (10 / 19), 1: 10},'balanced', {0: 0.8 * (10 / 19), 1: 10},
    #                                 {0: 0.6 * (10 / 19), 1: 10},{0: 0.4 * (10 / 19), 1: 10},{0: 0.2 * (10 / 19), 1: 10}]}]

    param_grid=param_grid4
    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        },
                             n_jobs=-1,refit='AUC',return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_

    print(grid_search.best_params_)
    print()
    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))


def cv_paraSelect_balance_RF(X_train_ftr, y_train_folds):
    """交叉验证"""
    rnd_clf = BalancedRandomForestClassifier(random_state=42,verbose=1,n_jobs=-1,n_estimators=400,max_depth=22,min_samples_split=20,min_samples_leaf=2)

    param_grid1=[ {'n_estimators':[100,150,200,250,300,350,400]}]
    param_grid2 = [{'max_depth': [16,18,20,22,24]}]
    param_grid3={'max_depth': range(18, 23, 2), 'min_samples_split': range(70, 131, 20)}  #0.88324997 {'max_depth': 22, 'min_samples_split': 70}
    param_grid4 = {'min_samples_split': range(20,61,10), 'min_samples_leaf': range(2, 15, 3)}  #0.88629471  {'min_samples_leaf': 2, 'min_samples_split': 20}
    param_grid5 = {'min_samples_split': range(5, 21, 5), 'min_samples_leaf': range(2, 12, 3)}
    # param_grid4 = {'max_features': range(15, 22, 2)}

    param_grid=param_grid4
    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        },
                             n_jobs=-1,refit='AUC',return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_

    print(grid_search.best_params_)
    print()
    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))


def cv_paraSelect_KNN(X_train_ftr, y_train_folds):
    """交叉验证"""
    rnd_clf = KNeighborsClassifier(random_state=42,weights='uniform',n_neighbors=5)

    param_grid1=[ {'weights':['uniform','‘distance’']}]
    param_grid2={'n_neighbors': range(3, 16, 2)}

    param_grid=param_grid1
    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        },
                             n_jobs=3,refit='AUC',return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_

    print(grid_search.best_params_)
    print()
    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))


def cv_paraSelect_XGB(X_train_ftr, y_train_folds):
    """交叉验证"""
    rnd_clf =  xgb.XGBClassifier(scale_pos_weight=19,max_depth=11,learning_rate=0.01,n_estimators=500,random_state=42,objective='binary:logistic', use_label_encoder=False)

    param_grid1=[ {'learning_rate':[0.05],'scale_pos_weight': [19],'n_estimators':[100,200,300,400,500]}]
    param_grid2 = [{'learning_rate':[0.05],'scale_pos_weight': [19], 'max_depth': [3,5,7,9,11,13,15]}]  # 需要调整的哦！！！
    # 6: {'C': 0.8, 'class_weight': 'balanced', 'max_iter': 100}
    param_grid_calss_weight = [{'scale_pos_weight': [2*19,1.8*19,1.6*19,1.4*19,1.2*19,19,0.8*19,0.6*19,0.4*19,0.2*19],'learning_rate':[0.05]}]
    # 上面需要调整的哦！！！！！！！
    param_grid_lr=[{'learning_rate':[0.05],'n_estimators':[100]},
                   {'learning_rate':[0.03],'n_estimators':[300]},
                   {'learning_rate':[0.01],'n_estimators':[500]},
                   {'learning_rate':[0.005],'n_estimators':[1000]}]  # 需要调整的哦！！！！！

    param_grid=param_grid_lr  # 需要调整的哦！！
    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        },
                             n_jobs=-1,
                             refit=False,
                             return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_
    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))


def cv_paraSelect_lightGBM2(X_train_ftr, y_train_folds):
    """交叉验证"""
    # rnd_clf = lightgbm.LGBMClassifier(boosting_type='gbdt',max_depth=9,min_child_samples=110,n_estimators=400,
    #                                   num_leaves=30,colsample_bytree=0.4,subsample_freq=1,subsample=0.9,learning_rate=0.06,
    #                                   objective='binary', class_weight='balanced',reg_alpha=0.3,
    #                                   random_state=42, n_jobs=- 1, silent=True, importance_type='gain')
    rnd_clf = lightgbm.LGBMClassifier(boosting_type='gbdt', subsample_freq=0,n_estimators=100,
                                      learning_rate=0.1,min_child_samples=20,max_depth=7,num_leaves=22,
                                      objective='binary',
                                      random_state=42, n_jobs=- 1, silent=True, importance_type='gain')
    param_grid1=[ {'n_estimators':[20,30,40,50,60,70,80,90,100]}]
    param_grid2 = [{ 'max_depth': [3,5,7,9,11,13,15],'min_child_samples':range(20,101,10)}]
    #set 2^max_depth > num_leaves
    param_grid3 = [{'min_child_samples': range(10,51,5), 'num_leaves': range(2, 32, 4)}]
    param_grid4 = [{'colsample_bytree': [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]}]
    param_grid5 = [{'subsample': [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],'subsample_freq':[1]}]
    param_grid6 = [{'reg_alpha': [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]}]

    param_grid_lr=[{'learning_rate':[0.1],'n_estimators':[90]},
                   {'learning_rate':[0.09],'n_estimators':[100]},
                   {'learning_rate':[0.09],'n_estimators':[90]},
                   {'learning_rate':[0.08], 'n_estimators':[100]},
                   {'learning_rate':[0.1], 'n_estimators':[100]},
                   {'learning_rate':[0.1], 'n_estimators':[80]},
                   {'learning_rate':[0.12], 'n_estimators':[80]}]

    param_grid=param_grid_lr
    n_params=7

    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        'MCC':make_scorer(my_MCC_func, greater_is_better=True),
                                        },
                             n_jobs=-1,
                             refit='AUC',
                             return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_
    print(grid_search.best_params_)
    print()
    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])
    print('mean_test_MCC', dic_cv_result['mean_test_MCC'])
    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))

    x_name=[i for i in range(0,n_params)]
    myplt.show_class_weight(x_name,dic_cv_result)


def cv_paraSelect_lightGBM(X_train_ftr, y_train_folds):
    """交叉验证"""

    # rnd_clf = lightgbm.LGBMClassifier(boosting_type='gbdt', max_depth=9, min_child_samples=110, n_estimators=400,
    #                                   num_leaves=30, colsample_bytree=0.4, subsample_freq=1, subsample=0.9,
    #                                   learning_rate=0.06,
    #                                   objective='binary', class_weight='balanced', reg_alpha=0.3,
    #                                   random_state=42, n_jobs=- 1, silent=True, importance_type='gain')
    rnd_clf = lightgbm.LGBMClassifier(boosting_type='gbdt',n_estimators=400,max_depth=9,num_leaves=42,min_child_samples=180,
                                      colsample_bytree=0.6,objective='binary',class_weight='balanced',reg_alpha=0.1,learning_rate=0.05,
                                      random_state=42, n_jobs=- 1, importance_type='gain')
    param_grid1=[ {'n_estimators':[100,200,300,400,500]}]
    param_grid2 = [{ 'max_depth': [7,9,11,13,15],'min_child_samples':range(100,201,20)}]
    #set 2^max_depth < num_leaves
    param_grid3 = [{'min_child_samples': range(170,191,10), 'num_leaves': range(26, 51, 4)}]
    param_grid4 = [{'colsample_bytree': [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]}]
    param_grid5 = [{'subsample': [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],'subsample_freq':[1]}]
    param_grid6 = [{'reg_alpha': [0,0.1,0.2,0.3,0.4,0.5,0.6]}]

    param_grid_lr=[{'learning_rate':[0.1],'n_estimators':[200]},
                   {'learning_rate':[0.2],'n_estimators':[100]},
                   {'learning_rate':[0.05],'n_estimators':[400]},
                   {'learning_rate':[0.01], 'n_estimators':[400]},
                   {'learning_rate':[0.2], 'n_estimators':[200]},
                   {'learning_rate':[0.05], 'n_estimators':[200]}]

    param_grid=param_grid_lr
    # n_params=6

    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        'MCC':make_scorer(my_MCC_func, greater_is_better=True),
                                        },
                             n_jobs=-1,
                             refit='AUC',
                             return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_
    print(grid_search.best_params_)
    print()
    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])
    print('mean_test_MCC', dic_cv_result['mean_test_MCC'])
    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))


def cv_paraSelect_AdaBoost(X_train_ftr, y_train_folds):
    """交叉验证"""
    rnd_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5,min_samples_split=110,min_samples_leaf=5),random_state=1008,learning_rate=0.9,n_estimators=30)

    param_grid1 = [{'n_estimators': [30,40,50,60,70]}]

    param_grid_lr = [{'learning_rate': [1], 'n_estimators': [30]},
                     {'learning_rate': [0.9], 'n_estimators': [30]},
                     {'learning_rate': [0.9], 'n_estimators': [40]},
                     {'learning_rate': [0.8], 'n_estimators': [50]},
                     {'learning_rate': [0.8], 'n_estimators': [60]}]

    param_grid = param_grid_lr
    n_params = 5

    grid_search = GridSearchCV(rnd_clf,
                               param_grid,
                               cv=5,
                               scoring={'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        'MCC': make_scorer(my_MCC_func, greater_is_better=True),
                                        },
                               n_jobs=-1,
                               refit='AUC',
                               return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result = grid_search.cv_results_
    print(grid_search.best_params_)
    print()
    # print(dic_cv_result)
    print('mean_test_AUC', dic_cv_result['mean_test_AUC'])
    print('mean_test_f1', dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean', dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall', dic_cv_result['mean_test_recall'])
    print('mean_test_precision', dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy', dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality', dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))], dic_cv_result['params'])))

    x_name = [i for i in range(0, n_params)]
    myplt.show_class_weight(x_name, dic_cv_result)


def cv_paraSelect_ANN(X_train_ftr, y_train_folds):
    """交叉验证"""

    # rnd_clf = MLPClassifier(max_iter=200, alpha=1e-5, activation='relu', solver='adam', batch_size=200,
    #                         learning_rate_init=0.001, verbose=1,
    #                         hidden_layer_sizes=(100), random_state=1)
    rnd_clf = MLPClassifier(max_iter=500, alpha=1e-5, activation='tanh', solver='adam', batch_size=200,
                            learning_rate_init=0.001, verbose=1,hidden_layer_sizes=(400, 600, 400),
                            random_state=1)

    param_grid1=[ {'hidden_layer_sizes':[(100),(200),(50),(100,50),(300,50),(200,300),(200,100,50)]}]
                                        # [0.82096163 0.82278166 0.81055076 0.82212112 0.83480274 0.83592124 0.83181108]
    param_grid2 = [{ 'hidden_layer_sizes':[(300,400),(300,200),(300,500,300),(200,300,300,200),(200,300,100)]}]
                                        # [0.84271216 0.84359255 0.84580514 0.83942041 0.83700392]
    param_grid3 = [{'hidden_layer_sizes': [(400, 600, 400), (300, 500, 500, 300), (600, 400, 200),(500, 300, 500, 300)]}]
                                        # [0.84921151 0.84561723 0.84865867 0.84743576]
    param_grid4 = [{'hidden_layer_sizes': [(500, 600, 500), (600, 600, 500),(700, 800, 700), (700, 600, 700, 600)]}]
                                        # [0.84960067 0.85087883 0.85122158 0.84963272]
    param_grid5 = [{'batch_size': [100,200,300,400]}]
    param_grid6 = [{'activation': ['tanh']},{'learning_rate_init': [0.0005,0.002]}]
                                        #  [0.85499033 0.84334793 0.84475286]
    param_grid7 = [{'hidden_layer_sizes': [ (900, 1000, 900), (1000, 1000, 1000),]}]
                                        # [0.86044575 0.8460001 ]
    param_grid8 = [{'alpha': [0.0001,0.001]}]
    param_grid9 = [{'solver': ['sgd']}]
                     # [0.81414964]
    param_grid=param_grid9
    # n_params=6

    grid_search=GridSearchCV(rnd_clf,
                             param_grid,
                             cv=5,
                             scoring = {'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        'MCC':make_scorer(my_MCC_func, greater_is_better=True),
                                        },
                             n_jobs=-1,
                             refit='AUC',
                             return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result=grid_search.cv_results_
    print(grid_search.best_params_)
    print()

    f0 = open("data/feature_related/ann_best_params_.pkl", 'wb')
    pickle.dump(grid_search.best_params_, f0)
    f0.close()
    f0 = open("data/feature_related/ann_mean_test_AUC.pkl", 'wb')
    pickle.dump(dic_cv_result['mean_test_AUC'], f0)
    f0.close()

    # print(dic_cv_result)
    print('mean_test_AUC',dic_cv_result['mean_test_AUC'])
    print('mean_test_f1',dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean',dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall',dic_cv_result['mean_test_recall'])
    print('mean_test_precision',dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy',dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality',dic_cv_result['mean_test_Speciality'])
    print('mean_test_MCC', dic_cv_result['mean_test_MCC'])
    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))],dic_cv_result['params'])))



def cv_paraSelect_DT(X_train_ftr, y_train_folds):
    """交叉验证"""
    # rnd_clf = DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5)
    rnd_clf = DecisionTreeClassifier(random_state=42,class_weight="balanced",max_depth=10,min_samples_split=400,min_samples_leaf=120)
    param_grid1 = [{'max_depth': range(10, 31, 5),'min_samples_split': range(200, 301, 20)}]
    param_grid2= [{'min_samples_split': range(300, 401, 20),'min_samples_leaf': range(100, 200, 20)}]
    param_grid3 = [{'max_features': [10, 20, 30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,214]}]

    param_grid = param_grid3
    # n_params = 35

    grid_search = GridSearchCV(rnd_clf,
                               param_grid,
                               cv=5,
                               scoring={'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        'MCC': make_scorer(my_MCC_func, greater_is_better=True),
                                        },
                               n_jobs=-1,
                               refit='AUC',
                               return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result = grid_search.cv_results_
    print(grid_search.best_params_)
    print()
    # print(dic_cv_result)
    print('mean_test_AUC', dic_cv_result['mean_test_AUC'])
    print('mean_test_f1', dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean', dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall', dic_cv_result['mean_test_recall'])
    print('mean_test_precision', dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy', dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality', dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))], dic_cv_result['params'])))


def cv_paraSelect_svc(X_train_ftr, y_train_folds):
    """交叉验证"""
    localtime = time.asctime(time.localtime(time.time()))
    print("localtime: ",localtime)
    pastt=time.time()
    rnd_clf = SVC(class_weight='balanced', random_state=0, verbose=1,kernel='linear')
    param_grid1 = {"C": [0.1,0.5,1,3,5]}

    param_grid = param_grid1

    grid_search = GridSearchCV(rnd_clf,
                               param_grid,
                               cv=5,
                               scoring={'AUC': 'roc_auc',
                                        'f1': 'f1',
                                        'G_mean': make_scorer(my_Gmean_func, greater_is_better=True),
                                        'recall': 'recall',
                                        'precision': 'precision',
                                        'accuracy': 'accuracy',
                                        'Speciality': make_scorer(my_Speciality_func, greater_is_better=True),
                                        'MCC': make_scorer(my_MCC_func, greater_is_better=True),
                                        },
                               n_jobs=-1,
                               refit='AUC',
                               return_train_score=False, verbose=1)

    grid_search.fit(X_train_ftr, y_train_folds)
    dic_cv_result = grid_search.cv_results_
    print(grid_search.best_params_)

    f0 = open("data/feature_related/svc_best_params_.pkl", 'wb')
    pickle.dump(grid_search.best_params_, f0)
    f0.close()
    f0 = open("data/feature_related/svc_mean_test_AUC.pkl", 'wb')
    pickle.dump(dic_cv_result['mean_test_AUC'], f0)
    f0.close()

    print()
    # print(dic_cv_result)
    print('mean_test_AUC', dic_cv_result['mean_test_AUC'])
    print('mean_test_f1', dic_cv_result['mean_test_f1'])
    print('mean_test_G_mean', dic_cv_result['mean_test_G_mean'])
    print('mean_test_recall', dic_cv_result['mean_test_recall'])
    print('mean_test_precision', dic_cv_result['mean_test_precision'])
    print('mean_test_accuracy', dic_cv_result['mean_test_accuracy'])
    print('mean_test_Speciality', dic_cv_result['mean_test_Speciality'])

    print("index:params:")
    print(dict(zip([ind for ind in range(len(dic_cv_result['params']))], dic_cv_result['params'])))
    print("SVC交叉验证耗时：",(time.time()-pastt)/60)


def SVM_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    print("-------------------------------SVM训练...:----------------")
    pastt = time.time()
    rnd_clf = SVC(class_weight='balanced', random_state=0, verbose=1, kernel='linear')
    # 测试
    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------SVM耗时：--------------", (time.time() - pastt) / 60)
    y_pred_rf_train = []

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob = rnd_clf.predict_proba(X_test_ftr)[:, 1]
    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob


def RF_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    # 调参
    print("-------------------------------RF训练... :----------------")
    pastt = time.time()
    # rnd_clf = RandomForestClassifier(random_state=42, n_estimators=2000, class_weight='balanced', n_jobs=-1,
    #                                  max_depth=10, verbose=1)
    # rnd_clf = RandomForestClassifier(random_state=42, n_estimators=2000, n_jobs=-1, max_depth=7, verbose=1, max_samples=0.05)
    # rnd_clf = RandomForestClassifier(n_jobs=-1,random_state=42, bootstrap=True, n_estimators=400, max_depth=12,
    #                                  class_weight={0: 1.05 * (10 / 19), 1: 10},verbose=1)
    rnd_clf = RandomForestClassifier(random_state=42, verbose=1, bootstrap=True,class_weight="balanced",
                                     n_estimators=350, max_depth=19, min_samples_split=90, min_samples_leaf=10,
                                     n_jobs=-1)
    # class_weight="balanced",
    # 测试
    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------RF 耗时：--------------", (time.time() - pastt) / 60)

    y_pred_rf_train=[]

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]
    # 与特征重要性相关的两个变量：
    X_columnsname = X_train_ftr.columns.values
    feature_importance = rnd_clf.feature_importances_  # ndarray of shape (n_features,)
    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob


def balanceRF_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    # 调参
    print("-------------------------------balanceRF训练... :----------------")
    pastt = time.time()
    rnd_clf = BalancedRandomForestClassifier(random_state=42, verbose=1, n_jobs=-1, n_estimators=400, max_depth=22,
                                             min_samples_split=20, min_samples_leaf=2)
    # 测试
    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------balanceRF 耗时：--------------", (time.time() - pastt) / 60)

    y_pred_rf_train=[]

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]
    # 与特征重要性相关的两个变量：
    X_columnsname = X_train_ftr.columns.values
    feature_importance = rnd_clf.feature_importances_  # ndarray of shape (n_features,)
    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob


def balance_RF_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    # 调参
    print("-------------------------------RF训练... :----------------")
    pastt = time.time()

    rnd_clf = BalancedRandomForestClassifier(random_state=0)
    # class_weight="balanced",

    # 测试
    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------RF 耗时：--------------", (time.time() - pastt) / 60)

    y_pred_rf_train=[]

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]
    # 与特征重要性相关的两个变量：
    X_columnsname = X_train_ftr.columns.values
    feature_importance = rnd_clf.feature_importances_  # ndarray of shape (n_features,)
    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob


def GBDT_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    #  调参
    print("-------------------------------GBDT 训练... :----------------")
    pastt = time.time()
    rnd_clf = GradientBoostingClassifier(random_state=42, n_estimators=500,learning_rate=0.05,max_depth=7, verbose=1,subsample=0.05)
    # 测试
    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------GBDT 耗时：--------------", (time.time() - pastt) / 60)
    y_pred_rf_train = rnd_clf.predict(X_train_ftr)
    y_pred_rf = rnd_clf.predict(X_test_ftr)

    # 与特征重要性相关的两个变量：
    X_columnsname = X_train_ftr.columns.values
    feature_importance = rnd_clf.feature_importances_  # ndarray of shape (n_features,)

    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname


def xgb_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds,pos_weight):
    print("-------------------------------XGB 训练... :----------------")
    pastt = time.time()
    # rnd_clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, learning_rate=0.05, verbosity=1, random_state=42,
    #                           objective='binary:logistic', scale_pos_weight=19,use_label_encoder=False)
    rnd_clf = xgb.XGBClassifier(scale_pos_weight=pos_weight, max_depth=11, learning_rate=0.01, n_estimators=500,
                                random_state=42, verbosity=1, objective='binary:logistic', use_label_encoder=False)

    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------XGB 耗时：--------------", (time.time() - pastt) / 60)
    y_pred_rf_train = rnd_clf.predict(X_train_ftr)
    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]

    # 与特征重要性相关的两个变量：
    X_columnsname = X_train_ftr.columns.values
    feature_importance = rnd_clf.feature_importances_  # ndarray of shape (n_features,)

    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob


def lightGBM_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds,class_w):
    print("-------------------------------LightGBM 训练... :----------------")
    pastt = time.time()

    rnd_clf = lightgbm.LGBMClassifier(boosting_type='gbdt', max_depth=9, min_child_samples=110, n_estimators=400,
                                      num_leaves=30, colsample_bytree=0.4, subsample_freq=1, subsample=0.9,
                                      learning_rate=0.06,class_weight=class_w,
                                      objective='binary',reg_alpha=0.3,
                                      random_state=42, n_jobs=- 1, silent=True, importance_type='gain')
    #  class_weight='balanced',
    # rnd_clf = lightgbm.LGBMClassifier(boosting_type='gbdt', n_estimators=400, max_depth=9, num_leaves=42,
    #                                   min_child_samples=180,
    #                                   colsample_bytree=0.6, objective='binary', class_weight='balanced', reg_alpha=0.1,
    #                                   learning_rate=0.05,
    #                                   random_state=42, n_jobs=- 1, importance_type='gain')
    rnd_clf.fit(X_train_ftr, y_train_folds)
    # evals_result = model.evals_result()
    # print(evals_result)
    print("-------------------------------LightGBM 耗时：--------------", (time.time() - pastt) / 60)

    # y_pred_rf_train = rnd_clf.predict(X_train_ftr)
    y_pred_rf_train=[]

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]
    # 与特征重要性相关的两个变量：
    X_columnsname = X_train_ftr.columns.values
    feature_importance = rnd_clf.feature_importances_  # ndarray of shape (n_features,)


    # f0 = open("data/temporary/lgbm_clf_86.pkl", 'wb')
    # pickle.dump(rnd_clf, f0)
    # f0.close()
    # f0 = open("data/temporary/ftr_clmn_lst.pkl", 'wb')
    # pickle.dump(list(X_train_ftr.columns), f0)
    # f0.close()

    # f2 = open("data/temporary/X_test_hierarchy.pkl", 'rb')
    # X_test_hierarchy = pickle.load(f2)
    # f2.close()

    # print("X_test_hierarchy.shape:",X_test_hierarchy.shape)
    # X_test_hierarchy=pd.concat([X_test_hierarchy,y_test_folds.to_frame(),pd.DataFrame(y_pred_rf,columns=['pred_class']),pd.DataFrame(y_pred_prob,columns=['pred_proba'])],axis=1)
    # print("X_test_hierarchy.shape:", X_test_hierarchy.shape)
    # print("X_test_hierarchy.columns:", list(X_test_hierarchy.columns))
    # f0 = open("data/temporary/X_test_hierarchy.pkl", 'wb')
    # pickle.dump(X_test_hierarchy, f0)
    # f0.close()



    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob



def LR_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds,class_w):
    print("-------------------------------LR 训练... :----------------")
    pastt = time.time()

    rnd_clf = LogisticRegression(random_state=42,class_weight=class_w, n_jobs=-1, verbose=1, penalty='l2',max_iter=150)
    # class_weight='balanced',
    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------LR 耗时：--------------", (time.time() - pastt) / 60)
    y_pred_rf_train=[]

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]

    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob


def KNN_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    print("-------------------------------LR 训练... :----------------")
    pastt = time.time()

    rnd_clf =  KNeighborsClassifier(n_neighbors=3)

    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------LR 耗时：--------------", (time.time() - pastt) / 60)
    y_pred_rf_train=[]

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]

    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob


def DT_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    # 调参
    print("-------------------------------DT训练... :----------------")
    pastt = time.time()
    rnd_clf = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=400,class_weight="balanced",
                                     min_samples_leaf=120)
    # class_weight="balanced",
    # 测试
    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------DT 耗时：--------------", (time.time() - pastt) / 60)

    y_pred_rf_train = []

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob = rnd_clf.predict_proba(X_test_ftr)[:, 1]
    # 与特征重要性相关的两个变量：
    X_columnsname = X_train_ftr.columns.values
    feature_importance = rnd_clf.feature_importances_  # ndarray of shape (n_features,)
    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname, y_pred_prob



def ANN_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds):
    print("-------------------------------ANN 训练... :----------------")
    pastt = time.time()
    rnd_clf = MLPClassifier(max_iter=500, alpha=1e-5, activation='tanh', solver='adam', batch_size=200,
                            learning_rate_init=0.001, verbose=1,hidden_layer_sizes=(400, 600, 400),
                            random_state=1)

    rnd_clf.fit(X_train_ftr, y_train_folds)
    print("-------------------------------ANN 耗时：--------------", (time.time() - pastt) / 60)
    y_pred_rf_train=[]

    y_pred_rf = rnd_clf.predict(X_test_ftr)
    y_pred_prob=rnd_clf.predict_proba(X_test_ftr)[:, 1]

    return y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, y_pred_prob


def plot_precision_recall_vs_threshold(precisions, recalls, threshold):
    plt.plot(threshold, precisions[:-1], "b--", label="precision")
    plt.plot(threshold, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()


def evaluation_feature_importantce(feature_importance, X_columnsname):
    print("特征重要性：")
    ftr_imptnt = {}

    for name, score in zip(X_columnsname, feature_importance):
        ftr_imptnt[name] = score
    ftr_imptnt = sorted(ftr_imptnt.items(), key=lambda kv: (kv[1]), reverse=True)
    ftr_imptnt_dic=dict(ftr_imptnt)

    # for i in ftr_imptnt:
    #     print(i[0])
    print(list(ftr_imptnt_dic.keys()))
    # print(ftr_imptnt)  # 特征重要性元组
    return ftr_imptnt


def evaluation_index(y_test_folds, y_pred_rf,y_pred_prob):
    # print("分类报告：",classification_report(y_test_folds,y_pred_rf))
    print("evaluation_index")
    print(y_test_folds.shape)
    print(y_pred_rf.shape)
    print(y_pred_prob.shape)

    m = sm.confusion_matrix(y_test_folds, y_pred_rf)
    print('混淆矩阵为：', m, sep='\n')
    # fpr,tpr,thresholds=sm.roc_curve(y_test_folds,y_pred_rf)
    # plt.figure(figsize=(10,6))
    # plt.xlim(0,1)
    # plt.ylim(0.0,1.1)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Postive Rate')
    # plt.plot(fpr,tpr,linewidth=2,linestyle="-",color='red')
    # plt.show()
    list_cf_mtx=[[m[0][0],m[0][1]],[m[1][0],m[1][1]]]
    Precision = precision_score(y_test_folds, y_pred_rf)
    Recall = recall_score(y_test_folds, y_pred_rf)
    Speciality = m[0][0] / (m[0][0] + m[0][1])
    Accuracy = (m[0][0] + m[1][1]) / (m[0][0] + m[1][1] + m[0][1] + m[1][0])
    F1_score = f1_score(y_test_folds, y_pred_rf)
    AUC = sm.roc_auc_score(y_test_folds, y_pred_prob)
    G_mean=np.sqrt(Speciality*Recall)
    MCC=myfunction.compute_MCC(list_cf_mtx)

    print("Recall：", Recall)
    print("Specificity：", Speciality)
    print("AUC值：", AUC)
    print("G_mean：",G_mean)
    print("Accuracy：", Accuracy)
    print("Precision：", Precision)
    print("f1_score：", F1_score)
    print("MCC：",MCC)
    # print("袋外样本来估计泛化精度:",rnd_clf.oob_score_)

    # precision, recall, thresholds2 = sm.precision_recall_curve(y_test_folds,y_pred_rf)
    # plt.figure(figsize=(10, 6))
    # plt.xlim(0, 1)
    # plt.ylim(0.0, 1.1)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.plot(recall, precision, linewidth=2, linestyle="-", color='blue')
    # plt.show()
    return Recall, Speciality, AUC, G_mean,Accuracy,Precision, F1_score,MCC


def evaluation_feature_importantce_for_plot(feature_importance, X_columnsname):
    print("特征重要性：")
    # ftr_imptnt = {}

    ftr_imptnt = dict(zip(X_columnsname, feature_importance))
    ftr_imptnt = sorted(ftr_imptnt.items(), key=lambda kv: (kv[1]), reverse=True)
    # for i in ftr_imptnt:
    #     print(i[0])

    # print(ftr_imptnt)  # 特征重要性元组
    return ftr_imptnt


def importance_figure(feature_importance, X_columnsname,top_num,columns_name,lst_nm_onehot_eci,lst_nm_onehot_cci,EC_onehot_name):
    ftr_imptnt = evaluation_feature_importantce_for_plot(feature_importance, X_columnsname)
    feature_importance_df = pd.DataFrame(ftr_imptnt, columns=['feature', 'importance'])
    # cols = (feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
    #                                                                                                ascending=False)[
    #         :1000].index)
    # best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    netwk_ftr_lst=['shortest_distance', 'hcp1_max']
    netwk_ftr_lst.extend(EC_onehot_name)
    histry_ftr_lst=['los_last_time', 'zfy_last_time', 'weighted_average_cost_past_3y', 'med_zfy_last_3y',
                             'max_zfy_last_3y',
                             'zyci_last_3y', 'last_time_cost_per_day', 'min_zfy_last_3y', 'std_zfy_last_3y',
                             'interval_thistimery_pasttimecy']
    CI_ftr_lst=lst_nm_onehot_eci
    CI_ftr_lst.extend(lst_nm_onehot_cci)
    disease_group_ftr_lst=columns_name
    feature_importance_df['feature group']=feature_importance_df['feature'].apply(lambda x:'network' if x in netwk_ftr_lst else ('history' if x in histry_ftr_lst else('CI' if x in CI_ftr_lst else('disease group' if x in disease_group_ftr_lst else 'baseline'))))

    best_features = feature_importance_df.head(top_num)
    # plt.figure(figsize=(15, feature_importance_df.shape[0] * 0.22))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False),hue="feature group",dodge=False, orient="h")
    plt.title('Features Importance')
    plt.show()



def LDA_plot(Precision_lst, Recall_lst, Speciality_lst, Accuracy_lst, F1_score_lst, AUC_lst,G_mean_lst):
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    x_name = ['PCA-5','PCA-10','PCA-20','PCA-30','PCA-40','213组']

    plt.plot(x_name, AUC_lst, '.-', label='AUC')
    plt.plot(x_name, F1_score_lst, '.-', label='f1')
    plt.plot(x_name, G_mean_lst, '.-', label='G_mean')
    plt.plot(x_name, Recall_lst, '.-', label='recall')
    plt.plot(x_name, Precision_lst, '.-', label='precision')
    plt.plot(x_name, Accuracy_lst, '.-', label='accuracy')
    plt.plot(x_name, Speciality_lst, '.-', label='Speciality')
    # plt.xticks(rotation=30)
    plt.xlabel("PCA降维后维度")
    plt.legend(loc='lower right')
    plt.show()


def controlled_experiment_plot(results_evlt,list_section):
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    x_name = ['Precision', 'Recall', 'Speciality', 'Accuracy', 'F1_score', 'AUC','G_mean']
    index=-1
    for i in list_section:
        index+=1
        plt.plot(x_name, results_evlt[index], '.-', label='ftr_flag='+str(i))

    # plt.xticks(rotation=30)
    # plt.xlabel("")
    plt.legend(loc='lower right')
    plt.show()


# if __name__ == "__main__":
#     dir = "data"
#     gml_path = dir + "/gml_dir/CC_all_modularity.gml"
#     prevalence_path = dir + "/med_data_mtx_dic/dic_disease_prevalence_rate.pkl"
#     xlsx_path =  "data/csv_xslx/ICD3_group_disease_names.xlsx"
#     xlsxpath="data/csv_xslx/AA_MULTI_DESC_TENSOR.xlsx"
#     # 处理基础特征
#     df,columns_name = read_data_feature_extract(xlsx_path,xlsxpath,0.95)   #220疾病组的columns_name
#
#     # # # 添加ECI和CCI特征
#     # df,lst_nm_onehot_cci,lst_nm_apper_time_cci,lst_nm_score_cci,lst_nm_onehot_eci,lst_nm_apper_time_eci,lst_nm_score_eci= myfunction.add_ftr_ECI_CCI(df)
#     # #将eci_cci特征的名字装入一个list中，传入data_preprcss_and_split（）函数中
#     # lst_name_of_cci_eci=[lst_nm_onehot_cci,lst_nm_apper_time_cci,lst_nm_score_cci,lst_nm_onehot_eci,lst_nm_apper_time_eci,lst_nm_score_eci]
#     lst_name_of_cci_eci = [[], [], [], [], [], []]
#
#     # 处理网络特征,划分训练集和测试集  （因为网络特征和OR特征要用训练集数据计算，所以，先划分训练集和测试集，再提取网络特征）
#     X_train_raw, X_test_raw, y_train_raw, y_test_raw, numOfModule,EC_onehot_name = add_network_features(gml_path, df, prevalence_path, 0.95)
#     # 标准化处理连续性特征
#     X_train_raw, X_test_raw, y_train_raw, y_test_raw=ftr_preprcss_standard( X_train_raw, X_test_raw, y_train_raw, y_test_raw,EC_onehot_name)
#
#     results_evlt = []
#     tuble_section = []
#     for i in [3]: # 不同的特征组合
#         tuble_section.append(i)
#         X_train_ftr, y_train_folds, X_test_ftr, y_test_folds = data_preprcss_and_split(i, X_train_raw, X_test_raw,y_train_raw, y_test_raw, numOfModule,columns_name,lst_name_of_cci_eci)
#         y_test_folds, y_pred_rf, y_pred_rf_train, y_train_folds, feature_importance, X_columnsname,y_pred_prob=xgb_prdct(X_train_ftr, y_train_folds, X_test_ftr, y_test_folds,'balanced')
#
#         # 混淆矩阵,各评估指标
#         print("featrue_set_flag=",i)
#         print("测试集上的表现")
#         Recall, Speciality, AUC, G_mean, Accuracy, Precision, F1_score, MCC = evaluation_index(y_test_folds, y_pred_rf,y_pred_prob)
#
#         # 特征重要性
#         ftr_imptnt = evaluation_feature_importantce(feature_importance, X_columnsname)
#
#         results_evlt.append([Precision, Recall, Speciality, Accuracy, F1_score, AUC, G_mean])
#     controlled_experiment_plot(results_evlt, tuble_section)


import pickle
import matplotlib.pyplot as plt
import cx_Oracle
import pandas as pd
from igraph import *
import networkx as nx
import time
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from my_function.myplot import hcp_prevalence
import seaborn as sns
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.cluster import KMeans,Birch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import multiprocessing   # 导入进程模块
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer,confusion_matrix

def add_ftr_ECI_CCI(df):
    """计算ECI和CCI"""
    print("-----------------计算ECI和CCI中--------------------------------------")
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    # db = cx_Oracle.connect('system', 'Yp251200.', '127.0.0.1:1521/ORCL', mode=cx_Oracle.SYSDBA)
    # 操作游标
    cr1 = db.cursor()
    sql1 = 'select SFZH,RN,JBDM,JBDM1,JBDM2,JBDM3,JBDM4,JBDM5,JBDM6,JBDM7,JBDM8,JBDM9,JBDM10,JBDM11,JBDM12,JBDM13,JBDM14,JBDM15,ALL_FLAGS from scott.YP_TTL_IHD_2YEARS'
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    # print(cyzd_data)   list of data
    names = [i[0] for i in cr1.description]
    # print(names)
    df_right_icd4 = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()

    # 左merge
    print("左连接前，df的长度：",df.shape)
    df = df.merge(df_right_icd4, how='left', on=['SFZH','RN'])
    print("左连接后，df的长度：",df.shape)
    print("################################## df的长度不应该发生改变 ##################################")
    # 对df进行操作，取四位编码：
    df['rybq_flags'] = df['ALL_FLAGS'].apply(lambda x: [i for i in x.split(',')])
    df.drop(columns='ALL_FLAGS', axis=1, inplace=True)

    JBDM_list= ['JBDM', 'JBDM1', 'JBDM2', 'JBDM3', 'JBDM4', 'JBDM5', 'JBDM6', 'JBDM7', 'JBDM8', 'JBDM9', 'JBDM10', 'JBDM11', 'JBDM12', 'JBDM13', 'JBDM14', 'JBDM15']
    cyzd_data = np.array(df).tolist()
    # 只保留入院诊断（QYBQ=1）
    list_ry_diaseses_4 = []
    dic_column_name_index=dict(zip(list(df.columns),[index2 for index2 in range(len(list(df.columns)))]))
    dic_column_name_index['JBDM0']=dic_column_name_index['JBDM']
    for i in tqdm(cyzd_data):
        index = -1
        list_ry_disease_perR = []
        for j in i[dic_column_name_index['rybq_flags']]:
            index += 1
            if j == '1':
                str_index = 'JBDM' + str(index)
                disease=i[dic_column_name_index[str_index]]
                if disease is  not None:
                    disease_4icd=disease[0:3]+disease[4:5]
                    list_ry_disease_perR.append(disease_4icd)
        list_ry_diaseses_4.append(list_ry_disease_perR)  # 入院疾病列表 list of lists :every list in the outer list is a record

    df = pd.concat([df, pd.DataFrame({'ry_diseases_icd4': list_ry_diaseses_4})], axis=1)
    df.drop(columns='rybq_flags', axis=1, inplace=True)
    df.drop(columns=JBDM_list, axis=1, inplace=True)
    del list_ry_diaseses_4
    # 对四位编码：剔除一下异常编码(同三位编码的剔除规则)
    df['ry_diseases_icd4'] = df['ry_diseases_icd4'].apply(lambda x: [i for i in x if len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[2] <= '9'])
    # 打印一个csv出来看看icd4和icd3
    df.to_csv("data/feature_related/df_include_icd4.csv")
    # CCI特征
    dic_CCI_3,dic_CCI_4,category_name_cci=CCI_ECI_dic('CCI')
    df_onehot_cci,df_apper_time_cci,df_score_cci=compute_cci_eci('CCI',dic_CCI_3, dic_CCI_4, category_name_cci, df)
    # ECI特征
    dic_ECI_3, dic_ECI_4, category_name_eci = CCI_ECI_dic('ECI')
    df_onehot_eci, df_apper_time_eci, df_score_eci = compute_cci_eci('ECI',dic_ECI_3, dic_ECI_4, category_name_eci, df)
    df=pd.concat([df,df_onehot_cci,df_apper_time_cci,df_score_cci,df_onehot_eci, df_apper_time_eci, df_score_eci],axis=1)
    return df,list(df_onehot_cci.columns),list(df_apper_time_cci.columns),list(df_score_cci.columns),list(df_onehot_eci.columns),list(df_apper_time_eci.columns),list(df_score_eci.columns)

def compute_cci_eci(flag,dic_CCI_3,dic_CCI_4,category_name,df):
    """可计算ECI和CCI特征
    flag=='ECI'或'CCI'
    dic_CCI_3：传入ECI或者CCI相应的三位编码字典：{'I21': ('Myocardial_Infarction', 1),......} （由函数CCI_ECI_dic（）返回得到）
    dic_CCI_4：传入ECI或者CCI相应的四位编码字典
    category_name：传入ECI或者CCI相应的类别名称
    df
    return:三组新特征，df格式
    """
    dic_category_name=dict(zip(category_name,[cn for cn in range(len(category_name))]))

    onehot_lists = []
    appear_tm_lsts = []
    score_lists = []
    for index, ry_dss in enumerate(df["ry_diseases_icd4"]):  #按四位编码来
        # 先四位，再三位
        onehot_list = [0 for m in range(len(category_name))]
        appear_tm_lst=[0 for m in range(len(category_name))]
        score =0
        for ry_d in ry_dss:
            if ry_d in dic_CCI_4:
                onehot_list[dic_category_name[dic_CCI_4[ry_d][0]]]=1  # onehot
                appear_tm_lst[dic_category_name[dic_CCI_4[ry_d][0]]]+= 1  # appear_time_hot
                score+= dic_CCI_4[ry_d][1]  # score
            elif ry_d[0:3] in dic_CCI_3:
                onehot_list[dic_category_name[dic_CCI_3[ry_d[0:3]][0]]] = 1  # onehot
                appear_tm_lst[dic_category_name[dic_CCI_3[ry_d[0:3]][0]]] += 1  # appear_time_hot
                score+= dic_CCI_3[ry_d[0:3]][1]  # score
        onehot_lists.append(onehot_list)
        appear_tm_lsts.append(appear_tm_lst)
        score_lists.append(score)

    df_onehot=pd.DataFrame(onehot_lists,columns=[flag+"_onehot_"+category for category in category_name])
    df_apper_time=pd.DataFrame(appear_tm_lsts,columns=[flag+"_appeartime_"+category for category in category_name])
    df_score=pd.DataFrame(score_lists,columns=[flag+"_score_sum"])
    return df_onehot,df_apper_time,df_score

def CCI_ECI_dic(flag):
    """获取CCI字典或者ECI字典，
    flag_:若flag=='CCI'，则返回CCI字典；若flag=='ECI'，则返回ECI字典；
    return：字典：{'I21': ('Myocardial_Infarction', 1),......}
    :return:category_name:ndarry，共病组合的类别名"""
    if flag=='CCI':
        path='data/csv_xslx/Charlson comorbidity index(CCI).csv'
    if flag=='ECI':
        path='data/csv_xslx/Elixhauser comorbidity index(ECI).csv'
    df_CCI = pd.read_csv(path, encoding="gbk")
    df_CCI_3=df_CCI[df_CCI.Code_len==3]
    df_CCI_4=df_CCI[df_CCI.Code_len==4]
    tub_ctgry_scr_3=zip(df_CCI_3['Category'].values,df_CCI_3['score'].values)
    dic_CCI_3=dict(zip(df_CCI_3['Code'].values,tub_ctgry_scr_3))
    tub_ctgry_scr_4=zip(df_CCI_4['Category'].values,df_CCI_4['score'].values)
    dic_CCI_4=dict(zip(df_CCI_4['Code'].values,tub_ctgry_scr_4))
    category_name=df_CCI['Category'].drop_duplicates().values
    return dic_CCI_3,dic_CCI_4,category_name


def my_up_sample(X_train_ftr):
    # 过采样
    X_train_ftr_1=X_train_ftr[X_train_ftr['zfy_label']==1]
    for i in range(18):
        X_train_ftr=pd.concat([X_train_ftr,X_train_ftr_1],axis=0)
    print("X_train_ftr.shape[0]",X_train_ftr.shape[0])
    X_train_ftr=X_train_ftr.sample(frac=1,random_state=829).reset_index(drop=True)  # 打乱顺序
    # y_train=X_train_ftr['zfy_label']
    return X_train_ftr

def compute_MCC(list1):
    TN=list1[0][0]
    FP=list1[0][1]
    FN=list1[1][0]
    TP=list1[1][1]
    return (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

def NCR_imblearn(X_train_ftr):
    """调库实现ncr"""
    y_train_folds = X_train_ftr['zfy_label']
    X_train_ftr.drop(columns=['zfy_label'], axis=1, inplace=True)
    len_bfr=y_train_folds.shape[0]
    print("ncr--------------------------:")
    pastt = time.time()
    ncr = NeighbourhoodCleaningRule(n_jobs=-1)

    X_train_ftr, y_train_folds = ncr.fit_resample(X_train_ftr, y_train_folds)
    print("ncr耗时-----------------------：", (time.time() - pastt) / 60)
    X_train_ftr.reset_index(drop=True)
    y_train_folds.reset_index(drop=True)
    X_train_ftr=pd.concat([X_train_ftr,y_train_folds],axis=1)
    print("-----------------ncr清除样本个数：",len_bfr-X_train_ftr.shape[0])
    return X_train_ftr


def Neighbor_clearn_rule(df,savepath):
    """
    实现Neighbor Clean Rule,对训练集进行下采样
    将训练集传入
    """
    print("Neighbor Clean Rule----------")

    df=df.reset_index(drop=True)
    # df_0_row=df[df['zfy_label']==0]
    # df_1_row=df[df['zfy_label']==1]
    # len_df_0_sm=df_0_row.shape[0]
    # df_0_sm=df_0_row.drop_duplicates()
    # print("负类中完全一样的样本数：(删除)",len_df_0_sm-df_0_sm.shape[0])
    # df=pd.concat([df_0_sm,df_1_row],axis=0).reset_index(drop=True)
    print("NearestNeighbors------------------")
    pastt=time.time()
    neigh = NearestNeighbors(n_neighbors=4,n_jobs=-1)
    y=df['zfy_label']
    x=df.drop(columns='zfy_label', axis=1, inplace=False)
    print(x.shape)
    neigh.fit(x)
    neighbor4_lsts=neigh.kneighbors(x, return_distance=False)   #样本x和三个最近邻的索引#<class 'numpy.ndarray'>
    print("NearestNeighbors-----------耗时：", (time.time() - pastt) / 60)
    pastt1=time.time()
    neighbor4_df=pd.DataFrame(neighbor4_lsts,columns=['ngb0_idx','ngb1_idx','ngb2_idx','ngb3_idx'])
    df_index=pd.DataFrame([i for i in range(neighbor4_df.shape[0])],columns=['x_idx'])
    neighbor4_df=pd.concat([df_index,neighbor4_df,y],axis=1)  #最近邻1，2，3的索引和x的标签

    neighbor4_df['ngb1_label']=neighbor4_df['ngb1_idx'].apply(lambda x:y.loc[x])
    neighbor4_df['ngb2_label'] = neighbor4_df['ngb2_idx'].apply(lambda x: y.loc[x])
    neighbor4_df['ngb3_label'] = neighbor4_df['ngb3_idx'].apply(lambda x: y.loc[x])

    # 若 x 为多数类 y=0
    neighbor4_df['sum_of_ngb_lbl']=neighbor4_df['ngb1_label']+neighbor4_df['ngb2_label']+neighbor4_df['ngb3_label']
    for index,df_group in neighbor4_df.groupby(['zfy_label']):   #index 的取值：1和0  1对应于高花费住院记录，2对应于非高花费住院记录
        if index==0:  # 多数类
            df_0_all=df_group
            print("未进行NCR时，负样本个数：",df_0_all.shape[0])
            # df_0=df_group[~(df_group['sum_of_ngb_lbl']>=2)]  #删除最近邻中有两个及以上少数类的多数类样本
            # print("NCR第一步删除的多数类样本个数：",df_group.shape[0]-df_0.shape[0])
        elif index == 1:  # 少数类
            df_1_all=df_group  #复制一份副本，存放所有少数类样本的index

    # df_0_all=df_0_all[~(df_0_all['sum_of_ngb_lbl']>=2)]

    df_1=df_1_all[df_1_all['sum_of_ngb_lbl'] <=1]   # x为少数类样本，选出最近邻中有两个及以上多数类的少数类样本

    lst_dlt=pd.concat([df_1['ngb1_idx'],df_1['ngb2_idx'],df_1['ngb3_idx']],axis=0).drop_duplicates().values # list: 所有要删除的多数类样本的index
    # list_1_index=list(df_1_all.index)
    list_1_index=df_1_all['x_idx'].tolist()   #所有少数类样本的索引 list
    lst_dlt2=[r for r in lst_dlt if r not in list_1_index ] #要删除的多数类的索引
    del lst_dlt
    # for i in tqdm(lst_dlt2):
    #     df_0_all=df_0_all[~(df_0_all['x_idx']==i)]

    df_0_all=df_0_all[(df_0_all['sum_of_ngb_lbl']>=2)]
    df_0_all=df_0_all['x_idx'].tolist()   #要删除的多数类的索引 list

    lst_dlt2.extend(df_0_all)
    set_dlt=set(lst_dlt2)
    lst_dlt2=list(set_dlt)
    df.drop(index=lst_dlt2,axis=0,inplace=True)
    df.reset_index(drop=True)
    # df=pd.concat([df,df_index],axis=1)
    # for i in tqdm(set_dlt):
    #     df = df[~(df['x_idx'] == i)]
    print("训练集样本个数：", df.shape[0])
    print("NCR后，去除负样本个数：",len(lst_dlt2))
    # print("正样本个数：",df_1_all)
    # df=pd.concat([df_0_all['x_idx'],df_1_all['x_idx']],axis=0)
    # print()
    # df_0_all=df_0_all[df_0_all['x_idx'] not in lst_dlt2]
    # # df_0=df_group[~(df_group['sum_of_ngb_lbl']>=2)]  #删除最近邻中有两个及以上少数类的多数类样本
    # # print("NCR第一步删除的多数类样本个数：",df_group.shape[0]-df_0.shape[0])
    # for i in lst_dlt:
    #     if i in

    print("Neighbor Clean Rule-----------耗时：",(time.time()-pastt1)/60)

    f0 = open(savepath, 'wb')
    pickle.dump(df, f0)
    f0.close()

    return df


def Neighbor_clearn_rule2(df,savepath,columns_name):
    """
    实现Neighbor Clean Rule,对训练集进行下采样
    将训练集传入
    """
    print("Neighbor Clean Rule----------")

    df=df.reset_index(drop=True)
    # df_0_row=df[df['zfy_label']==0]
    # df_1_row=df[df['zfy_label']==1]
    # len_df_0_sm=df_0_row.shape[0]
    # df_0_sm=df_0_row.drop_duplicates()
    # print("负类中完全一样的样本数：(删除)",len_df_0_sm-df_0_sm.shape[0])
    # df=pd.concat([df_0_sm,df_1_row],axis=0).reset_index(drop=True)
    print("NearestNeighbors------------------")
    pastt=time.time()
    neigh = NearestNeighbors(n_neighbors=4,n_jobs=-1)
    y=df['zfy_label']
    x=df.drop(columns='zfy_label', axis=1, inplace=False)
    columns_name.extend(['shortest_distance', 'hcp1_max'])
    x=df[columns_name]

    print(x.shape)
    neigh.fit(x)
    neighbor4_lsts=neigh.kneighbors(x, return_distance=False)   #样本x和三个最近邻的索引#<class 'numpy.ndarray'>
    print("NearestNeighbors-----------耗时：", (time.time() - pastt) / 60)
    pastt1=time.time()
    neighbor4_df=pd.DataFrame(neighbor4_lsts,columns=['ngb0_idx','ngb1_idx','ngb2_idx','ngb3_idx'])
    df_index=pd.DataFrame([i for i in range(neighbor4_df.shape[0])],columns=['x_idx'])
    neighbor4_df=pd.concat([df_index,neighbor4_df,y],axis=1)  #最近邻1，2，3的索引和x的标签

    neighbor4_df['ngb1_label']=neighbor4_df['ngb1_idx'].apply(lambda x:y.loc[x])
    neighbor4_df['ngb2_label'] = neighbor4_df['ngb2_idx'].apply(lambda x: y.loc[x])
    neighbor4_df['ngb3_label'] = neighbor4_df['ngb3_idx'].apply(lambda x: y.loc[x])

    # 若 x 为多数类 y=0
    neighbor4_df['sum_of_ngb_lbl']=neighbor4_df['ngb1_label']+neighbor4_df['ngb2_label']+neighbor4_df['ngb3_label']
    for index,df_group in neighbor4_df.groupby(['zfy_label']):   #index 的取值：1和0  1对应于高花费住院记录，2对应于非高花费住院记录
        if index==0:  # 多数类
            df_0_all=df_group
            print("未进行NCR时，负样本个数：",df_0_all.shape[0])
            # df_0=df_group[~(df_group['sum_of_ngb_lbl']>=2)]  #删除最近邻中有两个及以上少数类的多数类样本
            # print("NCR第一步删除的多数类样本个数：",df_group.shape[0]-df_0.shape[0])
        elif index == 1:  # 少数类
            df_1_all=df_group  #复制一份副本，存放所有少数类样本的index

    # df_0_all=df_0_all[~(df_0_all['sum_of_ngb_lbl']>=2)]

    df_1=df_1_all[df_1_all['sum_of_ngb_lbl'] <=1]   # x为少数类样本，选出最近邻中有两个及以上多数类的少数类样本

    lst_dlt=pd.concat([df_1['ngb1_idx'],df_1['ngb2_idx'],df_1['ngb3_idx']],axis=0).drop_duplicates().values # list: 所有要删除的多数类样本的index
    # list_1_index=list(df_1_all.index)
    list_1_index=df_1_all['x_idx'].tolist()   #所有少数类样本的索引 list
    lst_dlt2=[r for r in lst_dlt if r not in list_1_index ] #要删除的多数类的索引
    del lst_dlt
    # for i in tqdm(lst_dlt2):
    #     df_0_all=df_0_all[~(df_0_all['x_idx']==i)]

    df_0_all=df_0_all[(df_0_all['sum_of_ngb_lbl']>=2)]
    df_0_all=df_0_all['x_idx'].tolist()   #要删除的多数类的索引 list

    lst_dlt2.extend(df_0_all)
    set_dlt=set(lst_dlt2)
    lst_dlt2=list(set_dlt)
    df.drop(index=lst_dlt2,axis=0,inplace=True)
    df.reset_index(drop=True)
    # df=pd.concat([df,df_index],axis=1)
    # for i in tqdm(set_dlt):
    #     df = df[~(df['x_idx'] == i)]
    print("训练集样本个数：", df.shape[0])
    print("NCR后，去除负样本个数：",len(lst_dlt2))
    # print("正样本个数：",df_1_all)
    # df=pd.concat([df_0_all['x_idx'],df_1_all['x_idx']],axis=0)
    # print()
    # df_0_all=df_0_all[df_0_all['x_idx'] not in lst_dlt2]
    # # df_0=df_group[~(df_group['sum_of_ngb_lbl']>=2)]  #删除最近邻中有两个及以上少数类的多数类样本
    # # print("NCR第一步删除的多数类样本个数：",df_group.shape[0]-df_0.shape[0])
    # for i in lst_dlt:
    #     if i in

    print("Neighbor Clean Rule-----------耗时：",(time.time()-pastt1)/60)

    f0 = open(savepath, 'wb')
    pickle.dump(df, f0)
    f0.close()

    f0 = open("data/ncr/delete_rows_ncr.pkl", 'wb')
    pickle.dump(lst_dlt2, f0)
    f0.close()

    return df


def construct_network_graph(type,percentile,dic_disease_prevalence_rate,edge_list,write_file,list_modularity):
    """功能：构网构图
    输入参数：_percentile 分位数，只画出相关系数在percentile以上的边
    输入参数：_type: type='RR': RR；type='phi':phi；type='CC':CC
    输入参数：_dic_disease_prevalence_rate  流行率字典
    输入参数：_edge_list 边集，应和_type类型对应
    输入参数：_write_file:保存gml图的路径文件名
    输入参数：_list_modularity:每个节点所属社区

    备注：Community_Detection_v2中函数build_Graph()调用了该函数
    """
    if type == 'RR':
        # edge_list_RR的结构：[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间, 同时患两种疾病的人数], ....]

        print(len(edge_list))
        node_set=set()

        quantile_value=pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_RR=[i for i in edge_list if i[4]>=quantile_value]

        for edge in edge_list_RR:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list=sorted(list(node_set))  # 节点名称，排序后

        prevalence_rate=[dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g=Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name']=node_name_list
        g.vs['label']=node_name_list
        g.vs['prevalence']=prevalence_rate
        if list_modularity:
            g.vs['modularity_class'] = list_modularity
        g.add_edges((edge[0],edge[1]) for edge in edge_list_RR)
        RR_list=[0 for j in range(len(edge_list_RR))]
        CI_high=[0 for j in range(len(edge_list_RR))]
        CI_low=[0 for j in range(len(edge_list_RR))]
        for edge in edge_list_RR:
            edge_id=g.get_eid(edge[0],edge[1])
            RR_list[edge_id]=edge[4]
            CI_high[edge_id] = edge[6]
            CI_low[edge_id] = edge[5]
        g.es['weight'] = RR_list
        g.es['RR_CI_high']=CI_high
        g.es['RR_CI_low']=CI_low
        print(summary(g))
        g.write(write_file,"gml")
        # plot(g)

    if type=='phi':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数], ....]
        print(len(edge_list))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_phi = [i for i in edge_list if i[4] >= quantile_value]

        for edge in edge_list_phi:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率
        print(node_name_list)
        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        if list_modularity:
            g.vs['modularity_class'] = list_modularity
        g.add_edges((edge[0], edge[1]) for edge in edge_list_phi)
        phi_list = [0 for j in range(len(edge_list_phi))]
        for edge in edge_list_phi:
            edge_id = g.get_eid(edge[0], edge[1])
            phi_list[edge_id] = edge[4]
        g.es['weight'] = phi_list
        print(summary(g))
        g.write(write_file, "gml")
        # plot(g)

    if type=='CC':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的CC值，t值，无意义位，同时患两种病的人数], ....]
        print(len(edge_list))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_CC = [i for i in edge_list if i[4] >= quantile_value]

        for edge in edge_list_CC:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        if list_modularity:
            g.vs['modularity_class'] = list_modularity
        g.add_edges((edge[0], edge[1]) for edge in edge_list_CC)
        CC_list = [0 for j in range(len(edge_list_CC))]
        for edge in edge_list_CC:
            edge_id = g.get_eid(edge[0], edge[1])
            CC_list[edge_id] = edge[4]
        g.es['weight'] = CC_list
        print(summary(g))
        g.write(write_file, "gml")


def Compute_EC(gml_path):
    """
    计算各个节点在所属社区中的特征向量中心度
    _gml_path:读入图文件的路径
    返回值：_dic_all_nodes_EC 字典，{'M06': 0.030169044514624266,...}=> 整个网络中所有节点的特征向量中心度字典
    返回值:dic_module:节点-社区编号字典
    返回值：numOfModule：模块个数
    """
    #读入模块化后的网络
    mygraph = nx.read_gml(gml_path)
    # print(len(mygraph.nodes)) #93
    # print(len(mygraph.edges)) #1938
    # print(mygraph.nodes[0]['modularityclass'])

    # 获取社区个数;获取疾病：所属模块字典dic_module
    module_name=set()
    dic_module = {}
    for i in mygraph.nodes:
        module_name.add(mygraph.nodes[i]['modularityclass'])
        dic_module[i] = mygraph.nodes[i]['modularityclass']
    # print(module_name)
    numOfModule=len(module_name)

    k=-1
    dic_all_nodes_EC={}
    for module_id in range(numOfModule):
        # 社区单独成网
        module0=[]
        for i in mygraph.nodes:
            if mygraph.nodes[i]['modularityclass']==module_id:
                # print(mygraph.nodes[i]['modularityclass'])
                module0.append(mygraph.nodes[i]['name'])

        # 子社区
        community0=nx.subgraph(mygraph,module0)
        # print(len(community0.nodes))
        # print(len(community0.edges))

        # dic_node=dict(zip(community0.nodes,[2 for i in range(len(community0.nodes))]))
        # print(dic_node)
        # 计算特征向量中心度
        # print(mygraph.edges[('D50', 'D64')]['weight'])
        # dic_1=nx.eigenvector_centrality_numpy(community0, weight=None, max_iter=1000, tol=1e-6)
        dic_ec=nx.eigenvector_centrality(community0, max_iter=10000, tol=1e-6, nstart=None, weight='weight')
        k+=1
        # print("社区",k,"中节点的特征向量中心度：",dic_ec)
        dic_all_nodes_EC.update(dic_ec)
    return dic_all_nodes_EC,numOfModule,dic_module


def Compute_Centrality(gml_path):
    """
    输入参数：_gml_path:读入图文件的路径
    计算中心性指标,均为归一化后的值：
    介数中心度
    接近中心度
    度中心度
    强度中心度（strength degree）
    返回值：均为字典
    """
    mygraph = nx.read_gml(gml_path)
    print(len(mygraph.nodes))  # 93
    print(len(mygraph.edges))  # 1938
    betweenness_centrality=nx.betweenness_centrality(mygraph, k=None, normalized=True, weight='weight', endpoints=False, seed=None)
    closeness_centrality=nx.closeness_centrality(mygraph, u=None, distance='weight', wf_improved=True)
    degree_centrality=nx.degree_centrality(mygraph)

    # 计算strength_centrality
    # print(mygraph.edges)  # [('D50', 'D64'), ('D50', 'D70'),....
    dic_strength_centrality= dict(zip(mygraph.nodes, [0 for i in range(len(mygraph.nodes))]))  #初始化
    max_weight =0
    for i in mygraph.edges:
        weight=mygraph.edges[i]['weight']
        dic_strength_centrality[i[0]]+=weight
        dic_strength_centrality[i[1]] += weight
        if weight>max_weight:
            max_weight=weight
    #归一化
    for sc in dic_strength_centrality:
        dic_strength_centrality[sc]/=(max_weight*(len(dic_strength_centrality)-1))

    # print(dic_node_SC)
    # print(degree_centrality)
    return betweenness_centrality,closeness_centrality,degree_centrality,dic_strength_centrality

def get_dic_module(gml_path):
    mygraph = nx.read_gml(gml_path)
    dic_module={}
    numOfmodule=set()
    for i in mygraph.nodes:
        dic_module[i]=mygraph.nodes[i]['modularityclass']
        numOfmodule.add(mygraph.nodes[i]['modularityclass'])
    return dic_module

def compute_OMIGA(df,gml_path,prevalence_path):
    """
    _df含至少两列：cy_diseases_和zfy_label
                    cy_diseases_包含的是出院诊断，且每条记录的出院诊断不存在重复的疾病（即为set()）
    _path ：icd3字典读取路径
    _gml_path：图文件读取路径
    _cost_quantile95：19年数据的95临界点
    计算Ω,所有疾病都可以计算Ω值
    计算HCP值
    """
    # 初始化两个字典，分别存储疾病出现在高花费记录中的次数，和疾病出现的所有次数
    dic_high_cost_num={}
    dic_non_high_cost_num={}

    print("---------------开始计算Omiga值：-----------------")
    pastt0=time.time()
    for index,df_diseases in df.groupby(['zfy_label']):   #index 的取值：1和0  1对应于高花费住院记录，2对应于非高花费住院记录
        if index==0:
            for index_, disease in enumerate(df_diseases["cy_diseases_"]):  #df_diseases["diseases_"]中每行记录中的疾病无重复,对应于出院诊断
                for d in disease:
                    if d in dic_non_high_cost_num:
                        dic_non_high_cost_num[d]+=1
                    else:
                        dic_non_high_cost_num[d]=1
                        # dic_high_cost_num[d]=0
        elif index==1:   # 高花费记录
            for index_,disease in enumerate(df_diseases["cy_diseases_"]):
                for d in disease:
                    if d in dic_high_cost_num:
                        dic_high_cost_num[d]+=1
                    else:
                        dic_high_cost_num[d]=1
    #计算Ω
    dic_omiga={}  # 只有Ω值不为0的疾病且在训练集中出现的疾病才会出现在该字典中
    for icd_code in dic_high_cost_num:
        high_appear_counts=dic_high_cost_num[icd_code]

        if icd_code in dic_non_high_cost_num:
            total_appear_counts=high_appear_counts+dic_non_high_cost_num[icd_code]
        # if total_appear_counts!=0 and high_appear_counts!=0:
            omiga=high_appear_counts/total_appear_counts
            dic_omiga[icd_code]=omiga    # Ω字典，包含训练集中出现的icd前三位编码和对应的Ω值（高花费概率）
        else:
            dic_omiga[icd_code]=1
    print("----------------计算Omiga值耗时：---", (time.time() - pastt0) / 60)
    print("---------------开始计算HCP值：-----------------")
    pastt1=time.time()
    # 计算HCP
    # 在网络中的节点都有HCP值，不在网络中的节点HCP值等于Ω值
    # gml_path = "data/real_datas/includeIHD/CC_all_modularity.gml"
    mygraph = nx.read_gml(gml_path)
    # 初始化hcp
    hcp = dict(zip(mygraph.nodes, [0 for i in range(len(mygraph.nodes))]))
    hcp1 = dict(zip(mygraph.nodes, [0 for i in range(len(mygraph.nodes))]))

    # 计算边权之和
    dic_strength_centrality = dict(zip(mygraph.nodes, [0 for i in range(len(mygraph.nodes))]))  # 初始化
    for i in mygraph.edges:
        weight = mygraph.edges[i]['weight']
        dic_strength_centrality[i[0]] += weight
        dic_strength_centrality[i[1]] += weight

    for i in mygraph.edges:
        weight = mygraph.edges[i]['weight']
        if i[1] in dic_omiga:
            hcp[i[0]] += weight * dic_omiga[i[1]]
            hcp1[i[0]] += weight * dic_omiga[i[1]]
        if i[0] in dic_omiga:
            hcp[i[1]] += weight * dic_omiga[i[0]]
            hcp1[i[1]] += weight * dic_omiga[i[0]]
    cc_piao = hcp.copy()  # 可能存在零值
    for i in hcp1:
        hcp1[i] /= dic_strength_centrality[i]
        if i in dic_omiga:
            hcp1[i] += dic_omiga[i]
    for i in hcp:
        if i in dic_omiga:
            hcp[i] += dic_omiga[i]

    hcp_only = hcp.copy()  # 只包含网络中出现的节点

    # 观察hcp和流行率之间的关系
    x_coef, intrcpt, prevalence_dic = hcp_prevalence(prevalence_path, cc_piao)


    for i in dic_omiga:
        if i not in hcp:
            hcp[i] = dic_omiga[i]
            hcp1[i] = dic_omiga[i]

    print("----------------计算HCP值耗时：---", (time.time() - pastt1) / 60)
    return dic_omiga, hcp, hcp_only, hcp1  # 返回三个字典；len(dic_omiga)=Ω不为0的疾病个数；len(hcp_only):共病网络中的节点个数


def find_shortest_path(gml_path):
    """
    计算各个疾病节点到高花费节点的最短路径，
    #添加最短路径特征，若一条记录中所有节点都没有最短路径，则取max_dist作为该条记录的属性值；若有节点有最短路径，则取最短的最短路径作为属性值
    返回字典dic_dist:  e.g.{'I50': 0, 'E30': 1, 'F23': 3, ......},字典中记录的是各个疾病节点到高花费节点的路径长度
    """
    # gml_path="data/gml_dir/distance_OR_Graph_all.gml"
    print("-----------开始计算最短路径------------------------")
    pastt=time.time()
    G = nx.read_gml(gml_path)
    pred, dic_dist = nx.dijkstra_predecessor_and_distance(G, 'High Cost', cutoff=None, weight='weight')
    del dic_dist['High Cost']
    print("-----------计算最短路径耗时：------------------------",(time.time()-pastt)/60)
    print("到高花费节点有最短路径的节点数：",len(dic_dist))
    list_dist = sorted(dic_dist.items(), key=lambda kv: (kv[1]), reverse=True)
    max_dist=list_dist[0][1]
    min_dist=list_dist[len(list_dist)-1][1]
    print("最长的最短路径：",max_dist)
    print("最短的最短路径：",min_dist)

    return dic_dist,max_dist,min_dist


def compute_OR_comobidity_cost(df_X_train_raw):
    """
    计算OR,用训练集的OR计算
    输入参数：_df含至少两列：cy_diseases_和zfy_label
                    cy_diseases_包含的是出院诊断，且每条记录的出院诊断不存在重复的疾病（即为set()）
    """
    dic_non_high_cost_num = {}  # b
    dic_high_cost_num = {}  # a
    for index, df_diseases in df_X_train_raw.groupby(['zfy_label']):  # index 的取值：1和0  1对应于高花费住院记录，2对应于非高花费住院记录
        if index == 0:
            len_non_high_rcd_count = df_diseases.shape[0]
            print("len_non_high_rcd_count", df_diseases.shape[0])
            for index_, disease in enumerate(tqdm(df_diseases["cy_diseases_"])):  # df_diseases["diseases_"]中每行记录中的疾病无重复,对应于出院诊断
                for d1 in disease:
                    for d2 in disease:
                        if d1>d2:
                            disease_pair=(d2,d1)
                        elif d1<d2:
                            disease_pair=(d1,d2)
                        else:
                            disease_pair=-1
                        if disease_pair!=-1:
                            if disease_pair in dic_non_high_cost_num:
                                dic_non_high_cost_num[disease_pair] += 1
                            else:
                                dic_non_high_cost_num[disease_pair] = 1
                                # dic_high_cost_num[d]=0
        elif index == 1:  # 高花费记录
            len_high_rcd_count = df_diseases.shape[0]
            print("len_high_rcd_count", df_diseases.shape[0])
            for index_, disease in enumerate(tqdm(df_diseases["cy_diseases_"])):
                for d1 in disease:
                    for d2 in disease:
                        if d1>d2:
                            disease_pair=(d2,d1)
                        elif d1<d2:
                            disease_pair=(d1,d2)
                        else:
                            disease_pair=-1
                        if disease_pair!=-1:
                            if disease_pair in dic_high_cost_num:
                                dic_high_cost_num[disease_pair] += 1
                            else:
                                dic_high_cost_num[disease_pair] = 1
    for elem in dic_non_high_cost_num:
        dic_non_high_cost_num[elem]/=2
    for elem in dic_high_cost_num:
        dic_high_cost_num[elem]/=2


    set_dise1 = set(dic_high_cost_num.keys())
    set_dise2 = set(dic_non_high_cost_num.keys())
    set_dise = set_dise1 | set_dise2
    # set_dise=set_dise&diseases_set
    print("len(set_dise)",len(set_dise))
    # len_rcd = df.shape[0]  #1185
    dic_OR_greater_1 = {}
    dic_OR_less_1={}
    list_edges_OR=[]
    for i in set_dise:
        if i in dic_high_cost_num:
            a = dic_high_cost_num[i]
        else:
            a = 0
        if i in dic_non_high_cost_num:
            b = dic_non_high_cost_num[i]
        else:
            b = 0
        c = len_high_rcd_count - a
        d = len_non_high_rcd_count - b
        if c != 0 and d != 0 and a != 0 and b != 0:
            OR = (a / c) / (b / d)
            se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            ci_high = np.exp(np.log(OR) + 1.96 * se)
            ci_low = np.exp(np.log(OR) - 1.96 * se)
            if ci_low > 1 and OR > 1:
                # list_edge_OR=["High Cost",i,OR,ci_low,ci_high]
                # list_edges_OR.append(list_edge_OR)
                dic_OR_greater_1[i] = [OR, ci_low, ci_high]
            if ci_high < 1 and OR < 1:
                dic_OR_less_1[i] = [OR, ci_low, ci_high]

    # print("len(dic_OR)",len(dic_OR))  #186
    # list_all_edges_OR.extend(list_edges_OR)
    # f0 = open(savepath, 'wb')
    # pickle.dump(dic_OR, f0)
    # f0.close()
    return dic_OR_greater_1,dic_OR_less_1


def network_feature_OR_comobidity_cost(df, dic_OR_greater_1,dic_OR_less_1):
    """处理网络特征
         _df :特征df"""
    print("---------------开始计算网络特征：-----------------")
    pastt = time.time()
    num_greatest_1_list = []
    num_less_1_list = []

    for index, i in enumerate(tqdm(df["ry_diseases_"])):
        num_greatest_than_1=0
        num_less_than_1=0

        for j1 in i:
            for j2 in i:
                if j1==j2:
                    disease_pair=-1
                elif j1<j2:
                    disease_pair=(j1,j2)
                else:
                    disease_pair=(j2,j1)
                if disease_pair!=-1:
                    if disease_pair in dic_OR_greater_1:
                        num_greatest_than_1+=1
                    elif disease_pair in dic_OR_less_1:
                        num_less_than_1+=1
        num_greatest_than_1/=2
        num_less_than_1/=2

        num_greatest_1_list.append(num_greatest_than_1)
        num_less_1_list.append(num_less_than_1)

    df_num_greatest_1 = pd.DataFrame(num_greatest_1_list, columns=['comobidity_num_OR_G1'])
    df_num_less_1 = pd.DataFrame(num_less_1_list, columns=['comobidity_num_OR_L1'])
    df = pd.concat( [df, df_num_greatest_1,df_num_less_1],axis=1)

    print("----------------计算网络特征耗时：---", (time.time() - pastt) / 60)
    return df


def network_feature_shortest_dist(df, dic_shortest_distance,max_shortest_dist):
    """处理网络特征
         _df :特征df"""
    print("---------------开始计算网络特征：-----------------")
    pastt = time.time()
    shortest_dist_list = []

    for index, i in enumerate(df["ry_diseases_"]):
        min_shortest_dist=max_shortest_dist

        for j in i:
            if j in dic_shortest_distance:
                if dic_shortest_distance[j] < min_shortest_dist:
                    min_shortest_dist = dic_shortest_distance[j]

        shortest_dist_list.append(min_shortest_dist)
    df_shortest_dist = pd.DataFrame(shortest_dist_list, columns=['shortest_distance'])

    df = pd.concat( [df, df_shortest_dist],axis=1)

    print("----------------计算网络特征耗时：---", (time.time() - pastt) / 60)
    return df



def chapter_name_list(xlsxpath):
    """
    _xlsxpath=data/csv_xslx/AA_MULTI_DESC_TENSOR.xlsx
    返回值：dic_icd3_dchapter：{‘I25'：章-编码, ......}
    返回值：dic_clmnname_index：{章-编码：index, ......}
    """
    print("--------------------开始读章-编码文件--------")
    # pastt0 = time.time()
    df_chapter = pd.read_excel(xlsxpath, sheet_name='疾病分类编码')
    lst_dgroup=df_chapter['章-编码'].tolist()
    lst_d3icd = df_chapter['类目-编码'].tolist()
    dic_icd3_dchapter=dict(zip(lst_d3icd,lst_dgroup))
    columns_name=df_chapter['章-编码'].drop_duplicates().tolist()
    dic_clmnname_index=dict(zip(columns_name,[i for i in range(len(columns_name))]))
    print("章的数量：",len(columns_name))
    pastt = time.time()
    return dic_icd3_dchapter,dic_clmnname_index


def chapter_dic(path,df):
    """_df: 特征df"""
    print("--------------------开始读章-编码文件--------")
    pastt0=time.time()
    df_chapter=pd.read_excel(path,sheet_name='I系统22')
    # print(df_chapter['章-编码'])
    list_chapter=df_chapter['章-编码'].apply(lambda x:[x[0:3],x[4:7]]).tolist()
    print("----------------读章-编码文件耗时：---", (time.time() - pastt0) / 60)
    # print(list_chapter)

    # df_cpt_onehot=pd.DataFrame(np.zeros().reshape((df.shape[0], len(list_chapter))),
    #              columns=['ICD_chapter_'+str(i) for i in len(list_chapter)])  #初始化
    print("--------------------开始处理ICD10章编码指标--------")
    pastt=time.time()
    onehot_lists=[]
    for index, i in enumerate(tqdm(df["ry_diseases_"])):
        zero_list=[0 for ind in range(len(list_chapter))]
        for j in range(len(i)):
            if i[j]<=list_chapter[0][1]:
                zero_list[0]=1
            elif i[j]<=list_chapter[1][1]:
                zero_list[1] = 1
            elif i[j]<=list_chapter[2][1]:
                zero_list[2] = 1
            elif i[j]<=list_chapter[3][1]:
                zero_list[3] = 1
            elif i[j]<=list_chapter[4][1]:
                zero_list[4] = 1
            elif i[j]<=list_chapter[5][1]:
                zero_list[5] = 1
            elif i[j]<=list_chapter[6][1]:
                zero_list[6] = 1
            elif i[j]<=list_chapter[7][1]:
                zero_list[7] = 1
            elif i[j]<=list_chapter[8][1]:
                zero_list[8] = 1
            elif i[j]<=list_chapter[9][1]:
                zero_list[9] = 1
            elif i[j]<=list_chapter[10][1]:
                zero_list[10] = 1
            elif i[j]<=list_chapter[11][1]:
                zero_list[11] = 1
            elif i[j]<=list_chapter[12][1]:
                zero_list[12] = 1
            elif i[j]<=list_chapter[13][1]:
                zero_list[13] = 1
            elif i[j]<=list_chapter[14][1]:
                zero_list[14] = 1
            elif i[j]<=list_chapter[15][1]:
                zero_list[15] = 1
            elif i[j]<=list_chapter[16][1]:
                zero_list[16] = 1
            elif i[j]<=list_chapter[17][1]:
                zero_list[17] = 1
            elif i[j]<=list_chapter[18][1]:
                zero_list[18] = 1
            elif i[j]<=list_chapter[19][1]:
                zero_list[19] = 1
            elif i[j]<=list_chapter[20][1]:
                zero_list[20]=1
            elif i[j]<=list_chapter[21][1]:
                zero_list[21] = 1
        onehot_lists.append(zero_list)
    df_cpt_onehot=pd.DataFrame(onehot_lists, columns=['ICD_chapter_'+str(i) for i in range(len(list_chapter))])
    del onehot_lists
    print("----------------处理章编码onehot变量耗时：---",(time.time()-pastt)/60)

    df=pd.concat([df,df_cpt_onehot],axis=1)
    # print(df_cpt_onehot)
    return df


def disease_group220_dic(xlsxpath,df):

    # AA_MULTI_DESC_TENSOR.xlsx
    print("--------------------开始读疾病组-编码文件--------")
    # pastt0 = time.time()
    df_chapter = pd.read_excel(xlsxpath, sheet_name='疾病分类编码')
    # print(df_chapter['章-编码'])
    lst_dgroup=df_chapter['疾病组-编码'].tolist()
    lst_d3icd = df_chapter['类目-编码'].tolist()
    dic_icd3_dgroup=dict(zip(lst_d3icd,lst_dgroup))
    columns_name=df_chapter['疾病组-编码'].drop_duplicates().tolist()
    dic_clmnname_index=dict(zip(columns_name,[i for i in range(len(columns_name))]))
    print("疾病组的数量：",len(columns_name))
    pastt = time.time()
    onehot_lists = []
    for index, i in enumerate(df["ry_diseases_"]):
        zero_list = [0 for ind in range(len(columns_name))]
        for j in i:
            if j in dic_icd3_dgroup:
                zero_list[dic_clmnname_index[dic_icd3_dgroup[j]]]=1
        onehot_lists.append(zero_list)
    df_cpt_onehot = pd.DataFrame(onehot_lists, columns=columns_name)
    del onehot_lists
    print("----------------处理疾病组编码onehot变量耗时：---", (time.time() - pastt) / 60)
    df_cpt_onehot.loc['Row_sum'] = df_cpt_onehot.apply(lambda x: x.sum())
    df_index=df_cpt_onehot.loc['Row_sum']
    df_index = df_index[df_index>0]
    df_cpt_onehot = df_cpt_onehot[list(df_index.index)]
    df_cpt_onehot.drop(index='Row_sum',axis=0,inplace=True)
    columns_name = list(df_cpt_onehot.columns)
    print("疾病组的个数：",df_cpt_onehot.shape[1])
    df = pd.concat([df, df_cpt_onehot], axis=1)
    # print(list(df.index))
    # print(list(df.columns))
    # print(df_cpt_onehot)
    # print(df.head(10).values.tolist())
    return df,columns_name


def LDA_DR(x_train_fold,y_train_fold,x_test_fold,columns_name,n_dimension):
    """对疾病组进行降维
    columns_name_:220组疾病组的名字list
    n_dimension_:要降到的维度
    """
    # y_train_220=x_train_fold['ZFY']     #分区？
    x_train_220=x_train_fold[columns_name]
    x_test_220=x_test_fold[columns_name]
    print(x_train_220.shape)
    print(n_dimension)
    lda = LinearDiscriminantAnalysis(n_components=n_dimension)
    lda.fit(x_train_220, y_train_fold)
    x_train_220=lda.transform(x_train_220)
    x_test_220=lda.transform(x_test_220)   # 存在nan
    columns_name2=['LDA_'+str(i) for i in range(n_dimension)]
    df_x_train_220=pd.DataFrame(x_train_220,columns=columns_name2)
    df_x_test_220 = pd.DataFrame(x_test_220, columns=columns_name2)
    x_train_fold.drop(columns=columns_name,axis=1,inplace=True)
    x_test_fold.drop(columns=columns_name,axis=1,inplace=True)
    x_train_fold=pd.concat([x_train_fold,df_x_train_220],axis=1)
    x_test_fold = pd.concat([x_test_fold, df_x_test_220], axis=1)

    return x_train_fold,x_test_fold,columns_name2

def PCA_DR(x_train_fold,y_train_fold,x_test_fold,columns_name,n_dimension):
    """对疾病组进行降维
    columns_name_:220组疾病组的名字list
    n_dimension_:要降到的维度
    """
    # y_train_220=x_train_fold['ZFY']
    x_train_220=x_train_fold[columns_name]
    x_test_220=x_test_fold[columns_name]
    print(x_train_220.shape)
    print(n_dimension)
    lda = PCA(n_components=n_dimension)
    lda.fit(x_train_220)
    x_train_220=lda.transform(x_train_220)
    x_test_220=lda.transform(x_test_220)
    columns_name2=['PCA_'+str(i) for i in range(n_dimension)]
    df_x_train_220=pd.DataFrame(x_train_220,columns=columns_name2)
    df_x_test_220 = pd.DataFrame(x_test_220, columns=columns_name2)
    x_train_fold.drop(columns=columns_name,axis=1,inplace=True)
    x_test_fold.drop(columns=columns_name,axis=1,inplace=True)
    x_train_fold=pd.concat([x_train_fold,df_x_train_220],axis=1)
    x_test_fold = pd.concat([x_test_fold, df_x_test_220], axis=1)

    return x_train_fold,x_test_fold,columns_name2


def my_random_undersample(x_train_df):
    """
    从训练集对多数类进行简单的随机下采样，返回一个多数类和少数类数量相同的训练集，用于初调参数
    """
    x_train_pos = x_train_df[x_train_df['zfy_label'] == 1]
    x_train_neg = x_train_df[x_train_df['zfy_label'] == 0]
    x_train_neg_posnum = x_train_neg.sample(n=x_train_pos.shape[0], replace=False, random_state=42, axis=0)
    x_train = pd.concat([x_train_pos, x_train_neg_posnum], axis=0).reset_index(drop=True)
    y_train = x_train['zfy_label']
    x_train.drop(columns='zfy_label', axis=1, inplace=True)
    return x_train,y_train

def my_random_undersample2(x_train_df):
    """
    从训练集对多数类进行简单的随机下采样，返回一个多数类和少数类数量相同的训练集，用于初调参数
    """
    x_train_pos = x_train_df[x_train_df['zfy_label'] == 1]
    x_train_neg = x_train_df[x_train_df['zfy_label'] == 0]
    x_train_neg_posnum = x_train_neg.sample(n=x_train_pos.shape[0], replace=False, random_state=42, axis=0)
    x_train = pd.concat([x_train_pos, x_train_neg_posnum], axis=0).reset_index(drop=True)
    return x_train

def compute_OR(df):
    """
    计算OR,用训练集的OR计算
    输入参数：_df含至少两列：cy_diseases_和zfy_label
                    cy_diseases_包含的是出院诊断，且每条记录的出院诊断不存在重复的疾病（即为set()）
    """

    dic_non_high_cost_num = {}  # b
    dic_high_cost_num = {}  # a
    for index, df_diseases in df.groupby(['zfy_label']):  # index 的取值：1和0  1对应于高花费住院记录，2对应于非高花费住院记录
        if index == 0:
            len_non_high_rcd_count = df_diseases.shape[0]
            print("len_non_high_rcd_count", df_diseases.shape[0])
            for index_, disease in enumerate(df_diseases["cy_diseases_"]):  # df_diseases["diseases_"]中每行记录中的疾病无重复,对应于出院诊断
                for d in disease:
                    if d in dic_non_high_cost_num:
                        dic_non_high_cost_num[d] += 1
                    else:
                        dic_non_high_cost_num[d] = 1
                        # dic_high_cost_num[d]=0
        elif index == 1:  # 高花费记录
            len_high_rcd_count = df_diseases.shape[0]
            print("len_high_rcd_count", df_diseases.shape[0])
            for index_, disease in enumerate(df_diseases["cy_diseases_"]):
                for d in disease:
                    if d in dic_high_cost_num:
                        dic_high_cost_num[d] += 1
                    else:
                        dic_high_cost_num[d] = 1

    set_dise1 = set(dic_high_cost_num.keys())
    set_dise2 = set(dic_non_high_cost_num.keys())
    set_dise = set_dise1 | set_dise2
    print("len(set_dise)",len(set_dise))
    # len_rcd = df.shape[0]  #1185
    dic_OR = {}
    for i in set_dise:
        if i in dic_high_cost_num:
            a = dic_high_cost_num[i]
        else:
            a = 0
        if i in dic_non_high_cost_num:
            b = dic_non_high_cost_num[i]
        else:
            b = 0
        c = len_high_rcd_count - a
        d = len_non_high_rcd_count - b
        if c != 0 and d != 0 and a != 0 and b != 0:
            OR = (a / c) / (b / d)
            se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            ci_high = np.exp(np.log(OR) + 1.96 * se)
            ci_low = np.exp(np.log(OR) - 1.96 * se)
            if ci_low > 1 and OR > 1:
                dic_OR[i] = [OR, ci_low, ci_high]
            # if ci_high < 1 and OR < 1:
            #     dic_OR[i] = [OR, ci_low, ci_high]
    print("len(dic_OR)",len(dic_OR))  #186
    # f0 = open(savepath, 'wb')
    # pickle.dump(dic_OR, f0)
    # f0.close()
    return dic_OR


def write_csv():
    f0 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry.pkl", 'rb')
    df=pickle.load(f0)
    f0.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry.csv")

# gml_path = "../data/real_datas/includeIHD/CC_all_modularity.gml"
# Compute_Centrality(gml_path)

# read_path="../data/feature_related/df_del_R_wthout_IHDry.pkl"
# savepath="../data/feature_related/OR_dic.pkl"
# compute_OR(savepath, read_path)

def kmeans_cost():
    """用k-means算法对cost进行聚类分析"""
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry6_clearned.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    df=df['ZFY']
    quantile95=df.quantile(0.95)
    max_cost=df.max()
    min_cost=df.min()
    print("住院费用95分位值：",quantile95)   #住院费用95分位值： 40747.12
    x=df.values.reshape(-1, 1)
    # kmeans = KMeans(n_init=100,tol=0.0000000000000001,max_iter=100000000,n_clusters=2, random_state=42).fit(x)
    # kmeans = KMeans( tol=0.000001, max_iter=1000,
    #                 n_clusters=10, random_state=42).fit(x)
    kmeans = Birch(n_clusters=2)
    kmeans.fit(x)
    # print("kmeans.cluster_centers_", kmeans.cluster_centers_)  #kmeans.cluster_centers_ [[ 9340.34108027] [48158.51908543]]
    y=kmeans.predict(x)
    df=pd.concat([df,pd.DataFrame(y,columns=['kmeans_label'])],axis=1)
    # df['ZFY']=df['ZFY'].apply(lambda x:np.log(x))
    for index, df_idnumber in df.groupby(['kmeans_label']):
        case_lyfs = df_idnumber['kmeans_label'].values[0]
        sns.distplot((df_idnumber['ZFY']), label=case_lyfs,hist=False)
        print("num of code"+str(case_lyfs)+":  ",df_idnumber.shape[0])
        print("mean" + str(case_lyfs) + ":  ", df_idnumber['ZFY'].mean())
        print("max" + str(case_lyfs) + ":  ", df_idnumber['ZFY'].max())
        print("min" + str(case_lyfs) + ":  ", df_idnumber['ZFY'].min())
    plt.legend()
    # plt.xlim(0,200000)
    plt.show()
    # plt.scatter(df.index,df['ZFY'], c=df['kmeans_label'])
    # plt.show()

def process_last_time_cost_per_day():
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    df1=df[df['los_last_time']==-999]
    df2 = df[df['los_last_time'] != -999]
    df1['last_time_cost_per_day']=-999
    df=pd.concat([df1,df2],axis=0)
    df.sort_index(inplace=True)

    f0 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.csv")


def process_my_fault():
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.csv")


def get_network_ftr_comorbidity_attri(dir_network_mtx):
    """
    :param dir_network_mtx:   e.g. data_wo_threshold;  网络矩阵字典所在文件夹
    :return:
    """
    f2 = open(dir_network_mtx+"/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'rb')
    dic_cmbdty_attribution = pickle.load(f2)
    f2.close()

    f2 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    # dic_omiga, hcp, hcp_only, hcp1=compute_OMIGA(df_cydis_label, gml_path, prevalence_path)

    lst_cmbdty_attribution=[]
    lst_sum_attri=[]
    for index, i in enumerate(tqdm(df["ry_diseases_"])):
        lst_diseases_encoding=[[0 for i in range(len(dic_cmbdty_attribution))]]
        sum_attri=0
        for j in i:
            if j in dic_cmbdty_attribution:
                lst_diseases_encoding.append(dic_cmbdty_attribution[j])
        df_attri=pd.DataFrame(lst_diseases_encoding,columns=list(dic_cmbdty_attribution.keys()))
        max_attri_lst=df_attri.max().values.tolist()

        lst_cmbdty_attribution.append(max_attri_lst)

        for max_attri in max_attri_lst:
            sum_attri+=max_attri

        lst_sum_attri.append(sum_attri)
    # 将lst_cmbdty_attribution和lst_sum_attri存到df中
    # df["max_attri_vector"]
    df=pd.concat([df,pd.DataFrame(lst_cmbdty_attribution,columns=['max_attri_vector']),pd.DataFrame(lst_sum_attri,columns=["cmbdty_attri_sum"])],axis=1)

    f0 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_Histry9_clearned2.csv")


def get_network_ftr_comorbidity_attri_multi_p(df,flag):
    """
    :param dir_network_mtx:   e.g. data_wo_threshold;  网络矩阵字典所在文件夹
    :return:
    """
    f2 = open("../data_wo_threshold/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'rb')
    dic_cmbdty_attribution = pickle.load(f2)
    f2.close()

    # dic_omiga, hcp, hcp_only, hcp1=compute_OMIGA(df_cydis_label, gml_path, prevalence_path)

    lst_cmbdty_attribution=[]
    lst_sum_attri=[]
    for index, i in enumerate(tqdm(df["ry_diseases_"])):
        lst_diseases_encoding=[[0 for i in range(len(dic_cmbdty_attribution))]]
        sum_attri=0
        for j in i:
            if j in dic_cmbdty_attribution:
                lst_diseases_encoding.append(dic_cmbdty_attribution[j])
        df_attri=pd.DataFrame(lst_diseases_encoding,columns=list(dic_cmbdty_attribution.keys()))
        max_attri_lst=df_attri.max().values.tolist()

        lst_cmbdty_attribution.append(max_attri_lst)

        for max_attri in max_attri_lst:
            sum_attri+=max_attri

        lst_sum_attri.append(sum_attri)
    # 将lst_cmbdty_attribution和lst_sum_attri存到df中
    # df["max_attri_vector"]
    f0 = open("../data/feature_related/lst_cmbdty_attribution_" + flag + ".pkl", 'wb')
    pickle.dump(lst_cmbdty_attribution, f0)
    f0.close()

    #
    # df=pd.concat([df,pd.DataFrame(lst_cmbdty_attribution,columns=list(dic_cmbdty_attribution.keys())),pd.DataFrame(lst_sum_attri,columns=["cmbdty_attri_sum"])],axis=1)
    #
    # f0 = open("../data/feature_related/df_Histry9_clearned2_"+flag+".pkl", 'wb')
    # pickle.dump(df, f0)
    # f0.close()
    # df.to_csv("../data/feature_related/df_Histry9_clearned2_"+flag+".csv")


def concat_df():
    f2 = open("../data/feature_related/df_Histry9_clearned2_1.pkl", 'rb')
    df1 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/df_Histry9_clearned2_2.pkl", 'rb')
    df2 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/df_Histry9_clearned2_3.pkl", 'rb')
    df3 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/df_Histry9_clearned2_4.pkl", 'rb')
    df4 = pickle.load(f2)
    f2.close()
    print(df1.shape)
    print(df2.shape)
    print(df3.shape)
    print(df4.shape)

    # df1=pd.concat([df1,df2,df3,df4],axis=0).reset_index(drop=True)
    # f0 = open("../data/feature_related/df_Histry9_clearned3.pkl", 'wb')
    # pickle.dump(df1, f0)
    # f0.close()
    # df1.to_csv("../data/feature_related/df_Histry9_clearned3.csv")


def hist_attri_ftr():
    f2 = open("../data/feature_related/df_Histry9_clearned3.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    f2 = open("../data_wo_threshold/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'rb')
    dic_cmbdty_attribution = pickle.load(f2)
    f2.close()

    lst_columns=list(dic_cmbdty_attribution.keys())
    lst_columns.append('cmbdty_attri_sum')
    df[lst_columns].hist(bins=20, figsize=(20, 15))
    plt.show()


def concat_lst_cmbdty_attribution():
    f2 = open("../data/feature_related/lst_cmbdty_attribution_1.pkl", 'rb')
    df1 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/lst_cmbdty_attribution_2.pkl", 'rb')
    df2 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/lst_cmbdty_attribution_3.pkl", 'rb')
    df3 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/lst_cmbdty_attribution_4.pkl", 'rb')
    df4 = pickle.load(f2)
    f2.close()

    print(len(df1))
    print(len(df2))
    print(len(df3))
    print(len(df4))
    # df1.extend(df2)
    # df1.extend(df3)
    # df1.extend(df4)
    # f0 = open("../data/feature_related/lst_cmbdty_attribution.pkl", 'wb')
    # pickle.dump(df1, f0)
    # f0.close()


def lst_to_df_Histty9_cleaned3():
    f2 = open("../data/feature_related/lst_cmbdty_attribution.pkl", 'rb')
    lst_cmbdty_attribution = pickle.load(f2)
    f2.close()
    print(len(lst_cmbdty_attribution))

    f2 = open("../data_wo_threshold/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'rb')
    dic_cmbdty_attribution = pickle.load(f2)
    f2.close()

    f2 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    print("df.shape[0]:", df.shape[0])

    df_cmbdty_attri_144=pd.DataFrame(lst_cmbdty_attribution, columns=list(dic_cmbdty_attribution.keys()))
    df_cmbdty_attri_144['cmbdty_attri_sum'] = df_cmbdty_attri_144.apply(lambda x: x.sum(), axis=1)
    print("df_cmbdty_attri_144.shape:",df_cmbdty_attri_144.shape)

    df = pd.concat([df, df_cmbdty_attri_144], axis=1)
    print("df.shape[0]:",df.shape[0])

    f0 = open("../data/feature_related/df_Histry9_clearned3.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_Histry9_clearned3.csv")


def time_series_split(df):
    """先看一下20%的患者时间线"""
    date_quantile=df['RY_DATE'].quantile(0.8)   #2019-08-15

    # print("df['RY_DATE'].quantile(0):",df['RY_DATE'].quantile(0))
    # print("df['RY_DATE'].quantile(1):",df['RY_DATE'].quantile(1))
    df['ry_date_label'] = df['RY_DATE'].apply(lambda x: 1 if x >= date_quantile else 0)
    # print("date_quantile:",date_quantile)
    # print(df[df['ry_date_label']==0].shape[0])  # 259122
    # print(df[df['ry_date_label'] == 1].shape[0])  # 64785
    # print(df.groupby('ry_date_label')['zfy_label'].sum())   #0：12954； 1：3242
    X_train_raw=df[df['ry_date_label']==0].reset_index(drop=True)
    y_train_raw = X_train_raw['zfy_label'].to_frame()
    X_test_raw=df[df['ry_date_label']==1].reset_index(drop=True)
    y_test_raw=X_test_raw['zfy_label'].to_frame()

    return X_train_raw, X_test_raw, y_train_raw, y_test_raw


def time_series():
    """先看一下20%的患者时间线"""

    f2 = open("../data/feature_related/df_Histry9_clearned4.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    df = df.reset_index(drop=True)

    # 处理标签（二分类）
    # 住院费用分类
    # 先按照95%以上为high cost
    print("df.shape[0]:",df.shape[0])
    zyfy = df['ZFY']
    # quantile_value = zyfy.quantile(0.95)
    quantile_value = zyfy.quantile(0.95)
    print(quantile_value)  # 40747.12
    df['zfy_label'] = df['ZFY'].apply(lambda x: 1 if x >= quantile_value else 0)  # 高于95为1，低于95为0
    # print(df['RY_DATE'].values[0])
    # print(type(df['RY_DATE'].values[0]))  #<class 'numpy.datetime64'>
    date_quantile=df['RY_DATE'].quantile(0.8)   #2019-08-15

    print("df['RY_DATE'].quantile(0):",df['RY_DATE'].quantile(0))
    print("df['RY_DATE'].quantile(1):",df['RY_DATE'].quantile(1))
    df['ry_date_label'] = df['RY_DATE'].apply(lambda x: 1 if x >= date_quantile else 0)
    print("date_quantile:",date_quantile)
    print(df[df['ry_date_label']==0].shape[0])  # 259122
    print(df[df['ry_date_label'] == 1].shape[0])  # 64785
    print(df.groupby('ry_date_label')['zfy_label'].sum())   #0：12954； 1：3242




def threshold_adjust():
    f2 = open("../data_wo_threshold/results/y_pred_prob_lgbm_86_time_s_ub.pkl", 'rb')
    y_pred_prob = pickle.load(f2)   #<class 'numpy.ndarray'>
    f2.close()
    f2 = open("../data_wo_threshold/results/y_test_folds_lgbm_86_time_s_ub.pkl", 'rb')
    y_test_folds = pickle.load(f2)   #<class 'pandas.core.series.Series'>
    f2.close()

    # #转换为numpy数组
    y_test_folds=y_test_folds.values

    num_of_threshold=1
    y_pred=np.zeros((num_of_threshold,y_pred_prob.shape[0]))
    row_idx=0

    nda_results=np.zeros((7,num_of_threshold))
    nda_results[0]=np.linspace(0, 1, num=num_of_threshold)

    # for threshold in np.linspace(0, 1, num=num_of_threshold):
    for threshold in [0.245]:
        y_pred[row_idx]=np.where(y_pred_prob>=threshold, 1, 0)


        m = confusion_matrix(y_test_folds, y_pred[row_idx])  #混淆矩阵
        list_cf_mtx = [[m[0][0], m[0][1]], [m[1][0], m[1][1]]]
        Precision = precision_score(y_test_folds, y_pred[row_idx])
        Recall = recall_score(y_test_folds, y_pred[row_idx])
        Speciality = m[0][0] / (m[0][0] + m[0][1])
        Accuracy = (m[0][0] + m[1][1]) / (m[0][0] + m[1][1] + m[0][1] + m[1][0])
        F1_score = f1_score(y_test_folds, y_pred[row_idx])
        G_mean = np.sqrt(Speciality * Recall)
        nda_results[1:,row_idx]=[Precision,Recall,Speciality,Accuracy,F1_score,G_mean]
        row_idx += 1

        # 输出混淆矩    阵
        print(threshold)
        print(m)

    # plt.plot(nda_results[0],nda_results[1], label='Precision')
    # plt.plot(nda_results[0], nda_results[2], label='Recall')
    # plt.plot(nda_results[0], nda_results[3], label='Speciality')
    # plt.plot(nda_results[0], nda_results[4], label='Accuracy')
    # plt.plot(nda_results[0],nda_results[5], label='F1_score')
    # plt.plot(nda_results[0], nda_results[6], label='G_mean')
    # plt.legend(loc='lower right')
    # plt.show()






if __name__=="__main__":
    """"""

    threshold_adjust()
    # time_series()
    # lst_to_df_Histty9_cleaned3()
    # get_histry_feature()
    # data_PreProcessing()
    # write_csv()

    # kmeans_cost()
    # xlsxpath = "../data/csv_xslx/AA_MULTI_DESC_TENSOR.xlsx"
    # history_feature_weighted_avg2(xlsxpath)
    # process_last_time_cost_per_day()
    # process_my_fault()
    # get_network_ftr_comorbidity_attri("../data_wo_threshold")

    # concat_df()
    # concat_lst_cmbdty_attribution()
    # hist_attri_ftr()
    # concat_lst_cmbdty_attribution()
    # f2 = open("../data/feature_related/lst_cmbdty_attribution.pkl", 'rb')
    # lst = pickle.load(f2)
    # f2.close()
    # print(len(lst))

    # # 多进程处理
    # f2 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'rb')
    # df = pickle.load(f2)
    # f2.close()
    #
    # len_df=df.shape[0]
    # print(len_df)
    # # df1 = df[df.index < 100]
    # df1=df[df.index<int(1*len_df/4)]
    # df2=df[(df.index<int(2*len_df/4))&(df.index>=int(1*len_df/4))]
    # df3=df[(df.index<int(3*len_df/4))&(df.index>=int(2*len_df/4))]
    # df4=df[df.index>=int(3*len_df/4)]
    #
    # print(df1.shape)
    # print(df2.shape)
    # print(df3.shape)
    # print(df4.shape)
    # (80976, 25)
    # (80977, 25)
    # (80977, 25)
    # (80977, 25)

    #
    # p = multiprocessing.Process(target=get_network_ftr_comorbidity_attri_multi_p, args=(df1,"1"))  # 创建一个进程，args传参 必须是元组
    # p.start()  # 运行线程p
    # p2 = multiprocessing.Process(target=get_network_ftr_comorbidity_attri_multi_p, args=(df2,"2"))  # 创建一个进程，args传参 必须是元组
    # p2.start()  # 运行线程p
    # p3 = multiprocessing.Process(target=get_network_ftr_comorbidity_attri_multi_p, args=(df3,"3"))  # 创建一个进程，args传参 必须是元组
    # p3.start()  # 运行线程p
    # p4 = multiprocessing.Process(target=get_network_ftr_comorbidity_attri_multi_p,
    #                              args=(df4, "4"))  # 创建一个进程，args传参 必须是元组
    # p4.start()  # 运行线程p

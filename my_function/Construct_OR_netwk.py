import pickle
from igraph import *
from tqdm import tqdm
import numpy as np
import ML_v3_Tree
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
import time
import my_function.useful_fun as myfunction


def compute_OR_dss_cost(df_X_train_raw,diseases_set):
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
            for index_, disease in enumerate(
                    tqdm(df_diseases["cy_diseases_"])):  # df_diseases["diseases_"]中每行记录中的疾病无重复,对应于出院诊断
                for d in disease:
                    if d in dic_non_high_cost_num:
                        dic_non_high_cost_num[d] += 1
                    else:
                        dic_non_high_cost_num[d] = 1
                        # dic_high_cost_num[d]=0
        elif index == 1:  # 高花费记录
            len_high_rcd_count = df_diseases.shape[0]
            print("len_high_rcd_count", df_diseases.shape[0])
            for index_, disease in enumerate(tqdm(df_diseases["cy_diseases_"])):
                for d in disease:
                    if d in dic_high_cost_num:
                        dic_high_cost_num[d] += 1
                    else:
                        dic_high_cost_num[d] = 1

    set_dise1 = set(dic_high_cost_num.keys())
    set_dise2 = set(dic_non_high_cost_num.keys())
    set_dise = set_dise1 | set_dise2
    set_dise=set_dise&diseases_set
    print("len(set_dise)",len(set_dise))
    # len_rcd = df.shape[0]  #1185
    # dic_OR = {}
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
                list_edge_OR=["High Cost",i,OR,ci_low,ci_high]
                list_edges_OR.append(list_edge_OR)
            # if ci_high < 1 and OR < 1:
            #     dic_OR[i] = [OR, ci_low, ci_high]
    # print("len(dic_OR)",len(dic_OR))  #186
    # list_all_edges_OR.extend(list_edges_OR)
    # f0 = open(savepath, 'wb')
    # pickle.dump(dic_OR, f0)
    # f0.close()
    return list_edges_OR


def compute_disease_pair_OR(df_X_train_raw,diseases_set):
    """
    计算OR,用训练集的OR计算
    输入参数：_df含至少两列：cy_diseases_和zfy_label
                    cy_diseases_包含的是出院诊断，且每条记录的出院诊断不存在重复的疾病（即为set()）
    """

    print("len(diseases_set)",len(diseases_set))  #1139
    count=0
    list_all_edges_OR=[]
    dic_OR_edges={}
    print("-----------计算disease_pairs OR中------------------------")
    pastt=time.time()
    for disease_case in diseases_set:
        count+=1
        if count%100==0:
            print("已计算OR的疾病数：",count)
        df_X_train_raw['dss_case_label']=df_X_train_raw['cy_diseases_'].apply(lambda x:1 if disease_case in x else 0)

        #以下同compute_OR()中流程
        dic_non_high_cost_num = {}  # b
        dic_high_cost_num = {}  # a
        for index, df_diseases in df_X_train_raw.groupby(['dss_case_label']):  # index 的取值：1和0  1对应于高花费住院记录，2对应于非高花费住院记录
            if index == 0:
                len_non_high_rcd_count = df_diseases.shape[0]
                # print("len_non_high_rcd_count", df_diseases.shape[0])
                for index_, disease in enumerate(df_diseases["cy_diseases_"]):  # df_diseases["diseases_"]中每行记录中的疾病无重复,对应于出院诊断
                    for d in disease:
                        if d in dic_non_high_cost_num:
                            dic_non_high_cost_num[d] += 1
                        else:
                            dic_non_high_cost_num[d] = 1
                            # dic_high_cost_num[d]=0
            elif index == 1:
                len_high_rcd_count = df_diseases.shape[0]
                # print("len_high_rcd_count", df_diseases.shape[0])
                for index_, disease in enumerate(df_diseases["cy_diseases_"]):
                    for d in disease:
                        if d in dic_high_cost_num:
                            dic_high_cost_num[d] += 1
                        else:
                            dic_high_cost_num[d] = 1

        set_dise1 = set(dic_high_cost_num.keys())
        set_dise2 = set(dic_non_high_cost_num.keys())
        set_dise = set_dise1 | set_dise2
        set_dise.remove(disease_case)   #把本身去掉
        set_dise=set_dise&diseases_set
        # print("len(set_dise)",len(set_dise))
        # len_rcd = df.shape[0]  #1185
        dic_OR_edge = {}
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
                OR = (0.1*a *d) / (0.1*b *c)
                se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
                ci_high = np.exp(np.log(OR) + 1.96 * se)
                ci_low = np.exp(np.log(OR) - 1.96 * se)
                if ci_low > 1 and OR > 1:
                    # dic_OR[i] = [OR, ci_low, ci_high]
                    if disease_case>i:
                        list_edge_OR=[i,disease_case,OR,ci_low,ci_high]
                    else:
                        list_edge_OR = [disease_case,i, OR, ci_low, ci_high]
                        # dic_OR_edge[(disease_case,i)] = list_edge_OR
                # if ci_high < 1 and OR < 1:
                #     dic_OR[i] = [OR, ci_low, ci_high]
        # print("len(dic_OR)",len(dic_OR))  #186
                    list_edges_OR.append(list_edge_OR)
        list_all_edges_OR.extend(list_edges_OR)
        # dic_OR_edges.update(dic_OR_edge)

    print("-----------计算disease_pairsOR OR耗时：------------------------",(time.time()-pastt)/60)


    df_output=pd.DataFrame(list_all_edges_OR,columns=['disease1','disease2','OR','ci_low','ci_high'])
    # df_output.to_csv('data/OR_edges_test.csv')
    print("未去重前，共病网络有意义的OR边数：", df_output.shape[0])
    df_output=df_output.drop_duplicates(subset=['disease1','disease2'])
    print("去重后，共病网络有意义的OR边数：",df_output.shape[0])
    list_all_edges_OR=df_output.values.tolist()

    print("----------------------------应该减少一半----------------------------------------")

    return list_all_edges_OR


def read_data(percetile):
    xlsx_path = "data/csv_xslx/ICD3_group_disease_names.xlsx"
    xlsxpath = "data/csv_xslx/AA_MULTI_DESC_TENSOR.xlsx"
    # 处理基础特征  要用LR的话，还要对基础特征进行处理
    df, columns_name = ML_v3_Tree.read_data_feature_extract(xlsx_path, xlsxpath,percetile)  # 220疾病组的columns_name
    # X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(df, df['zfy_label'], test_size=0.2,
    #                                                                     random_state=42, shuffle=True,
    #                                                                     stratify=df['zfy_label'])

    # 按时序划分
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = myfunction.time_series_split(df)

    X_train_raw=X_train_raw[['cy_diseases_','zfy_label']]
    return X_train_raw


def get_diseases_set(df_X_train_raw):

    diseases_set=set()
    for record in df_X_train_raw['cy_diseases_']:
        diseases_set=diseases_set|record

    num_of_records=df_X_train_raw.shape[0]
    diseases_set_=set()
    print("---------------计算流行率中------------------------")

    for disease in diseases_set:   # 纳入流行率大于0.01%的疾病
        df_X_train_raw['dss_case_label'] = df_X_train_raw['cy_diseases_'].apply(lambda x: 1 if disease in x else 0)
        num_of_disease=df_X_train_raw['dss_case_label'].sum()
        if num_of_disease/num_of_records>0.0001:
            diseases_set_.add(disease)
    df_X_train_raw.drop(columns='dss_case_label', axis=1, inplace=True)
    return diseases_set_


def construct_OR_network_graph(percentile, save_dir, list_edges_OR,save_path,flag_edge_type):
    """功能：构网构图
    输入参数：percentile 分位数，只画出相关系数在percentile以上的边
    输入参数：type: type='OR';
    _list_edges_OR:list of lists:[disease,disease,OR,ci_low,ci_high] or ['High Cost',disease,OR,ci_low,ci_high]
    _flag_edge_type:"OR"  or "distance"
    """
    print("OR the num of edge:", len(list_edges_OR))
    node_set = set()

    # quantile_value=pd.Series([edge[4] for edge in edge_list_RR]).quantile(percentile)
    edge_list_RR = [i for i in list_edges_OR if i[2] >= percentile]

    for edge in edge_list_RR:
        node_set.add(edge[0])
        node_set.add(edge[1])
    node_name_list = sorted(list(node_set))  # 节点名称，排序后
    print("OR the num of node:", len(node_name_list))

    g = Graph()
    g.add_vertices(len(node_name_list))
    g.vs['name'] = node_name_list
    g.vs['label'] = node_name_list
    g.add_edges((edge[0], edge[1]) for edge in edge_list_RR)

    if flag_edge_type=="OR":
        RR_list = [0 for j in range(len(edge_list_RR))]
        CI_high = [0 for j in range(len(edge_list_RR))]
        CI_low = [0 for j in range(len(edge_list_RR))]
        for edge in edge_list_RR:
            edge_id = g.get_eid(edge[0], edge[1])
            RR_list[edge_id] = edge[2]
            CI_high[edge_id] = edge[4]
            CI_low[edge_id] = edge[3]
        g.es['weight'] = RR_list
        g.es['OR_high_CI'] = CI_high
        g.es['OR_low_CI'] = CI_low
    elif flag_edge_type=="distance":
        distance_list = [0 for j in range(len(edge_list_RR))]
        for edge in edge_list_RR:
            edge_id = g.get_eid(edge[0], edge[1])
            distance_list[edge_id] = edge[5]
        g.es['weight'] = distance_list

    print(summary(g))
    g.write(save_dir + save_path, "gml")


def compute_edge_distance(list_edges_OR):
    df_edges_OR=pd.DataFrame(list_edges_OR,columns=['node1','node2','OR_value','CI_low','CI_high'])
    max_or=df_edges_OR['OR_value'].max()
    min_or=df_edges_OR['OR_value'].min()
    df_edges_OR['distance']=df_edges_OR['OR_value'].apply(lambda x:(max_or-x)/(max_or-min_or))
    list_edges_OR=df_edges_OR.values.tolist()
    return list_edges_OR


def construct_OR_network(df_X_train_raw,percentile, save_dir,save_path1,save_path2):
    """
    1. 用训练集的数据构建OR网络，构成后保存到gml文件中，之后不再重复计算
    输入参数：_df:需要包含X_train_raw["cy_diseases_"]
    """
    diseases_set = get_diseases_set(df_X_train_raw)
    list_all_dss_pair_edges_OR=compute_disease_pair_OR(df_X_train_raw,diseases_set)
    list_cost_dss_edges_OR=compute_OR_dss_cost(df_X_train_raw,diseases_set)
    list_all_dss_pair_edges_OR.extend(list_cost_dss_edges_OR)   #OR网络中所有有意义的边

    list_all_dss_pair_edges_OR=compute_edge_distance(list_all_dss_pair_edges_OR)

    print("OR网络中有意义的边数：",len(list_all_dss_pair_edges_OR))

    construct_OR_network_graph(percentile, save_dir, list_all_dss_pair_edges_OR,save_path1,'OR')
    construct_OR_network_graph(percentile, save_dir, list_all_dss_pair_edges_OR, save_path2, 'distance')


def construct_OR_distance_network(percetile):
    """该函数用于构建疾病和高花费的OR网络及对应的distance网络"""
    df_X_train_raw = read_data(percetile)

    percetile_str=str(percetile)[2:4]

    save_dir = "data"
    save_path1 = "/gml_dir/OR_Graph_all_"+percetile_str+"_time_series.gml"
    save_path2 = "/gml_dir/distance_OR_Graph_all_"+percetile_str+"_time_series.gml"
    construct_OR_network(df_X_train_raw, 1, save_dir, save_path1, save_path2)

    # gml_path = "data/gml_dir/distance_OR_Graph_all.gml"
    # find_shortest_path(gml_path)


if __name__ == "__main__":
    percetile1=0.95
    # percetile2 = 0.9
    # percetile3 = 0.8
    construct_OR_distance_network(percetile1)
    # construct_OR_distance_network(percetile2)
    # construct_OR_distance_network(percetile3)

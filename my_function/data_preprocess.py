import multiprocessing

import my_function.useful_fun as myfunction
import pickle
import time
import cx_Oracle
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import isnan
import networkx as nx


# 提取历史特征，对建模部分的数据进行预处理
def get_histry_feature():
    """  提取历史特征+数据预处理 """
    print("------------------读入并处理2019年的数据中，剔除入院诊断中不含IHD的记录--------")

    # 打开数据库连接
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    # db = cx_Oracle.connect('system', 'Yp251200.', '127.0.0.1:1521/ORCL', mode=cx_Oracle.SYSDBA)
    # 操作游标
    cr1 = db.cursor()
    sql1 = 'select SFZH,LYFS,RN,XB,NL,YYDJ_J,YYDJ_D,RYTJ,RY_DATE,XZZ_XZQH2,ALL_DISEASE,ALL_FLAGS,ZFY from scott.YP_TTL_IHD_2YEARS'
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    # print(cyzd_data)   list of data
    names = [i[0] for i in cr1.description]
    # print(names)
    df = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()
    # print(df.head())
    df['diseases'] = df['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df['rybq_flags'] = df['ALL_FLAGS'].apply(lambda x: [i for i in x.split(',')])
    df.drop(columns=['ALL_DISEASE'], axis=1, inplace=True)
    df.drop(columns=['ALL_FLAGS'], axis=1, inplace=True)
    # names = [i[0] for i in cr1.description]
    cyzd_data = np.array(df).tolist()
    print("出院诊断有IHD的住院记录数： ", df.shape[0])
    # print(cyzd_data)
    # print(df.columns.values)
    # 只保留入院诊断（QYBQ=1）
    list_ry_diaseses = []
    diseases_index = df.columns.values.tolist().index('diseases')
    rybq_flags_index = df.columns.values.tolist().index('rybq_flags')
    for i in cyzd_data:
        index = -1
        list_ry_disease_perR = []
        for j in i[rybq_flags_index]:
            index += 1
            if j == '1':
                list_ry_disease_perR.append(i[diseases_index][index])
        list_ry_diaseses.append(list_ry_disease_perR)  # 入院疾病列表 list of lists :every list in the outer list is a record

    df = pd.concat([df, pd.DataFrame({'ry_diseases': list_ry_diaseses})], axis=1)
    # df.drop(columns='diseases', axis=1, inplace=True)
    df.drop(columns='rybq_flags', axis=1, inplace=True)

    # print(df['ry_diseases'])
    del list_ry_diaseses

    # 除去异常编码；df['ry_diseases_'] 存放的是入院诊断列表 ; df['cy_diseases_'] 存放的是出院诊断集合
    df['ry_diseases_'] = df['ry_diseases'].apply(lambda x: [i for i in x if
                                                            len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[
                                                                1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[
                                                                2] <= '9'])
    df['cy_diseases_'] = df['diseases'].apply(lambda x: set([i for i in x if
                                                             len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[
                                                                 1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[
                                                                 2] <= '9']))
    df.drop(columns='ry_diseases', axis=1, inplace=True)
    df.drop(columns='diseases', axis=1, inplace=True)
    # print(df.columns)
    # print(df['diseases'])

    # 除去入院诊断为空和入院诊断没有IHD的行
    del_index = []
    print("-------------------------去除入院诊断为空和入院诊断没有IHD的行————————————————")
    for index, i in enumerate(tqdm(df["ry_diseases_"])):
        if len(i) == 0:
            del_index.append(index)
        elif "I20" in i or "I21" in i or "I22" in i or "I23" in i or "I24" in i or "I25" in i:
            continue
        else:
            del_index.append(index)
    # for d_index in tqdm(del_index):  #26948条
    #     df=df[df.index!=d_index]
    print("未删除入院诊断中没有IHD的记录时，住院记录数为：", df.shape[0])
    df.drop(index=del_index, axis=0, inplace=True)
    df = df.reset_index(drop=True)
    print("删除入院诊断中没有IHD的记录后，住院记录数为：", df.shape[0])

    print("-------------------------提取历史特征中————————————————")
    # 提取历史特征：
    # 找出19年的入院诊断为IHD的患者的身份证号
    # df_sfzh=df['SFZH'].drop_duplicates()  # 19年的入院诊断为IHD的患者的身份证号
    df_sfzh = df[['SFZH', 'RN']]
    df_sfzh.columns = ['SFZH', 'RN_CASE']
    # print("纳入（19年）患者数（入院诊断含IHD）：",df_sfzh.shape[0])
    # 读取YP_P_16_19  #该表存储按19年出院诊断的身份账号索引的16-19年的所有住院记录
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    cr1 = db.cursor()
    sql1 = 'select SFZH,RN,RY_DATE,CY_DATE,ZFY from scott.YP_P_15_19'
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    # print(cyzd_data)   list of data
    names = [i[0] for i in cr1.description]
    # print(names)
    df_16to19 = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()

    df_sfzh = df_sfzh.merge(df_16to19, how='left', on=['SFZH'])
    records_histry = []
    time_flag=0
    print("开始提取历史特征------------------")
    pastt = time.time()
    for index, df_idnumber in df_sfzh.groupby(['SFZH', 'RN_CASE']):  # 唯一标识一条记录
        time_flag+=1
        if(time_flag%5000==0):
            print("time_flag-----",time_flag)
        case_sfzh = df_idnumber['SFZH'].values[0]
        case_RN = df_idnumber['RN_CASE'].values[0]

        case_ry_data = df_idnumber[df_idnumber['RN'] == case_RN]['RY_DATE'].values[0]
        case_cy_data = df_idnumber[df_idnumber['RN'] == case_RN]['CY_DATE'].values[0]
        los_this_time=(case_cy_data-case_ry_data).astype('timedelta64[D]')/np.timedelta64(1, 'D')

        record_histry = []
        record_histry.append(case_sfzh)
        record_histry.append(case_RN)
        record_histry.append(los_this_time)
        # 取前三年区间的记录

        df_idnumber['admssn_time'] = df_idnumber['RY_DATE'].apply(lambda x: 1 if ((case_ry_data-x).days<=1095 and (case_ry_data-x).days>0) else 0)

        df_idnumber = df_idnumber[df_idnumber['admssn_time'] == 1]  # 前三年的住院记录
        if df_idnumber.shape[0] != 0:
            # 前一次住院就诊费用
            last_time=df_idnumber['RN'].max()
            # print(case_RN-last_time)
            last_time_zfy = df_idnumber[df_idnumber['RN'] == last_time]['ZFY'].values[0]
            record_histry.append(last_time_zfy)

            los_last_time_ry= df_idnumber[df_idnumber['RN'] == last_time]['RY_DATE'].values[0]
            los_last_time_cy = df_idnumber[df_idnumber['RN'] == last_time]['CY_DATE'].values[0]
            los_last_time=(los_last_time_cy-los_last_time_ry).astype('timedelta64[D]')/np.timedelta64(1, 'D')
            record_histry.append(los_last_time)
            interval_now_last=(case_ry_data-los_last_time_cy).astype('timedelta64[D]')/np.timedelta64(1, 'D')
            record_histry.append(interval_now_last) #本次入院与上次出院的时间差

            # 前三年住院费用的统计指标
            mean_zfy = df_idnumber['ZFY'].mean()
            med_zfy = df_idnumber['ZFY'].median()
            max_zfy = df_idnumber['ZFY'].max()
            std_zfy = df_idnumber['ZFY'].std(ddof = 0)
            min_zfy = df_idnumber['ZFY'].min()
            record_histry.append(mean_zfy)
            record_histry.append(med_zfy)
            record_histry.append(max_zfy)
            record_histry.append(std_zfy)
            record_histry.append(min_zfy)

            # 新增历史特征last_time_cost_per_day
            last_time_cost_per_day=last_time_zfy/(los_last_time+1)
            record_histry.append(last_time_cost_per_day)
        else:
            record_histry.append(-999)  #last_time_zfy
            record_histry.append(-999)  #los_last_time
            record_histry.append(-999)   #interval_now_last  本次入院与上次出院的时间差
            record_histry.append(-999)  #mean_zfy
            record_histry.append(-999)  #med_zfy
            record_histry.append(-999)  #max_zfy
            record_histry.append(-999)  #std_zfy
            record_histry.append(-999)  #min_zfy
            record_histry.append(-999)  # last_time_cost_per_day
        record_histry.append(df_idnumber.shape[0])
        records_histry.append(record_histry)
    df_histry = pd.DataFrame(records_histry,
                             columns=['SFZH', 'RN', 'los_this_time', 'zfy_last_time', 'los_last_time',
                                      'interval_thistimery_pasttimecy', 'mean_zfy_last_3y', 'med_zfy_last_3y',
                                      'max_zfy_last_3y', 'std_zfy_last_3y', 'min_zfy_last_3y', 'zyci_last_3y', 'last_time_cost_per_day']
                             )
    del records_histry
    df = df.merge(df_histry, how='left', on=['SFZH', 'RN'])
    # df.drop(columns=['SFZH', 'RN'], axis=1, inplace=True)
    # df.drop(columns=['RN'], axis=1, inplace=True)
    print("提取历史特征耗时：",(time.time()-pastt)/60)

    f0 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7.csv")



def history_feature_weighted_avg2(xlsxpath):
    """处理历史特征"""
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()

    #加权平均
    dic_icd3_dchapter, dic_clmnname_index=myfunction.chapter_name_list(xlsxpath)

    # 提取历史特征：
    df_sfzh = df[['SFZH', 'RN','ry_diseases_']]
    df_sfzh.columns = ['SFZH', 'RN_CASE','ry_diseases_']
    # print("纳入（19年）患者数（入院诊断含IHD）：",df_sfzh.shape[0])
    # 读取YP_P_16_19  #该表存储按19年出院诊断的身份账号索引的16-19年的所有住院记录
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    cr1 = db.cursor()
    sql1 = 'select SFZH,RN,RY_DATE,ALL_DISEASE,ZFY,YYDJ_J from scott.YP_P_15_19'
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    # print(cyzd_data)   list of data
    names = [i[0] for i in cr1.description]
    # print(names)
    df_16to19 = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()

    df_16to19['diseases'] = df_16to19['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df_16to19['cy_diseases_'] = df_16to19['diseases'].apply(lambda x: [i for i in x if
                                                             len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[
                                                                 1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[
                                                                 2] <= '9'])
    df_16to19.drop(columns='diseases', axis=1, inplace=True)
    df_16to19.drop(columns='ALL_DISEASE', axis=1, inplace=True)

    df_sfzh = df_sfzh.merge(df_16to19, how='left', on=['SFZH'])
    records_histry = []
    time_flag = 0
    print("___________________group中____________________________")
    pastt=time.time()
    for index, df_idnumber in df_sfzh.groupby(['SFZH', 'RN_CASE']):  # 唯一标识一条记录
        time_flag+=1
        if(time_flag%5000==0):
            print("time_flag-----",time_flag)
        case_sfzh = df_idnumber['SFZH'].values[0]
        case_RN = df_idnumber['RN_CASE'].values[0]
        case_ry_data = df_idnumber[df_idnumber['RN'] == case_RN]['RY_DATE'].values[0]
        case_ry_diseases = df_idnumber[df_idnumber['RN'] == case_RN]['ry_diseases_'].values[0]
        case_dj_j=df_idnumber[df_idnumber['RN'] == case_RN]['YYDJ_J'].values[0]

        record_histry = []
        record_histry.append(case_sfzh)
        record_histry.append(case_RN)

        ry_dis_appear_hot=[0 for s in range(len(dic_clmnname_index)+1)]  #初始化
        for disease_ry in case_ry_diseases:
            if disease_ry in dic_icd3_dchapter:
                ry_dis_appear_hot[dic_clmnname_index[dic_icd3_dchapter[disease_ry]]]+=1
        if case_dj_j=='2':
            ry_dis_appear_hot[len(dic_clmnname_index)]=1
        else:
            ry_dis_appear_hot[len(dic_clmnname_index)] = 0
        df_idnumber['admssn_time'] = df_idnumber['RY_DATE'].apply(lambda x: 1 if ((case_ry_data-x).days<=1095 and (case_ry_data-x).days>0) else 0)

        df_idnumber = df_idnumber[df_idnumber['admssn_time'] == 1]  # 前三年的住院记录
        if df_idnumber.shape[0]>1:
            list_idnumber=df_idnumber[['cy_diseases_','ZFY','YYDJ_J']].values.tolist()
            # print(list_idnumber)
            list_sort_dist=[]
            dist_sum = 0
            for last_3y_rcd in list_idnumber:
                cy_dis_appear_hot=[0 for s in range(len(dic_clmnname_index)+1)]
                for disease_cy in last_3y_rcd[0]:
                    if disease_cy in dic_icd3_dchapter:
                        cy_dis_appear_hot[dic_clmnname_index[dic_icd3_dchapter[disease_cy]]] += 1
                if last_3y_rcd[2]=='2':
                    cy_dis_appear_hot[len(dic_clmnname_index)]=1
                else:
                    cy_dis_appear_hot[len(dic_clmnname_index)] = 0
                o_dist=0
                for k in range(len(dic_clmnname_index)):
                    o_dist+=(cy_dis_appear_hot[k]-ry_dis_appear_hot[k])**2
                o_dist=np.sqrt(o_dist)
                dist_sum+=o_dist
                # print("o_dist",o_dist)
                # print("dist_sum",dist_sum)
                list_sort_dist.append((last_3y_rcd[1],o_dist))
            weighted_avg=0
            if dist_sum!=0:
                for dist in list_sort_dist:
                    weight=(dist_sum-dist[1])/(dist_sum*(len(list_sort_dist)-1))
                    weighted_avg+=weight*dist[0]
            else:
                weighted_avg=df_idnumber['ZFY'].mean()
            record_histry.append(weighted_avg)
        elif df_idnumber.shape[0]==1:
            weighted_avg = df_idnumber['ZFY'].values[0]
            record_histry.append(weighted_avg)
        else:
            weighted_avg=-999
            record_histry.append(weighted_avg)
        records_histry.append(record_histry)
    print("_________________group耗时：--------------",(time.time()-pastt)/60)
    df_histry = pd.DataFrame(records_histry,columns=['SFZH', 'RN', 'weighted_average_cost_past_3y'])
    del records_histry
    df = df.merge(df_histry, how='left', on=['SFZH', 'RN'])

    f0 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.csv")


def data_PreProcessing():
    """数据预处理，清除住院时间交叉的id，los等于0的住院记录，住院费用不在1-99%的住院记录，离院方式不为1的住院记录"""
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()

    #清除住院时间交叉的id
    print('清除住院时间交叉的id------------------')
    pastt=time.time()
    df_dlt_sfzh=df[((df['interval_thistimery_pasttimecy']<0)&(df['interval_thistimery_pasttimecy']>-999))]
    list_dlt_sfzh=df_dlt_sfzh['SFZH'].drop_duplicates().tolist()
    print("len(list_dlt_sfzh)",len(list_dlt_sfzh))
    df['flag_dlt']=df['SFZH'].apply(lambda x:0 if x in list_dlt_sfzh else 1)   # 这里太慢了
    df_len1 = df.shape[0]
    df=df[df['flag_dlt']==1]
    df.drop(columns='flag_dlt', axis=1, inplace=True)
    df=df.reset_index(drop=True)
    print("住院时间发生交叉的患者的2019年所有住院记录数(删除)：",df.shape[0]-df_len1)
    print('清除住院时间交叉的id耗时：------------------',(time.time()-pastt)/60)
    print("2019年纳入住院记录数：", df.shape[0])
    #清除住院时长等于0和住院费用不在1-99%之间的住院记录
    df_len2=df.shape[0]
    quantile_1perc = df['ZFY'].quantile(0.01)
    quantile_99perc = df['ZFY'].quantile(0.99)
    df = df[((df['los_this_time'] > 0) & (df['ZFY'] <= quantile_99perc) & (df['ZFY'] >= quantile_1perc))]
    # df = df[((df['los_this_time'] > 0) & (df['ZFY'] >= quantile_1perc))]

    print("住院时长等于0和住院费用不在1-99%之间的住院记录数(删除)：", df.shape[0] - df_len2)

    #清除离院方式不为1的住院记录
    df_len3 = df.shape[0]
    df=df[df['LYFS']=='1']
    print("离院方式不为1的住院记录数(删除)：", df.shape[0] - df_len3)

    df = df.reset_index(drop=True)

    f0 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.csv")


def history_feature_weighted_avg(xlsxpath):
    """处理历史特征"""
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()

    #处理标准差
    df1 = df[((df['std_zfy_last_3y']!=-999) & (df['std_zfy_last_3y'].isnull()!=True))]
    df2 = df[df['std_zfy_last_3y'].isnull()==True]
    df3 = df[df['std_zfy_last_3y']==-999]
    df1['std_zfy_last_3y']=df1['std_zfy_last_3y']*(df1['zyci_last_3y']-1)/df1['zyci_last_3y']
    df2['std_zfy_last_3y']=0
    df=pd.concat([df1,df2,df3],axis=0)
    df.sort_index(inplace=True)

    #新增历史特征last_time_cost_per_day
    df['last_time_cost_per_day']=df['zfy_last_time']/(df['los_last_time']+1)

    #修改异常值取值
    df['interval_thistimery_pasttimecy'] = df['interval_thistimery_pasttimecy'].apply(lambda x:-999 if x==9999 else x)

    #加权平均
    dic_icd3_dchapter, dic_clmnname_index=myfunction.chapter_name_list(xlsxpath)

    # 提取历史特征：
    df_sfzh = df[['SFZH', 'RN','ry_diseases_']]
    df_sfzh.columns = ['SFZH', 'RN_CASE','ry_diseases_']
    # print("纳入（19年）患者数（入院诊断含IHD）：",df_sfzh.shape[0])
    # 读取YP_P_16_19  #该表存储按19年出院诊断的身份账号索引的16-19年的所有住院记录
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    cr1 = db.cursor()
    sql1 = 'select SFZH,RN,RY_DATE,ALL_DISEASE,ZFY from scott.YP_P_15_19'
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    # print(cyzd_data)   list of data
    names = [i[0] for i in cr1.description]
    # print(names)
    df_16to19 = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()

    df_16to19['diseases'] = df_16to19['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df_16to19['cy_diseases_'] = df_16to19['diseases'].apply(lambda x: [i for i in x if
                                                             len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[
                                                                 1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[
                                                                 2] <= '9'])
    df_16to19.drop(columns='diseases', axis=1, inplace=True)
    df_16to19.drop(columns='ALL_DISEASE', axis=1, inplace=True)

    df_sfzh = df_sfzh.merge(df_16to19, how='left', on=['SFZH'])
    records_histry = []
    time_flag = 0
    print("___________________group中____________________________")
    pastt=time.time()
    for index, df_idnumber in df_sfzh.groupby(['SFZH', 'RN_CASE']):  # 唯一标识一条记录
        time_flag+=1
        if(time_flag%5000==0):
            print("time_flag-----",time_flag)
        case_sfzh = df_idnumber['SFZH'].values[0]
        case_RN = df_idnumber['RN_CASE'].values[0]
        case_ry_data = df_idnumber[df_idnumber['RN'] == case_RN]['RY_DATE'].values[0]
        case_ry_diseases = df_idnumber[df_idnumber['RN'] == case_RN]['ry_diseases_'].values[0]

        record_histry = []
        record_histry.append(case_sfzh)
        record_histry.append(case_RN)

        ry_dis_appear_hot=[0 for s in range(len(dic_clmnname_index))]  #初始化
        for disease_ry in case_ry_diseases:
            if disease_ry in dic_icd3_dchapter:
                ry_dis_appear_hot[dic_clmnname_index[dic_icd3_dchapter[disease_ry]]]+=1

        df_idnumber['admssn_time'] = df_idnumber['RY_DATE'].apply(lambda x: 1 if ((case_ry_data-x).days<=1095 and (case_ry_data-x).days>0) else 0)

        df_idnumber = df_idnumber[df_idnumber['admssn_time'] == 1]  # 前三年的住院记录
        if df_idnumber.shape[0]>1:
            list_idnumber=df_idnumber[['cy_diseases_','ZFY']].values.tolist()
            # print(list_idnumber)
            list_sort_dist=[]
            dist_sum = 0
            for last_3y_rcd in list_idnumber:
                cy_dis_appear_hot=[0 for s in range(len(dic_clmnname_index))]
                for disease_cy in last_3y_rcd[0]:
                    if disease_cy in dic_icd3_dchapter:
                        cy_dis_appear_hot[dic_clmnname_index[dic_icd3_dchapter[disease_cy]]] += 1
                o_dist=0
                for k in range(len(dic_clmnname_index)):
                    o_dist+=(cy_dis_appear_hot[k]-ry_dis_appear_hot[k])**2
                o_dist=np.sqrt(o_dist)
                dist_sum+=o_dist
                # print("o_dist",o_dist)
                # print("dist_sum",dist_sum)
                list_sort_dist.append((last_3y_rcd[1],o_dist))
            weighted_avg=0
            if dist_sum!=0:
                for dist in list_sort_dist:
                    weight=(dist_sum-dist[1])/(dist_sum*(len(list_sort_dist)-1))
                    weighted_avg+=weight*dist[0]
            else:
                weighted_avg=df_idnumber['ZFY'].mean()
            record_histry.append(weighted_avg)
        elif df_idnumber.shape[0]==1:
            weighted_avg = df_idnumber['ZFY'].values[0]
            record_histry.append(weighted_avg)
        else:
            weighted_avg=-999
            record_histry.append(weighted_avg)
        records_histry.append(record_histry)
    print("_________________group耗时：--------------",(time.time()-pastt)/60)
    df_histry = pd.DataFrame(records_histry,columns=['SFZH', 'RN', 'weighted_average_cost_past_3y'])
    del records_histry
    df = df.merge(df_histry, how='left', on=['SFZH', 'RN'])

    f0 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.csv")


def processing_history_miss():
    """

    :return:
    """
    # csv_path=''
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7_clearned.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()

    df['zfy_last_time']=df['zfy_last_time'].apply(lambda x:0 if x==-999 else x)
    df['los_last_time']=df['los_last_time'].apply(lambda x:0 if x==-999 else x)
    df['los_last_time']=df['los_last_time'].apply(lambda x:0 if x==-999 else x)
    df['interval_thistimery_pasttimecy'] = df['interval_thistimery_pasttimecy'].apply(lambda x: 1096 if x == -999 else x)
    df['mean_zfy_last_3y'] = df['mean_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    df['med_zfy_last_3y'] = df['med_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    df['max_zfy_last_3y'] = df['max_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    df['std_zfy_last_3y'] = df['std_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    df['min_zfy_last_3y'] = df['min_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    df['weighted_average_cost_past_3y'] = df['weighted_average_cost_past_3y'].apply(lambda x: 0 if x == -999 else x)
    df['last_time_cost_per_day'] = df['last_time_cost_per_day'].apply(lambda x: 0 if x == -999 else x)

    f0 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry8_clearned.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry8_clearned.csv")


def data_PreProcessing2():
    """数据预处理，清除住院时间交叉的id，los等于0的住院记录，历史特征-999用0替换"""
    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry7.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    print(df.shape[0])
    #清除住院时间交叉的id
    print('清除住院时间交叉的id------------------')
    pastt=time.time()
    df_dlt_sfzh=df[((df['interval_thistimery_pasttimecy']<0)&(df['interval_thistimery_pasttimecy']>-999))]
    list_dlt_sfzh=df_dlt_sfzh['SFZH'].drop_duplicates().tolist()
    print("len(list_dlt_sfzh)",len(list_dlt_sfzh))
    df['flag_dlt']=df['SFZH'].apply(lambda x:0 if x in list_dlt_sfzh else 1)   # 这里太慢了
    df_len1 = df.shape[0]
    df=df[df['flag_dlt']==1]
    df.drop(columns='flag_dlt', axis=1, inplace=True)
    df=df.reset_index(drop=True)
    print("住院时间发生交叉的患者的2019年所有住院记录数(删除)：",df.shape[0]-df_len1)
    print('清除住院时间交叉的id耗时：------------------',(time.time()-pastt)/60)
    print("2019年纳入住院记录数：", df.shape[0])
    #清除住院时长等于0和住院费用不在1-99%之间的住院记录
    df_len2=df.shape[0]
    # quantile_1perc = df['ZFY'].quantile(0.01)
    # quantile_99perc = df['ZFY'].quantile(0.99)
    # df = df[((df['los_this_time'] > 0) & (df['ZFY'] <= quantile_99perc) & (df['ZFY'] >= quantile_1perc))]
    df = df[df['los_this_time'] > 0]

    print("住院时长等于0住院记录数(删除)：", df.shape[0] - df_len2)

    # #清除离院方式不为1的住院记录
    # df_len3 = df.shape[0]
    # df=df[df['LYFS']=='1']
    # print("离院方式不为1的住院记录数(删除)：", df.shape[0] - df_len3)

    df = df.reset_index(drop=True)
    print(df.shape[0])

    # df['zfy_last_time'] = df['zfy_last_time'].apply(lambda x: 0 if x == -999 else x)
    # df['los_last_time'] = df['los_last_time'].apply(lambda x: 0 if x == -999 else x)
    # df['los_last_time'] = df['los_last_time'].apply(lambda x: 0 if x == -999 else x)
    # df['interval_thistimery_pasttimecy'] = df['interval_thistimery_pasttimecy'].apply(
    #     lambda x: 1096 if x == -999 else x)
    # df['mean_zfy_last_3y'] = df['mean_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    # df['med_zfy_last_3y'] = df['med_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    # df['max_zfy_last_3y'] = df['max_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    # df['std_zfy_last_3y'] = df['std_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    # df['min_zfy_last_3y'] = df['min_zfy_last_3y'].apply(lambda x: 0 if x == -999 else x)
    # # df['weighted_average_cost_past_3y'] = df['weighted_average_cost_past_3y'].apply(lambda x: 0 if x == -999 else x)
    # df['last_time_cost_per_day'] = df['last_time_cost_per_day'].apply(lambda x: 0 if x == -999 else x)


    # f0 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'wb')
    # pickle.dump(df, f0)
    # f0.close()
    # df.to_csv("../data/feature_related/df_Histry9_clearned2.csv")


def proc_weighted_averagr_cost_ftr():
    f2 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()

    f2 = open("../data/feature_related/df_del_R_wthout_IHDry_wth_Histry8_clearned.pkl", 'rb')
    df_hv_wt = pickle.load(f2)
    f2.close()

    df_hv_wt=df_hv_wt[['SFZH','RN','weighted_average_cost_past_3y']]
    df = df.merge(df_hv_wt, how='left', on=['SFZH','RN'])


    f0 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_Histry9_clearned2.csv")


def proc_weighted_averagr_cost_ftr2():
    f2 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()
    df2=df.copy(deep=True)

    df = df[df['weighted_average_cost_past_3y'].isnull() == True]
    print(df.shape)
    # 加权平均
    xlsxpath=  "../data/csv_xslx/AA_MULTI_DESC_TENSOR.xlsx"
    dic_icd3_dchapter, dic_clmnname_index = myfunction.chapter_name_list(xlsxpath)

    # 提取历史特征：
    df_sfzh = df[['SFZH', 'RN', 'ry_diseases_']]
    df_sfzh.columns = ['SFZH', 'RN_CASE', 'ry_diseases_']
    # print("纳入（19年）患者数（入院诊断含IHD）：",df_sfzh.shape[0])
    # 读取YP_P_16_19  # 该表存储按19年出院诊断的身份账号索引的16-19年的所有住院记录
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    cr1 = db.cursor()
    sql1 = 'select SFZH,RN,RY_DATE,ALL_DISEASE,ZFY,YYDJ_J from scott.YP_P_15_19'
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    # print(cyzd_data)   list of data
    names = [i[0] for i in cr1.description]
    # print(names)
    df_16to19 = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()

    df_16to19['diseases'] = df_16to19['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df_16to19['cy_diseases_'] = df_16to19['diseases'].apply(lambda x: [i for i in x if
                                                                       len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[
                                                                           1] >= '0' and i[1] <= '9' and i[2] >= '0' and
                                                                       i[
                                                                           2] <= '9'])
    df_16to19.drop(columns='diseases', axis=1, inplace=True)
    df_16to19.drop(columns='ALL_DISEASE', axis=1, inplace=True)

    df_sfzh = df_sfzh.merge(df_16to19, how='left', on=['SFZH'])
    records_histry = []
    time_flag = 0
    print("___________________group中____________________________")
    pastt = time.time()
    for index, df_idnumber in df_sfzh.groupby(['SFZH', 'RN_CASE']):  # 唯一标识一条记录
        time_flag += 1
        if (time_flag % 5000 == 0):
            print("time_flag-----", time_flag)
        case_sfzh = df_idnumber['SFZH'].values[0]
        case_RN = df_idnumber['RN_CASE'].values[0]
        case_ry_data = df_idnumber[df_idnumber['RN'] == case_RN]['RY_DATE'].values[0]
        case_ry_diseases = df_idnumber[df_idnumber['RN'] == case_RN]['ry_diseases_'].values[0]
        # case_dj_j = df_idnumber[df_idnumber['RN'] == case_RN]['YYDJ_J'].values[0]

        record_histry = []
        record_histry.append(case_sfzh)
        record_histry.append(case_RN)

        ry_dis_appear_hot = [0 for s in range(len(dic_clmnname_index))]  # 初始化
        for disease_ry in case_ry_diseases:
            if disease_ry in dic_icd3_dchapter:
                ry_dis_appear_hot[dic_clmnname_index[dic_icd3_dchapter[disease_ry]]] += 1
        # if case_dj_j == '2':
        #     ry_dis_appear_hot[len(dic_clmnname_index)] = 1
        # else:
        #     ry_dis_appear_hot[len(dic_clmnname_index)] = 0
        df_idnumber['admssn_time'] = df_idnumber['RY_DATE'].apply(
            lambda x: 1 if ((case_ry_data - x).days <= 1095 and (case_ry_data - x).days > 0) else 0)

        df_idnumber = df_idnumber[df_idnumber['admssn_time'] == 1]  # 前三年的住院记录
        if df_idnumber.shape[0] > 1:
            list_idnumber = df_idnumber[['cy_diseases_', 'ZFY', 'YYDJ_J']].values.tolist()
            # print(list_idnumber)
            list_sort_dist = []
            dist_sum = 0
            for last_3y_rcd in list_idnumber:
                cy_dis_appear_hot = [0 for s in range(len(dic_clmnname_index))]
                for disease_cy in last_3y_rcd[0]:
                    if disease_cy in dic_icd3_dchapter:
                        cy_dis_appear_hot[dic_clmnname_index[dic_icd3_dchapter[disease_cy]]] += 1
                # if last_3y_rcd[2] == '2':
                #     cy_dis_appear_hot[len(dic_clmnname_index)] = 1
                # else:
                #     cy_dis_appear_hot[len(dic_clmnname_index)] = 0
                o_dist = 0
                for k in range(len(dic_clmnname_index)):
                    o_dist += (cy_dis_appear_hot[k] - ry_dis_appear_hot[k]) ** 2
                o_dist = np.sqrt(o_dist)
                dist_sum += o_dist
                # print("o_dist",o_dist)
                # print("dist_sum",dist_sum)
                list_sort_dist.append((last_3y_rcd[1], o_dist))
            weighted_avg = 0
            if dist_sum != 0:
                for dist in list_sort_dist:
                    weight = (dist_sum - dist[1]) / (dist_sum * (len(list_sort_dist) - 1))
                    weighted_avg += weight * dist[0]
            else:
                weighted_avg = df_idnumber['ZFY'].mean()
            record_histry.append(weighted_avg)
        elif df_idnumber.shape[0] == 1:
            weighted_avg = df_idnumber['ZFY'].values[0]
            record_histry.append(weighted_avg)
        else:
            weighted_avg = 0
            record_histry.append(weighted_avg)
        records_histry.append(record_histry)
    print("_________________group耗时：--------------", (time.time() - pastt) / 60)
    df_histry = pd.DataFrame(records_histry, columns=['SFZH', 'RN', 'weighted_average_cost_past_3y_2'])
    del records_histry

    df2 = df2.merge(df_histry, how='left', on=['SFZH', 'RN'])

    print("aaaaaaaaaaaaaaaaaa")
    a=df2['weighted_average_cost_past_3y'].values
    b=df2['weighted_average_cost_past_3y_2'].values
    c=[]
    for idx in range(a.shape[0]):
        if isnan(a[idx])==False:
            c.append(a[idx])
        else:
            c.append(b[idx])
    df2.drop(columns='weighted_average_cost_past_3y', axis=1, inplace=True)
    df2.drop(columns='weighted_average_cost_past_3y_2', axis=1, inplace=True)
    weighted_avg_df=pd.DataFrame(c,columns=['weighted_average_cost_past_3y'])
    df2=pd.concat([df2,weighted_avg_df],axis=1)
    print("bbbbbbbbbbbbbbbbbb")

    # df2['weighted_average_cost_past_3y']=df2['weighted_average_cost_past_3y_2'].apply(lambda x:)
    # weight_avg_cost_clmn = df2.apply(lambda x: x['weighted_average_cost_past_3y_2'] if x['weighted_average_cost_past_3y'].isnull()==True else  , axis=1)


    f0 = open("../data/feature_related/df_Histry9_clearned2.pkl", 'wb')
    pickle.dump(df2, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_Histry9_clearned2.csv")

    print(df2[df2['weighted_average_cost_past_3y'].isnull() == True].shape)
    print(df2.shape)


def network_ftr_comorbidity_attri(dir_gml):
    """
    :param dir_gml: "data_wo_threshold"
    :return:
    """
    mygraph = nx.read_gml(dir_gml+"/gml_dir/CC_all_modularity_-1layer.gml")

    node_list=sorted(list(mygraph.nodes))
    dic_node_cols=dict(zip(node_list,[i for i in range(len(node_list))]))

    dic_cmbdty_attribution={}
    for begin_node in node_list:
        attribution_lst=[0 for i in range(len(node_list))]
        attribution_lst[dic_node_cols[begin_node]]=1   #初始化

        for edge in mygraph.edges:
            if edge[0]==begin_node:
                attribution_lst[dic_node_cols[edge[1]]]=mygraph.edges[edge]['weight']
            elif edge[1]==begin_node:
                attribution_lst[dic_node_cols[edge[0]]]=mygraph.edges[edge]['weight']
        dic_cmbdty_attribution[begin_node]=attribution_lst

    f0 = open(dir_gml+"/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'wb')
    pickle.dump(dic_cmbdty_attribution, f0)
    f0.close()

    return dic_cmbdty_attribution


def history_cmbdty_attri(df,flag):
    """"""
    # f2 = open("../data/feature_related/df_Histry9_clearned3.pkl", 'rb')
    # df = pickle.load(f2)
    # f2.close()

    f2 = open("../data_wo_threshold/med_data_mtx_dic/dic_cmbdty_attribution.pkl", 'rb')
    dic_cmbdty_attribution = pickle.load(f2)
    f2.close()
    lst_cmbdty_name=list(dic_cmbdty_attribution.keys())

    # 提取历史特征：
    # print(list(df.columns))
    # print("list(df.columns):",len(list(df.columns)))
    columns_left=['SFZH', 'RN', 'ry_diseases_']
    columns_left.extend(lst_cmbdty_name)
    # print(columns_left)
    # print("len(columns_left):",len(columns_left))
    df_sfzh = df[columns_left]
    columns_left_rename=['SFZH', 'RN_CASE', 'ry_diseases_']
    columns_left_rename.extend(lst_cmbdty_name)
    df_sfzh.columns = columns_left_rename
    # print("纳入（19年）患者数（入院诊断含IHD）：",df_sfzh.shape[0])
    # 读取YP_P_16_19  #该表存储按19年出院诊断的身份账号索引的16-19年的所有住院记录
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    cr1 = db.cursor()
    sql1 = 'select SFZH,RN,RY_DATE,ALL_DISEASE,ZFY,YYDJ_J from scott.YP_P_15_19'
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    # print(cyzd_data)   list of data
    names = [i[0] for i in cr1.description]
    # print(names)
    df_16to19 = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()

    df_16to19['diseases'] = df_16to19['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df_16to19['cy_diseases_'] = df_16to19['diseases'].apply(lambda x: [i for i in x if
                                                                       len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[
                                                                           1] >= '0' and i[1] <= '9' and i[2] >= '0' and
                                                                       i[
                                                                           2] <= '9'])
    df_16to19.drop(columns='diseases', axis=1, inplace=True)
    df_16to19.drop(columns='ALL_DISEASE', axis=1, inplace=True)

    df_sfzh = df_sfzh.merge(df_16to19, how='left', on=['SFZH'])
    records_histry = []
    time_flag = 0
    print("___________________group中____________________________")
    pastt = time.time()
    for index, df_idnumber in df_sfzh.groupby(['SFZH', 'RN_CASE']):  # 唯一标识一条记录
        time_flag += 1
        if (time_flag % 500 == 0):
            print("flag:",flag)
            print("time_flag-----", time_flag)
            print("已耗时：",(time.time()-pastt)/60)
        case_sfzh = df_idnumber['SFZH'].values[0]
        case_RN = df_idnumber['RN_CASE'].values[0]
        case_ry_data = df_idnumber[df_idnumber['RN'] == case_RN]['RY_DATE'].values[0]
        record_histry = []
        record_histry.append(case_sfzh)
        record_histry.append(case_RN)

        lst_case_cmbdty_attris=df_idnumber[df_idnumber['RN'] == case_RN][lst_cmbdty_name].values.tolist()[0]

        df_idnumber['admssn_time'] = df_idnumber['RY_DATE'].apply(
            lambda x: 1 if ((case_ry_data - x).days <= 1095 and (case_ry_data - x).days > 0) else 0)

        df_idnumber = df_idnumber[df_idnumber['admssn_time'] == 1]  # 前三年的住院记录
        if df_idnumber.shape[0] > 1:
            list_idnumber = df_idnumber[['cy_diseases_', 'ZFY', 'YYDJ_J']].values.tolist()
            list_sort_dist = []
            dist_sum = 0
            for last_3y_rcd in list_idnumber:
                lst_diseases_encoding = [[0 for i in range(len(dic_cmbdty_attribution))]]

                for disease_cy in last_3y_rcd[0]:
                    if disease_cy in dic_cmbdty_attribution:
                        lst_diseases_encoding.append(dic_cmbdty_attribution[disease_cy])
                df_attri = pd.DataFrame(lst_diseases_encoding, columns=list(dic_cmbdty_attribution.keys()))
                max_attri_lst = df_attri.max().values.tolist()

                o_dist = 0
                for k in range(len(dic_cmbdty_attribution)):
                    o_dist += (max_attri_lst[k] - lst_case_cmbdty_attris[k]) ** 2
                o_dist = np.sqrt(o_dist)
                dist_sum += o_dist
                list_sort_dist.append((last_3y_rcd[1], o_dist))

            weighted_avg = 0
            if dist_sum != 0:
                for dist in list_sort_dist:
                    weight = (dist_sum - dist[1]) / (dist_sum * (len(list_sort_dist) - 1))
                    weighted_avg += weight * dist[0]
            else:
                weighted_avg = df_idnumber['ZFY'].mean()
            record_histry.append(weighted_avg)
        elif df_idnumber.shape[0] == 1:
            weighted_avg = df_idnumber['ZFY'].values[0]
            record_histry.append(weighted_avg)
        else:
            weighted_avg = 0
            record_histry.append(weighted_avg)
        records_histry.append(record_histry)
    print("_________________group耗时：--------------", (time.time() - pastt) / 60)

    f0 = open("../data/feature_related/lst_similarity_weight_cost" + flag + ".pkl", 'wb')
    pickle.dump(records_histry, f0)
    f0.close()

    # df_histry = pd.DataFrame(records_histry, columns=['SFZH', 'RN', 'weighted_similarity_history_cost'])
    #
    # # del records_histry
    # df = df.merge(df_histry, how='left', on=['SFZH', 'RN'])
    #
    # f0 = open("../data/feature_related/df_similarity_weight_cost"+flag+".pkl", 'wb')
    # pickle.dump(df, f0)
    # f0.close()


def concat_df_weight_cmbdty_his_cost():
    f2 = open("../data/feature_related/lst_similarity_weight_cost1.pkl", 'rb')
    lst_similarity_weight_cost1 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/lst_similarity_weight_cost2.pkl", 'rb')
    lst_similarity_weight_cost2 = pickle.load(f2)
    f2.close()
    f2 = open("../data/feature_related/lst_similarity_weight_cost3.pkl", 'rb')
    lst_similarity_weight_cost3 = pickle.load(f2)
    f2.close()
    print(len(lst_similarity_weight_cost1))
    print(len(lst_similarity_weight_cost2))
    print(len(lst_similarity_weight_cost3))

    df_histry1 = pd.DataFrame(lst_similarity_weight_cost1, columns=['SFZH', 'RN', 'weighted_similarity_history_cost'])
    df_histry2 = pd.DataFrame(lst_similarity_weight_cost2, columns=['SFZH', 'RN', 'weighted_similarity_history_cost'])
    df_histry3 = pd.DataFrame(lst_similarity_weight_cost3, columns=['SFZH', 'RN', 'weighted_similarity_history_cost'])

    df_histry1=pd.concat([df_histry1,df_histry2,df_histry3],axis=0)

    f2 = open("../data/feature_related/df_Histry9_clearned3.pkl", 'rb')
    df = pickle.load(f2)
    f2.close()

    df = df.merge(df_histry1, how='left', on=['SFZH','RN'])

    f0 = open("../data/feature_related/df_Histry9_clearned4.pkl", 'wb')
    pickle.dump(df, f0)
    f0.close()
    df.to_csv("../data/feature_related/df_Histry9_clearned4.csv")


def get_node_col(dir_gml):
    """
    :param dir_gml: "data_wo_threshold"
    :return:
    """
    mygraph = nx.read_gml(dir_gml+"/gml_dir/CC_all_modularity_-1layer.gml")

    node_list=sorted(list(mygraph.nodes))
    dic_node_cols=dict(zip(node_list,[i for i in range(len(node_list))]))
    pickle.dump(dic_node_cols, open("../data_model/dic_sorted_node_cols.pkl", "wb"))


if __name__=="__main__":

    """提取历史特征，对建模部分的数据进行预处理"""
    get_node_col("../data_wo_threshold")
    # get_histry_feature()
    # data_PreProcessing()
    #
    # xlsxpath = "../data/csv_xslx/AA_MULTI_DESC_TENSOR.xlsx"
    # history_feature_weighted_avg2(xlsxpath)
    # processing_history_miss()
    # proc_weighted_averagr_cost_ftr2()
    # network_ftr_comorbidity_attri("../data_wo_threshold")
    # data_PreProcessing2()

    # concat_df_weight_cmbdty_his_cost()

    # # # 多进程处理
    # f2 = open("../data/feature_related/df_Histry9_clearned3.pkl", 'rb')
    # df = pickle.load(f2)
    # f2.close()
    # #
    # len_df=df.shape[0]
    # print(len_df)
    # # df1 = df[df.index < 100]
    # df1=df[df.index<int(1*len_df/3)]
    # df2=df[(df.index<int(2*len_df/3))&(df.index>=int(1*len_df/3))]
    # # df3=df[(df.index<int(3*len_df/4))&(df.index>=int(2*len_df/4))]
    # df3=df[df.index>=int(2*len_df/3)]
    # #
    # print(df1.shape)
    # print(df2.shape)
    # print(df3.shape)
    # # print(df4.shape)
    # # (80976, 25)
    # # (80977, 25)
    # # (80977, 25)
    # # (80977, 25)
    # #
    # p1 = multiprocessing.Process(target=history_cmbdty_attri, args=(df1,"1"))  # 创建一个进程，args传参 必须是元组
    # p1.start()  # 运行线程p
    # p2 = multiprocessing.Process(target=history_cmbdty_attri, args=(df2,"2"))  # 创建一个进程，args传参 必须是元组
    # p2.start()  # 运行线程p
    # p3 = multiprocessing.Process(target=history_cmbdty_attri, args=(df3,"3"))  # 创建一个进程，args传参 必须是元组
    # p3.start()  # 运行线程p
    # # p4 = multiprocessing.Process(target=history_cmbdty_attri, args=(df4, "4"))  # 创建一个进程，args传参 必须是元组
    # # p4.start()  # 运行线程p

import cx_Oracle
import pandas as pd
import numpy as np
import time
from scipy.sparse import *
import pickle
import os
import math
import igraph
from igraph import *
from tqdm import tqdm

def get_sex_chronic_disease(chronic,sex):
    #对疾病进行筛选，得到符合条件的疾病 慢性和性别特异性疾病 sex=1 代表要考虑性别特异性疾病
    """两个字典 chronic=1 返回慢病字典 sex=1 返回性别特异性字典  1代表男性 2代表女性 0代表不是性别特异性"""
    df=pd.read_csv("data3/csv_xslx/dis_sex_chronic.csv",encoding="gbk")
        #大于0 是宽松的条件， 等于1 是严格的条件，我们暂定宽松的条件
    if(chronic==1  and sex==0):#性别的1是男性 2代码女性特异性
        df_ans=df[ df.chronic>0]
        dic={}
        for i in df_ans["dis"].values:
            dic[i]=1
    else:
        df_ans=df[ df.SexDisease==1]#男性特异性
        dic={}
        for i in df_ans["dis"].values:
            dic[i]=1

        df_ans=df[ df.SexDisease==2]#女性特异性
        for i in df_ans["dis"].values:
            dic[i]=2

    return dic# 能够查询代表满足要求


def construct_sparse_matrix(df_sfzh_diseases,save_dir):
    """
    输入参数df[['SFZH','diseases']]
    该函数输出的结果：字典：dic_cols,dic_rows；
    存储pkl文件 稀疏矩阵：patient_disease_csr_matrix
    功能：根据从数据库读入的数据生成患者-慢病稀疏矩阵，（一人多条记录合并为一条记录）
    """
    identity=df_sfzh_diseases['SFZH'].drop_duplicates()
    identity=sorted(identity)
    #身份证号-行标 字典
    dic_rows=dict(i for i in zip(identity,range(len(identity))))

    disease = set()
    pastt=time.time()
    print("开始生成所有疾病的集合-------------------")
    for diesease_set in df_sfzh_diseases['diseases']:
        disease=disease.union(diesease_set)  #所有疾病的集合，去重后
    print("未除去急性病时，疾病种类数：",len(disease))
    dic = get_sex_chronic_disease(1, 0)  # 得到慢病字典
    disease.intersection_update(dic)  #移除disease中不属于dic的元素 ，得到的是慢病集合
    print("除去急性病后，疾病种类数：", len(disease))
    print("生成所有慢病疾病的集合 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))

    disease=sorted(disease)
    #所有纳入患者中存在的慢病-列标 字典
    dic_cols=dict(j for j in zip(disease,range(len(disease))))

    row=[]
    col=[]
    print("开始生成稀疏矩阵的row,col列表")
    pastt2=time.time()
    for sfzh, diseases in df_sfzh_diseases.groupby('SFZH'):
        sfzh_row_value=dic_rows[sfzh]
        for d_set in diseases['diseases']:
            for d in d_set:
                if dic_cols.get(d, -1) != -1:
                    row.append(sfzh_row_value)
                    col.append(dic_cols[d])
    print("生成稀疏矩阵的row,col 耗时：%.3f 分钟"%((time.time()-pastt2)/60))

    #对(row,col)去重
    df_row_col=pd.DataFrame({'row':row,'col':col}).drop_duplicates()
    # print(df_row_col)
    row=df_row_col.iloc[:,0].values
    col = df_row_col.iloc[:,1].values   #ndarray类型

    print("创建稀疏矩阵中-------------------------")
    pastt3=time.time()
    #创建稀疏矩阵，用scipy.sparse中的coo_matrix
    data=np.ones(len(df_row_col),dtype=np.int8)
    patient_disease_coo_matrix=coo_matrix((data,(row,col)),shape=(len(dic_rows),len(dic_cols)),dtype=np.int32)

    patient_disease_csr_matrix=patient_disease_coo_matrix.tocsr()
    # patient_disease_csc_matrix=patient_disease_coo_matrix.tocsc()

    print("共慢病稀疏矩阵的维度为：",patient_disease_csr_matrix.shape)
    print("数据保存中-----------------------------")
    f1=open(save_dir+"/med_data_mtx_dic/patient_disease_csr_matrix.pkl", 'wb')
    pickle.dump(patient_disease_csr_matrix,f1)
    f1.close()
    f4=open(save_dir+"/med_data_mtx_dic/dic_cols.pkl",'wb')
    pickle.dump(dic_cols,f4)
    f4.close()
    f5=open(save_dir+"/med_data_mtx_dic/dic_rows.pkl", 'wb')
    pickle.dump(dic_rows, f5)
    f5.close()
    print("创建稀疏矩阵，保存稀疏矩阵 耗时：%.3f 分钟"%((time.time()-pastt3)/60))


def choose_petient_based_on_matrix(save_dir,flag):
    """
    功能：剔除共慢病数少于2的病人
    _flag:  0 : 剔除； 1 ： 不剔除；
    需要读入的文件：patient_disease_csr_matrix，dic_rows  #患者-慢病矩阵；患者-编号字典
    输出文件：dic_rows__disease_count_more_than_one.pkl 包含所有慢病数大于等于2的患者的身份证号和矩阵行标的对应信息
    返回值：_less_than2_sfzh  存储慢病小于2的病人的身份证号
    """
    print("剔除不满足的患者（共慢病数少于2）--------------------------")
    pastt1 = time.time()
    f1 = open(save_dir+"/med_data_mtx_dic/patient_disease_csr_matrix.pkl", 'rb')
    patient_disease_csr_matrix=pickle.load(f1)
    row_len,row_width=patient_disease_csr_matrix.shape
    f1.close()
    print("剔除共慢病数少于2的患者前，矩阵的维度：", patient_disease_csr_matrix.shape)
    f2 = open(save_dir+"/med_data_mtx_dic/dic_rows.pkl", 'rb')
    dic_rows = pickle.load(f2)
    f2.close()

    sfzh_list = list(dic_rows.keys())
    sumofdisease = patient_disease_csr_matrix.sum(axis=1).squeeze().tolist()  # 求行和（每个人患病数）
    count_disease = sumofdisease[0]  # 第i个人有count_disease[i]种共慢病

    less_than2_index=[]  # 存储行和小于2的行标（慢病小于2的病人索引）
    less_than2_sfzh=[]  # 存储行和小于2的身份证号（慢病小于2的病人）
    if flag==0:  # flag==0时，才做剔除这步操作
        for i in range(len(count_disease)):
            if count_disease[i]<2:
                less_than2_index.append(i)

    if len(less_than2_index)!=0:  # 若存在共慢病数少于2个的患者，则剔除，更新稀疏矩阵，更新dic_rows
        for i in less_than2_index:
            less_than2_sfzh.append(sfzh_list[i])
            del dic_rows[sfzh_list[i]]  # 删除共慢病小于2的患者

    new_sfzh_list=list(dic_rows.keys())
    patient_disease_csr_matrix=patient_disease_csr_matrix[[dic_rows[m] for m in new_sfzh_list],:]  # 生成新的csr矩阵
    dic_rows_new = dict(i for i in zip(new_sfzh_list, range(len(new_sfzh_list))))   # 更新身份证号-行标字典
    print("剔除共慢病数少于2的患者后，矩阵的维度：",patient_disease_csr_matrix.shape)
    f3 = open(save_dir+"/med_data_mtx_dic/dic_rows_new.pkl", 'wb')
    pickle.dump(dic_rows_new, f3)
    f3.close()
    f4 = open(save_dir+"/med_data_mtx_dic/csr_mtrx_clr_row.pkl", 'wb')
    pickle.dump(patient_disease_csr_matrix, f4)
    f4.close()
        # del_coo_mtx_rols(less_than2_index,row_len,row_width)
    print("剔除共慢病数少于2的患者 耗时：%.3f 分钟" % ((time.time() - pastt1) / 60))
    # print(less_than2_sfzh)
    return less_than2_sfzh  # 存储慢病小于2的病人的身份证号，list


def choose_diseases_based_on_matrix(num_male,num_female,flag,prevalence_threshold,save_dir):
    """
    处理稀疏矩阵的列，去除流行率小于1%的疾病
    输入参数：
        _num_male,nem_female为纳入患者中的男女人数，由函数construct_sex_count_dict()可计算得到
        _flag： flag为1，则去除I20-I25，共病网络不考虑这六种疾病 ; flag=0,则将I20-I25纳入共病网络
        _prevalence_threshold:选取的疾病的流行率下限
        _save_dir:读取及保存文件的文件夹名字
    输出文件：csc_matrix_final.pkl 去除了流行率小于1%的列后的新矩阵；dic_cols_new.pkl 新的疾病-列映射
    返回值：_dic_disease_prevalence   返回疾病流行度字典，与稀疏矩阵的列顺序对应
    note:该函数中，将csr矩阵转换为了csc矩阵(多余...)
    """
    f1 = open(save_dir+"/med_data_mtx_dic/csr_mtrx_clr_row.pkl", 'rb')
    patient_disease_csr_matrix = pickle.load(f1)
    f1.close()
    f2 = open(save_dir+"/med_data_mtx_dic/dic_cols.pkl", 'rb')
    dic_cols = pickle.load(f2)
    # print('dic_cols长度',len(dic_cols))
    f2.close()
    print("csr->csc matrix--------------------------------")
    patient_disease_csc_matrix=patient_disease_csr_matrix.tocsc()  #转变为按列存储
    print("去除流行率小于1%的疾病前，稀疏矩阵的维度：", patient_disease_csc_matrix.shape)
    print("除去流行程度小于1%的疾病-----------------------")
    pastt=time.time()

    if flag==1:
        # ###去除I20-I25
        if 'I20' in dic_cols:
            del dic_cols['I20']
        if 'I21' in dic_cols:
            del dic_cols['I21']
        if 'I22' in dic_cols:
            del dic_cols['I22']
        if 'I23' in dic_cols:
            del dic_cols['I23']
        if 'I24' in dic_cols:
            del dic_cols['I24']
        if 'I25' in dic_cols:
            del dic_cols['I25']

    sex_disease_dic=get_sex_chronic_disease(0,1)  # 获得性别特异性疾病的字典 ; 1:男性 2：女性
    # 处理性别特异性疾病，计算prevalence_rate时，计入的总人数是不同的
    col_names_new=[]
    dic_disease_prevalence={}
    dic_disease_prevalence_rate={}
    for key in dic_cols.keys():
        num_patient=num_female+num_male
        if key in sex_disease_dic:
            if sex_disease_dic[key]==1:
                num_patient=num_male
            else:
                num_patient=num_female
        col_index=dic_cols[key]

        prevalence_rate=patient_disease_csc_matrix[:,col_index].sum()/num_patient
        # print("prevalence:",prevalence_rate)
        if (prevalence_rate>prevalence_threshold):
            col_names_new.append(key)
            dic_disease_prevalence[key]=patient_disease_csc_matrix[:,col_index].sum()
            dic_disease_prevalence_rate[key]=prevalence_rate
    # print(dic_disease_prevalence_rate)

    csc_matrix_final=patient_disease_csc_matrix[:,[dic_cols[i] for i in col_names_new]]  # 生成新的稀疏矩阵，去除了流行率小于1%的疾病后
    print("去除流行率小于1%的疾病后，稀疏矩阵的维度：",csc_matrix_final.shape)
    dic_cols=dict([i for i in zip(col_names_new,range(len(col_names_new)))]) # 新的疾病-列映射
    print(" 除去流行程度小于0.01的疾病 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))

    preva_df = pd.DataFrame(dic_disease_prevalence_rate, index=[1])
    preva_df.sort_values(by=1, axis=1, ascending=False,inplace=True)
    preva_df=preva_df.iloc[:,0:20]
    print("流行率前20的疾病：")
    for i in preva_df:
        print(i, preva_df.loc[1][i])

    f3 = open(save_dir+"/med_data_mtx_dic/csc_matrix_final.pkl", 'wb')
    pickle.dump(csc_matrix_final, f3)
    f3.close()
    f4 = open(save_dir+"/med_data_mtx_dic/dic_cols_new.pkl", 'wb')
    pickle.dump(dic_cols, f4)
    f4.close()
    f5 = open(save_dir+"/med_data_mtx_dic/dic_disease_prevalence_rate.pkl", 'wb')
    pickle.dump(dic_disease_prevalence_rate, f5)
    f5.close()
    return  dic_disease_prevalence   # 返回疾病流行度字典，与稀疏矩阵的列顺序对应


def construct_sex_count_dict_2(df,less_than2_sfzh):
    """
    输入参数：df[['sfzh','xb']]，dic_rows字典,
    功能：统计纳入患者的男女人数，以字典形式输出；生成与dic_rows中身份证号相映射的性别列表
    返回值：xb：性别列表，与dic_rows中身份证号顺序相对应，1为男性，2为女性；
    返回值：sex_count_dict：男女人数字典，形如{'male': 87, 'female': 111}
    方法：不同于construct_sex_count_dict_2处：挑出前面筛选人时被剔除了的sfzh,从原DF中去除，然后对身份证号进行排序
    """
    print('construct_sex_count_dict统计中------------')
    pastt=time.time()
    # xb=[]
    df=df.drop_duplicates()
    print("计算男女人数中------，未剔除人前的总人数：",df.shape[0])
    for sfzh in tqdm(less_than2_sfzh):
        df = df[~(df['SFZH'] == sfzh)]
    sex_df=df.sort_values(by="SFZH",axis=0,ascending=True)
    sex_count_dict={'male':0,'female':0}
    for i in sex_df['XB']:
        # xb.append(i)
        if i == '1':
            sex_count_dict['male']+=1
        else:
            sex_count_dict['female']+=1
    print(" 性别统计 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))
    print('男性患者人数：',sex_count_dict['male'],'女性患者人数',sex_count_dict['female'])
    return sex_count_dict


def compute_Cij(dic_cols,csc_matrix_final,save_dir):
    """
    计算Cij矩阵,是一个上三角矩阵，同时患i和j两种疾病的人数，主对称轴元素均为0
    输入参数：dic_cols:疾病-列表映射；
    输入参数：csc_matrix_final:慢病稀疏矩阵
    返回值：Cij矩阵
    """
    print("开始计算Cij--------------------------")
    pastt=time.time()
    Cij=np.zeros((len(dic_cols),len(dic_cols)))
    for i in tqdm(range(len(dic_cols))):
        for j in range(i+1,len(dic_cols)):
            cij=0
            two_cols_sum=(csc_matrix_final[:,i]+csc_matrix_final[:,j])
            for s in two_cols_sum.data:
                if s==2:  #一个人同时患两种病
                    cij+=1
            Cij[i][j]=cij
            # Cij[j][i]=cij
            #还是用稀疏矩阵存储？？暂时不用
    # print(Cij)
    print(" 生成Cij矩阵 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))
    f1 = open(save_dir+"/med_data_mtx_dic/Cij.pkl", 'wb')
    pickle.dump(Cij, f1)
    f1.close()
    return Cij


def compute_RR_CI(Cij,prevalence,N,dic_cols,save_dir):
    """
    计算RR值及其置信区间 ,99%的置信区间，置信区间不包含1，则有意义
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间,同时患两种疾病的人数],....]
    """
    Cij_num=0
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            if Cij[i][j]!=0:
                Cij_num+=1
                prevalence1 = prevalence[list_cols_name[i]]
                prevalence2 = prevalence[list_cols_name[j]]
                RR_ij=(Cij[i][j]/prevalence1)*(N/prevalence2)
                if RR_ij<0:
                    print("#############RR 溢出啦###########################################",RR_ij)

                Sigma=1/Cij[i][j]+(1/prevalence1)*(1/prevalence2)-1/N-(1/N)*(1/N)   #会产生除零错误，所以应该在计算前判断Cij是否为零；（Cij为零时，RR值也为零）
                low=RR_ij*np.exp(-1*2.56*Sigma)
                high=RR_ij*np.exp(2.56*Sigma)
                if(RR_ij>1 and low>1):  #这里只考虑了两个节点联系比随机情况下更强的情况
                    edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,RR_ij,low,high,Cij[i][j]])
                    #上面一行：添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间，同时患两种病的人数],....]
    print("不为零的Cij的个数：",Cij_num)
    print("RR，有意义的边数：", len(edge_list))
    f1 = open(save_dir+"/med_data_mtx_dic/edge_list_RR.pkl", 'wb')
    pickle.dump(edge_list, f1)
    f1.close()
    return edge_list


def compute_phi_significated(Cij,prevalence,N,dic_cols,save_dir):
    """
    计算phi值及t值 ,99%的置信水平
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]
    """
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            prevalence1=prevalence[list_cols_name[i]]
            prevalence2 = prevalence[list_cols_name[j]]
            a = (0.1*prevalence1) * (0.1*prevalence2) * (0.1*(N - prevalence1))*(0.1*(N - prevalence2))
            # a = prevalence1 * prevalence2 * (N - prevalence1) * (N - prevalence2)
            if (a) <= 0:
                print("##############################phi 溢出啦#######################",a)
            phi_ij=((0.1*Cij[i][j])*(0.1*N)-(0.1*prevalence1)*(0.1*prevalence2))/np.sqrt(a)
            t=0  #初始化t
            n=0
            if abs(phi_ij) < 1:  # phi=1时，会发生除零错误,|phi|>1时，会发生计算错误
                n = max(prevalence1, prevalence2)
                # n=N     # 注意测试一下
                t = (phi_ij * math.sqrt(n - 2)) / np.sqrt(1 - (phi_ij ** 2))
            elif phi_ij>1 or phi_ij<-1: # 不会大于1
                print("###############有phi大于1 或者小于-1 ，考虑截断,phi值为：################",phi_ij)
                # 若phi=1，只能是这种情况：A病和B病必定同时出现，且A病和B病不单独出现，这时的phi=1；因为前面步骤去除了流行度小于1%的疾病，所以这种情况基本不会发生吧
                t=0
            else:
                t=2.77
                n = max(prevalence1, prevalence2)
                print("###############有phi等于-1、1 ，n = max(prevalence1, prevalence2)值为：################", n)
            if ((n>1000 and phi_ij>0 and t>=2.58) or (n>500 and phi_ij>0 and t>=2.59) or (n>200 and phi_ij>0 and t>=2.60) or (n>90 and phi_ij>0 and t>=2.63) or (n>80 and phi_ij>0 and t>=2.64) or (n>70 and phi_ij>0 and t>=2.65) or (n>60 and phi_ij>0 and t>=2.66) or (n>50 and phi_ij>0 and t>=2.68) or (n>40 and phi_ij>0 and t>=2.70) or (n>38 and phi_ij>0 and t>=2.71) or (n>35 and phi_ij>0 and t>=2.72) or (n>33 and phi_ij>0 and t>=2.73) or (n>31 and phi_ij>0 and t>=2.74) or (n>30 and phi_ij>0 and t>=2.75) or (n>28 and phi_ij>0 and t>=2.76) or (n>27 and phi_ij>0 and t>=2.77) ):#这里只考虑了两个节点联系比随机情况下更强的情况
                edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,phi_ij,t,-999,Cij[i][j]])
                # 添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]

    print("phi，有意义的边数：",len(edge_list))
    # print("########n选N,还是选max(prevalence1, prevalence2)")
    f1 = open(save_dir+"/med_data_mtx_dic/edge_list_phi.pkl", 'wb')
    pickle.dump(edge_list, f1)
    f1.close()
    return edge_list


def compute_CCxy_significated(Cij,prevalence,N,dic_cols,save_dir):
    '''
    计算CCxy值及t值 ,99%的置信水平
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的CCxy值，t值，无意义位，同时患两种病的人数],....]
    '''
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            prevalence1=prevalence[list_cols_name[i]]
            prevalence2 = prevalence[list_cols_name[j]]
            CCxy=(0.1*Cij[i][j]*math.sqrt(2))/math.sqrt(((0.1*prevalence1)**2)+((0.1*prevalence2)**2))

            t=0  #初始化t
            n=0
            if CCxy < 0:
                print("###############有CCxy溢出啦#################", CCxy)
                t = 0
            elif CCxy < 1:  # CCxy=1时，会发生除零错误,|CCxy|>1时，会发生计算错误
                n = max(prevalence1, prevalence2)
                # n=N
                if n>1:
                    t = (CCxy * math.sqrt(n - 2)) / math.sqrt(1 - (CCxy ** 2))
                else:
                    t=0
            elif CCxy == 1:
                #若CCxy=1，只能是这种情况：对任何一个人，必定同时患A病和B病，且A病和B病不单独出现，这时的CCxy=1；因为前面步骤去除了流行度小于1%的疾病，所以这种情况基本不会发生吧
                t=0
                n = max(prevalence1, prevalence2)
                print("###############有CCxy等于1", "n= max(prevalence1, prevalence2)值为：", n)
            else:
                print("###############有CCxy大于等于1？", "CCxy值为：#################", CCxy)
            if ((n>1000 and t>=2.58) or (n>500 and t>=2.59) or (n>200 and t>=2.60) or (n>90 and t>=2.63) or (n>80 and t>=2.64) or (n>70  and t>=2.65) or (n>60 and t>=2.66) or (n>50 and t>=2.68) or (n>40 and t>=2.70) or (n>38 and t>=2.71) or (n>35 and t>=2.72) or (n>33 and t>=2.73) or (n>31 and t>=2.74) or (n>30 and t>=2.75) or (n>28 and t>=2.76) or (n>27 and t>=2.77)
                    or (n>26 and t>=2.78) or (n>25 and t>=2.79) or (n>24 and t>=2.80) or (n>23 and t>=2.81) or (n>22 and t>=2.82) or (n>21  and t>=2.83) or (n>20 and t>=2.85) or (n>19 and t>=2.86) or (n>18 and t>=2.88) or (n>17 and t>=2.90) or (n>16 and t>=2.92) or (n>15 and t>=2.95) or (n>14 and t>=2.98) or (n>13 and t>=3.01) or (n>12 and t>=3.06) or (n>11 and t>=3.11) or (n>10 and t>=3.17) or (n>9 and t>=3.25) or (n>8 and t>=3.36) or (n>7 and t>=3.50) or (n>6 and t>=3.71) or (n>5 and t>=4.03) or (n>4 and t>=4.60) or (n>3 and t>=5.84) or (n>2 and t>=9.93) or (n>1 and t>=63.66)):#这里只考虑了两个节点联系比随机情况下更强的情况
                edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,CCxy,t,-999,Cij[i][j]])
                #添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]
    print("CCxy，有意义的边数:",len(edge_list))
    f1 = open(save_dir+"/med_data_mtx_dic/edge_list_CC.pkl", 'wb')
    pickle.dump(edge_list, f1)
    f1.close()
    return edge_list



def read_raw_data():
    print("读取数据中----------")
    pastt=time.time()
    #打开数据库连接
    db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    #操作游标
    cr=db.cursor()
    sql='select sfzh,xb,All_disease from scott.YP_3YEARS_Principal'  #读2015-2017的所有数据，直接从数据库读
    cr.execute(sql)
    table_data=cr.fetchall()
    print(type(table_data))
    names=[i[0] for i in cr.description]
    print(names)
    df=pd.DataFrame(table_data,columns=names)
    cr.close()
    db.close()

    #去除异常编码
    df['diseases']=df['ALL_DISEASE'].apply(lambda x: set([i for i in x.split(',') if len(i)>2 and i[0]>='A' and i[0]<='Z' and i[1]>='0' and i[1]<='9' and i[2]>='0' and i[2]<='9']))
    df.drop(columns=['ALL_DISEASE'],axis=1,inplace=True)
    # print(df['diseases'])  #Index(['SFZH', 'XB', 'NL', 'diseases'], dtype='object')   #diseases是一个集合

    print("从数据库读取数据，处理ALL_DISEASE字段 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))
    return df

def construct_network_graph(type,percentile,save_dir):
    '''功能：构网构图
    输入参数：percentile 分位数，只画出相关系数在percentile以上的边
    输入参数：type: type='RR': RR；type='phi':phi；type='CC':CC
    '''
    f2 = open(save_dir+"/med_data_mtx_dic/dic_disease_prevalence_rate.pkl", 'rb')
    dic_disease_prevalence_rate = pickle.load(f2)
    f2.close()

    # node_name_list = sorted(list(dic_disease_prevalence_rate.keys()))

    if type=='RR':
        # edge_list_RR的结构：[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间, 同时患两种疾病的人数], ....]
        f1 = open(save_dir+"/med_data_mtx_dic/edge_list_RR.pkl", 'rb')
        edge_list_RR = pickle.load(f1)
        f1.close()
        print("RR the num of edge:",len(edge_list_RR))
        node_set=set()

        quantile_value=pd.Series([edge[4] for edge in edge_list_RR]).quantile(percentile)
        edge_list_RR=[i for i in edge_list_RR if i[4]>=quantile_value]

        for edge in edge_list_RR:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list=sorted(list(node_set))  #节点名称，排序后
        print("RR the num of node:", len(node_name_list))
        prevalence_rate=[dic_disease_prevalence_rate[i] for i in node_name_list]  #节点对应的流行率

        g=Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name']=node_name_list
        g.vs['label']=node_name_list
        g.vs['prevalence']=prevalence_rate
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
        g.write(save_dir+"/gml_dir/RR_Graph_all.gml","gml")
        # plot(g)

    if type=='phi':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数], ....]
        f1 = open(save_dir+"/med_data_mtx_dic/edge_list_phi.pkl", 'rb')
        edge_list_phi = pickle.load(f1)
        f1.close()
        print("phi the num of edge:",len(edge_list_phi))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list_phi]).quantile(percentile)
        edge_list_phi = [i for i in edge_list_phi if i[4] >= quantile_value]

        for edge in edge_list_phi:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        print("phi the num of node:", len(node_name_list))
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        g.add_edges((edge[0], edge[1]) for edge in edge_list_phi)
        phi_list = [0 for j in range(len(edge_list_phi))]
        for edge in edge_list_phi:
            edge_id = g.get_eid(edge[0], edge[1])
            phi_list[edge_id] = edge[4]
        g.es['weight'] = phi_list
        print(summary(g))
        g.write(save_dir+"/gml_dir/phi_Graph_all.gml", "gml")
        # plot(g)

    if type=='CC':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的CC值，t值，无意义位，同时患两种病的人数], ....]
        f1 = open(save_dir+"/med_data_mtx_dic/edge_list_CC.pkl", 'rb')
        edge_list_CC = pickle.load(f1)
        f1.close()
        print("CC the num of edge:",len(edge_list_CC))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list_CC]).quantile(percentile)
        edge_list_CC = [i for i in edge_list_CC if i[4] >= quantile_value]

        for edge in edge_list_CC:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        print("CC the num of node:",len(node_name_list))
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        g.add_edges((edge[0], edge[1]) for edge in edge_list_CC)
        CC_list = [0 for j in range(len(edge_list_CC))]
        for edge in edge_list_CC:
            edge_id = g.get_eid(edge[0], edge[1])
            CC_list[edge_id] = edge[4]
        g.es['weight'] = CC_list
        print(summary(g))
        g.write(save_dir+"/gml_dir/CC_Graph_all.gml", "gml")



if __name__=="__main__":

    dir='data_wo_threshold'

    # 从数据库读取数据
    df=read_raw_data()  # 读取的数据表名称需要在函数read_raw_data()中去修改
    construct_sparse_matrix(df[['SFZH','diseases']],dir)

    # 剔除共病数小于2的患者
    # flag=0 剔除；flag=1 不剔除
    flag=1
    less_than2_sfzh = choose_petient_based_on_matrix(dir,flag)

    # 统计男女人数
    sex_count_dict = construct_sex_count_dict_2(df[['SFZH', 'XB']],less_than2_sfzh)  # 计算男女人数

    # 去除流行度小于1%的疾病
    prevalence_threshold=0
    with_out_IHD=0
    dic_prevalence=choose_diseases_based_on_matrix(sex_count_dict['male'],sex_count_dict['female'],with_out_IHD,prevalence_threshold,dir)  #去除流行程度小于1%的疾病，生成患者-慢病稀疏矩阵

    # 生成共慢病稀疏矩阵，计算Cij
    f2=open(dir+"/med_data_mtx_dic/dic_cols_new.pkl", 'rb')
    dic_cols_new = pickle.load(f2)
    f2.close()
    f3 = open(dir+"/med_data_mtx_dic/csc_matrix_final.pkl", 'rb')
    csc_matrix_final = pickle.load(f3)
    f3.close()
    Cij=compute_Cij(dic_cols_new,csc_matrix_final,dir)

    # 计算RR,选取有意义的边
    # compute_RR_CI(Cij,dic_prevalence,(sex_count_dict['male']+sex_count_dict['female']),dic_cols_new,dir)
    # compute_phi_significated(Cij,dic_prevalence,(sex_count_dict['male']+sex_count_dict['female']),dic_cols_new,dir)
    compute_CCxy_significated(Cij,dic_prevalence,(sex_count_dict['male']+sex_count_dict['female']),dic_cols_new,dir)

    #前面为计算网络中的系数
    #构网，画图
    # construct_network_graph('RR',0,dir)
    # construct_network_graph('phi', 0,dir)
    construct_network_graph('CC', 0,dir)
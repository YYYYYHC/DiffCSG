import os
import pdb
import pandas as pd
    # import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    # import os
def extract_times():
    param_type = 'hyperparam'
    scale_value = '0.2'
    iternum = '5000'
    file_path = f'/home/yyyyyhc/nvdiffrast/exp/benchmark_test_{param_type}_{scale_value}_{iternum}/res.csv'
    save_path = f'/home/yyyyyhc/nvdiffrast/times_save/benchmark_test_{param_type}_{scale_value}_{iternum}.csv'
    with open(file_path, 'r') as f:
        records = f.read()

    record_lt = records.split('result:')
    res = {'name':[],'time':[],'scale':[],'param_type':[],'iternum':[]}
    for record in record_lt:
        record_items = record.split(',')
        res['name'].append(record_items[-2])
        res['time'].append(record_items[-1].replace('\n',''))
        res['scale'].append(scale_value)
        res['param_type'].append(param_type)
        res['iternum'].append(iternum)
    df = pd.DataFrame(res)

    # 保存DataFrame到CSV文件
    df.to_csv(save_path, index=False)  # 设置index=False来避免在CSV中添加行索引

    # pdb.set_trace()
def ana_times():


    # 设定文件夹路径
    folder_path = '/home/yyyyyhc/nvdiffrast/times_save'

    # 列出文件夹中所有的文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 初始化一个空的DataFrame来存储 'time' 列的数据
    combined_time_data = pd.DataFrame()

    # 读取每个文件，并提取 'time' 列
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df.loc[df['time'] > 1000, 'time'] = 900
        
        df['time'] = np.round(np.log(df['time'])).astype(int)
        # 假设 'time' 列是你感兴趣的列
        combined_time_data = pd.concat([combined_time_data, df[['time']]], ignore_index=True)

    count_all = len(combined_time_data)
    count_1 = (combined_time_data<np.log(10)).sum()
    count_2 = ((combined_time_data>=np.log(10)) & (combined_time_data<np.log(100))).sum()
    count_3 = ((combined_time_data>=np.log(100)) & (combined_time_data<np.log(1000))).sum()
    
    print(count_1/count_all, count_2/count_all, count_3/count_all)
    pdb.set_trace()
    # 绘制直方图
    # pdb.set_trace()
    plt.hist(combined_time_data['time'], bins=[0,np.log(10),np.log(100),np.log(1000)], alpha=0.75)  # bins 参数控制直方图的条形数
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time')

    # 保存图像到文件
    plt.savefig('time_distribution_histogram.png')

    # 关闭图像，释放资源
    plt.close()

    
    
if __name__ == '__main__':
    # extract_times()
    ana_times()
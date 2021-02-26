import pandas as pd


def read_objective_data(obj_data_path=r'..\客观图像质量指标测定-李展20210218.xlsx'):
    """读取客观图像质量指标测定

    受限于原 Excel 文件的存储排列方式，有些地方不便于扩展
    """
    obj_data = pd.read_excel(obj_data_path, sheet_name=None, header=None)  

    # 合并多个 Sheet
    sheet_keys = list(obj_data.keys())
    pd_obj_data = pd.DataFrame()
    column_keys = []
    for i in sheet_keys:
        sheet1 = obj_data[i]
        sheet1 = sheet1.drop([6], axis=1)      # magic number
        if "morning1" == i:
            sheet1 = sheet1.drop([0], axis=0)  # remove the comments 
        sheet1 = sheet1.dropna()  
        # reset row indexs
        sheet1 = sheet1.reset_index(drop=True)
        if "morning2" == i:
            column_keys = sheet1.values[0]     # magic number
        sheet1 = sheet1.drop([0], axis=0)   
        pd_obj_data = pd_obj_data.append(sheet1, ignore_index=True)

    # add columns indexs  
    column_keys[0] = "IMG_ID"
    pd_obj_data.columns = column_keys 

    print(obj_data_path)
    print(pd_obj_data)
    return pd_obj_data


def read_sbjective_data(sbj_data_path=r'..\志清雪清主观数据标定及加和分析20210218.xlsx'):
    """读取主观图像质量指标测定

    受限于原 Excel 文件的存储排列方式，有些地方不便于扩展
    """
    origin_envir_data = pd.read_excel(sbj_data_path, sheet_name=0, skiprows=[0], usecols="B,W,X")  
    sbj_sum_data = pd.read_excel(sbj_data_path, sheet_name=1, skiprows=[0], usecols="A:O")

    # 缺失值处理，'同时刻三张图片（1、2、3）的L列求和'...
    fill_columns = [-3, -2, -1]
    for i in fill_columns:  # 多列切片会使 inplace 失效
        sbj_sum_data.iloc[:, i].fillna(method='backfill', inplace=True)

    pd_sbj_data = pd.merge(origin_envir_data, sbj_sum_data, on='IMG_ID',  how='outer', suffixes=['_L', '_R'])
    
    print(sbj_data_path)
    print(pd_sbj_data)
    return pd_sbj_data
 

class ExcelMapper:

    def __init__(self, filename):
        self.f = filename
        self.data = read_objective_data(filename)
    
    def get_data(self):
        return self.data

    def get_PM2_5(self, img_id):
        pd_ob_data = self.data
        loc = pd_ob_data['IMG_ID'].str.contains(img_id)
        if loc.any():
            row_data = pd_ob_data[loc]['PM2.5']
            return row_data.values[0]
        else:
            return -1


if __name__ == '__main__':
    data_obj = read_objective_data()
    writer1 = pd.ExcelWriter('temp_obj.xlsx')
    data_obj.to_excel(writer1, float_format='%.5f')
    writer1.save()

    data_sbj = read_sbjective_data()
    writer2 = pd.ExcelWriter('temp_sbj.xlsx')
    data_sbj.to_excel(writer2, float_format='%.5f')
    writer2.save()
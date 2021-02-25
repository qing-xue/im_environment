import pandas as pd


def read_objective_data(ob_data_path=r'..\客观图像质量指标测定-李展20210218.xlsx'):
    """读取客观图像质量指标测定

    受限于原 Excel 文件的存储排列方式，有些地方不便于扩展
    """
    ob_data = pd.read_excel(ob_data_path, sheet_name=None, header=None)  

    # 合并多个 Sheet
    sheet_keys = list(ob_data.keys())
    pd_ob_data = pd.DataFrame()
    column_keys = []
    for i in sheet_keys:
        sheet1 = ob_data[i]
        sheet1 = sheet1.drop([6], axis=1)      # magic number
        if "morning1" == i:
            sheet1 = sheet1.drop([0], axis=0)  # remove the comments 
        sheet1 = sheet1.dropna()  
        # reset row indexs
        sheet1 = sheet1.reset_index(drop=True)
        if "morning2" == i:
            column_keys = sheet1.values[0]     # magic number
        sheet1 = sheet1.drop([0], axis=0)   
        pd_ob_data = pd_ob_data.append(sheet1, ignore_index=True)

    # add columns indexs  
    column_keys[0] = "IMG_ID"
    pd_ob_data.columns = column_keys 

    print(ob_data_path)
    print(pd_ob_data)
    return pd_ob_data


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
    read_objective_data()
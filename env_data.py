# 每个图片只需要这些信息：拍摄时间、图像大小（4k多x3k多那个分辨率）、
# 曝光时间、光圈、ISO感光度、焦距、测光模式这几个数据。

import exifread
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import PIL.ExifTags
from PIL import Image

def get_exif(fn):
    img = Image.open(fn)
    exif = { PIL.ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in PIL.ExifTags.TAGS }
    
    return exif

# ExcelWriter
def pd_to_exel( data_df, name = 'env_data.xlsx' ):
    cols = list(data_df)
    cols.insert(0, cols.pop(cols.index('IMG_ID')))
    data_df = data_df.loc[:, cols]

    writer = pd.ExcelWriter(name)
    data_df.to_excel(writer, float_format='%.5f')
    writer.save()

img_addrs = glob.glob('F:/workplace/public_dataset/环境数据/*/*.png')
data_df = pd.DataFrame()
plt.ion()    # 打开交互模式

for i in range(len(img_addrs)):
    im1 = Image.open(img_addrs[i])
    im1 = im1.convert('RGB')
    im1.save(str(i) + '.jpg')
    img = exifread.process_file(open(str(i) + '.jpg', 'rb'))

    # img = exifread.process_file(open(img_addrs[i], 'rb'))
    # img = get_exif(open(img_addrs[i], 'rb'))

    img_id = img_addrs[i].split('环境数据', 1 )[1]

    keys = list(img.keys())
    values = [str(x) for x in img.values()]
    rowdata = dict(zip(keys, values))
    # del rowdata['JPEGThumbnail']
    rowdata['IMG_ID'] = img_id

    # human evaluation
    # im_show = Image.open(img_addrs[i])
    # plt.figure(img_id)   # 图像窗口名称
    # plt.imshow(im_show)

    # sunlight       = input("有无日光：有日光1 ———— 无日光0  {0,1}：")
    # illumination   = input("光照好坏：好 5    ———— 坏 0    {0,1,2,3,4,5} ：")
    # fog_density    = input("雾浓程度：浓 5    ———— 薄 0    {0,1,2,3,4,5} ：")
    # # overall_color  = input("整体色彩：鲜明5   ———— 暗淡0    {0,1,2,3,4,5} ：")
    # # definition     = input("清晰度：  清晰5   ———— 模糊0    {0,1,2,3,4,5} ：")
    # rowdata['有无日光'] = sunlight
    # rowdata['光照好坏'] = illumination
    # rowdata['雾浓程度'] = fog_density
    # rowdata['整体色彩'] = overall_color
    # rowdata['清晰度']   = definition

    plt.close()

    df = pd.DataFrame(rowdata, columns=rowdata.keys(), index=[0])
    data_df = data_df.append(df, ignore_index=True)

    # 阶段性保存
    # if i % 5 == 0 and i > 0:
    #     pd_to_exel(data_df, name = 'env_data_temp' + str(i) + '.xlsx')

pd_to_exel(data_df, name='1005.xlsx')
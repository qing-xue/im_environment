# im_environment

此为制作训练数据集的项目，注意不要上传项目所含的数据或模型。


# 剔除数据说明
根据PM2.5和清晰度的两列数据进行剔除。理论上PM2.5值越高，清晰度越低。根据PM2.5的值从高到低排序，剔除PM2.5值很低但清晰度很低的值或者是PM2.5值很高但清晰度很高的值。
比如\20191007上午\1.jpg中PM2.5为14，而清晰度为1，理论上清晰度要接近5。
所以选择剔除掉，以此类推。
由于数据集本身并不大，尽量选择剔除差别特别大的图片。

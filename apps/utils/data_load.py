import apps.PCA.PCA
import os
import cv2 as cv
import numpy as np

# 读取数据
# 参数 ： 读取文件的目录
# 返回值 ：图像 标签（文件名称） 名字集合 （所有的标签）
def LoadData(data):
    # data表示训练数据集所在的目录，要求图片尺寸一致
    # images：[m, height, width] 其中m代表样本个数，height代表图片高度，width代表宽度
    # names: 名字的集合
    # labels: 标签

    images = []
    labels = []
    names = []

    label = 0
    # 过滤所有的文件夹
    for subDirname in os.listdir(data):
        subjectPath = os.path.join(data, subDirname)
        if os.path.isdir(subjectPath):
            # 每一个文件夹下存放着一个人的照片
            names.append(subDirname)
            for fileName in os.listdir(subjectPath):
                imgPath = os.path.join(subjectPath, fileName)
                img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

                # 对图像进行pca降维 (x,y分别降维到 50个特征点)
                img = apps.PCA.PCA.PCA_Data(img)
                img = np.transpose(img)
                img = apps.PCA.PCA.PCA_Data(img)
                img = np.transpose(img)

                #将图像进行展平
                img = img.ravel()
                images.append(img)
                labels.append(label)
            label += 1
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels, names
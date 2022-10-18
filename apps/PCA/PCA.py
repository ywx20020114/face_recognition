from sklearn.decomposition import PCA

# 对图片数据进行PCA降维
# 参数 ： 图片原始特征
# 返回值 ： PCA降维后的特征
def PCA_Data(features):

    pca = PCA(n_components=50)
    pca.fit(features)
    features = pca.transform(features)

    return features
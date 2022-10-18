from sklearn.svm import SVC
import apps.utils.data_load

# 构建SVM模型 并且进行训练
# 参数 ： 训练集所在目录
# return  ： 训练好的模型
def svc(data):
    # data = 'C:\Workspace\PycharmProjects\Face_recognition\Data'
    images, labels, names = apps.utils.data_load.LoadData(data)
    # Xtrain,Xtest,ytrain,ytest = train_test_split(images,labels,random_state=42)

    svc = SVC(kernel='rbf',class_weight='balanced')
    svc.fit(images,labels)
    return svc



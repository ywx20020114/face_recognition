import face_get
import recognition


# 进行测试
if __name__=='__main__':
    data = 'C:\Workspace\PycharmProjects\Face_recognition\Data'
    a = input("请输入操作名称")
    if a=='人脸录入':
        face_get.generator(data)
    if a=='人脸识别':
        recognition.FaceRecognize(data)


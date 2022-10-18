import cv2 as cv
import os
import apps.MTCNN
from apps.yolov5.yolo import YOLO
from apps.yolov5 import my_predict
# 生成自己的人脸数据 保存到指定目录中
# 参数 ： 数据保存的位置
def generator(data,inputname):

    # 创建yolo模型
    yolo = YOLO()

    name = inputname

    # 拼接路径
    path = os.path.join(data, name)
    # 如果路径存在则删除
    # if os.path.isdir(path):
    #     shutil.rmtree(path) #递归删除文件夹

    # 读取到该文件下 已经存在的最大index
    index = 0
    if os.path.exists(path):
        for fileName in os.listdir(path):
            index += 1

    # 如果没有文件夹  创建文件
    if not os.path.exists(path):
        os.mkdir(path)


    # 打开摄像头
    camera = cv.VideoCapture(0)
    # cv.namedWindow('Face')


    # 计数
    count = 0
    while True:
        # 读取一帧图像
        ret, frame = camera.read()

        # 判断图片是否读取成功
        if ret:
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            #人脸检测

            x, y, w, h = my_predict.predict(yolo,frame)

            # x = int(x)
            # y = int(y)
            # w = int(w)
            # h = int(h)
            # 在原图像上绘制矩形
            if w == -1:
                continue
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # 调整图像大小 和ORL人脸库图像一样大小
            f = cv.resize(frame[y:y+h,x:x+w],(92,112))

            # 保存人脸
            cv.imwrite('%s/%s.jpg'%(path,str(count+index)),f)
            count += 1
            # cv.imshow('Face', frame)
            # #如果按下q键则退出
            # if cv.waitKey(100) & 0xff == ord('q') or count == 20:
            #     break
            image = cv.imencode('.jpg', frame)[1].tobytes()
            if count >= 50:
                return

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
    camera.release()
    # cv.destroyAllWindows()



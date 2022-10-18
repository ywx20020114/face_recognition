from apps.recognition import FaceRecognize as fr
from apps.face_get import generator as gt
from apps.utils.get_name import get_name
import os
import cv2
from flask import Flask, render_template, Response, request

app = Flask(__name__)
# 人脸数据保存位置
data = 'apps/Data'

# 识别结果保存名字位置
Name = 'apps/Name'

# 首页界面
@app.route('/')
def index():
    return render_template('index.html')

# 人脸识别进入界面
@app.route('/face_recognition')
def face_recognition():
    return render_template("face_recognition.html")

# 进行识别界面
@app.route('/face_recognition/video0')
def show0():
    return render_template('video0.html')

# 进行识别界面摄像头显示
@app.route('/face_recognition/video_feed0')
def video_feed0():
    return Response(fr(data), mimetype='multipart/x-mixed-replace; boundary=frame')

# 识别结果界面
@app.route('/face_recognition/result')
def result():
    name=get_name()
    return render_template('result.html',name=name)

# 识别结果返回
@app.route('/face_recognition/result/face')
def face():
    name = get_name()
    path = os.path.join(data, name, '10.jpg')
    image1 = cv2.imread(path)
    image = cv2.imencode('.jpg', image1)[1].tobytes()
    return Response(image, mimetype="image/jpeg")

# 人脸录入界面
@app.route('/face_get')
def face_get():
    return render_template("face_get.html")


# 进行录入界面
@app.route('/face_get/video_feed1', methods=['GET', 'POST'])
def video_feed1():
    name = request.form.get("name", type=str)
    return Response(gt(data, name), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')

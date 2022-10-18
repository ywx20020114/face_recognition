from apps.yolov5.yolo import YOLO
import cv2
from PIL import Image


# 需要传入Yolo模型和cv格式的image
def predict(model, image):
    # 将cv格式的Image转化为PIL格式
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 获取左上角坐标和宽和高
    x, y, width, height = model.detect_image(image)
    return x, y, width, height


if __name__ == "__main__":
    # 这是测试用例
    yolo = YOLO()
    i = 0
    while i <= 3:
        image_path = 'VOCdevkit/VOC2007/JPEGImages/' + '000' + str(i) + '.jpg'
        image = cv2.imread(image_path)
        x,y,width,height = predict(yolo, image)
        print(x,y,width,height)
        roi_gray = image[y:y + height, x:x + width]
        cv2.imshow('',roi_gray)
        cv2.waitKey()
        cv2.destroyAllWindows()
        i = i + 1

import sys
import numpy as np
import tensorflow as tf
import cv2

# 构造PNET

# P-Net包含三个卷积层，每个卷积核的大小均为3✖3，注意到P-Net 没有全连接层。
# （1）作用：判断是否含人脸，并给出人脸框和关键点的位置，为O-Net提供人脸候选框。
# （2）输入：尺寸大小为 12✖12的三通道图像
# （3)输出：包含三部分：
# a.是否人脸的概率1✖1✖2向量（之所以有两个值（0和1的概率）是因为为了方便计算交叉熵）；
# b.人脸检测框坐标（左上点和右下点）1✖1✖4向量；
# c.人脸关键点（5个关键点）坐标1✖1✖10向量。

def create_pnet():
    #训练时输入图像大小为12x12x3，测试时输入图像宽高不小于12
    input = tf.keras.Input(shape=[None, None, 3])
    x = tf.keras.layers.Conv2D(10, (3,3), strides = (1,1), padding='valid', name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1,2], name='PReLU1')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=1,padding='valid',name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='PReLU2')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='valid', name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='PReLU3')(x)

    classifier = tf.keras.layers.Conv2D(2,(1,1), activation='softmax', name='conv4-1')(x)
    bbox_regress = tf.keras.layers.Conv2D(4,(1,1), name='conv4-2')(x)

    model = tf.keras.models.Model([input], [classifier, bbox_regress])
    return model

# 构造RNET

#P-Net网络结构与P-Net的网络结构类似，也包含三个卷积层，前两个卷积核的大小均为3✖3，第三个卷积核的大小为2✖2，
# 且其相比于P-Net 多了一个全连接层。
# （1）作用：对P-Net 输出可能为人脸候选框图像进一步进行判定，同时细化人脸检测目标框精度。
# （2）输入：尺寸大小为 24✖24的三通道图像
# （3）输出：包含三部分：
# a.是否人脸的概率的1✖1✖2向量；
# b.人脸检测框坐标（左上点和右下点）1✖1✖4向量；
# c.人脸关键点坐标1✖1✖10向量。

def create_rnet():
    input = tf.keras.Input(shape=[24, 24, 3])
    x = tf.keras.layers.Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = tf.keras.layers.Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = tf.keras.layers.Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

    x = tf.keras.layers.Permute((3, 2, 1))(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, name='conv4')(x)
    x = tf.keras.layers.PReLU(name='prelu4')(x)

    classifier = tf.keras.layers.Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = tf.keras.layers.Dense(4, name='conv5-2')(x)

    model = tf.keras.models.Model([input], [classifier, bbox_regress])

    return model

# 构造ONET

# O-Net网络结构相比R-Net的网络结构，多了一个3✖3卷积层，
# （1）作用：对R-Net 输出可能为人脸的图像进一步进行判定，同时细化人脸检测目标框精度。
# （2）输入：尺寸大小为 48✖48的三通道图像
# （3）输出：包含三部分：
# a.是否人脸的概率的1✖1✖2向量；
# b.人脸检测框坐标（左上点和右下点）1✖1✖4向量；
# c.人脸关键点坐标1✖1✖10向量。

def create_onet():
    input = tf.keras.layers.Input(shape = [48,48,3])
    # 48,48,3 -> 23,23,32
    x = tf.keras.layers.Conv2D(32, (3, 3),
                                strides=1,
                                padding='valid',
                                name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                name='prelu1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = tf.keras.layers.Conv2D(64, (3, 3),
                                strides=1,
                                padding='valid',
                                name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                name='prelu2')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2)(x)
    # 8,8,64 -> 4,4,64
    x = tf.keras.layers.Conv2D(64, (3, 3),
                                strides=1,
                                padding='valid',
                                name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                name='prelu3')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    # 4,4,64 -> 3,3,128
    x = tf.keras.layers.Conv2D(128, (2, 2),
                                strides=1,
                                padding='valid',
                                name='conv4')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1,2],
                                name='prelu4')(x)
    # 3,3,128 -> 128,12,12
    x = tf.keras.layers.Permute((3,2,1))(x)

    # 1152 -> 256
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, name='conv5') (x)
    x = tf.keras.layers.PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    classifier = tf.keras.layers.Dense(2,
                                        activation='softmax',
                                        name='conv6-1')(x)
    bbox_regress = tf.keras.layers.Dense(4,name='conv6-2')(x)
    landmark_regress = tf.keras.layers.Dense(10,name='conv6-3')(x)

    model = tf.keras.models.Model([input], [classifier, bbox_regress, landmark_regress])

    return model


# NMS 主要是用于对多个候选框去除重合率大的冗余候选框，从而保留区域内最优候选框，其过程是一个迭代遍历的过程。
def nms(rectangles, threshold):
    if (len(rectangles) == 0):
        return rectangles
    bbx = np.array(rectangles)
    x1 = bbx[:, 0]
    y1 = bbx[:, 1]
    x2 = bbx[:, 2]
    y2 = bbx[:, 3]
    score = bbx[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    index = np.array(score.argsort())
    pick = []
    while len(index) > 0:
        xx1 = np.maximum(x1[index[-1]], x1[index[0:-1]])
        yy1 = np.maximum(y1[index[-1]], y1[index[0:-1]])
        xx2 = np.minimum(x2[index[-1]], x2[index[0:-1]])
        yy2 = np.minimum(y2[index[-1]], y2[index[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        inter_scale = inter / (area[index[-1]] + area[index[0:-1]] - inter)
        pick.append(index[-1])
        index = index[np.where(inter_scale < threshold)]
    ret = bbx[pick].tolist()

    return ret


# 使用MCTTNN进行人脸检测
# 参数 ： 检测的图片
# 返回值 人脸框的x,y,w,h 信息
# 此版本为了增加实时检测的速度  只使用了Pnet 和 Rnet 没有使用耗时最久的 Onet
def MCTNN(image):
    # def MTCNN(image):
    # 读取输入图像，并转为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = min(image.shape[0], image.shape[1])
    min_face_size = image_size * 0.05
    # print(image.shape)
    # print(type(image))

    # 计算缩放图像所需尺度因子，使得最顶层图像尺寸不小于12
    scales = []
    factor = 0.709
    for i in range(0, 10):
        if ((factor ** i) * image_size > 12):
            scales.append(factor ** i)
    # print(scales)

    # 构造金字塔，后面会把输入图像金字塔组成batch输入到PNET，实际上会把每一层金字塔扩展为底层大小
    pyramid_imgs = []
    for scale in scales:
        new_row = int(image.shape[0] * scale)
        new_col = int(image.shape[1] * scale)
        img_scaled = cv2.resize(image, (new_col, new_row))

        img_ = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
        img_[0:new_row, 0: new_col] = img_scaled
        # cv2.imshow("img_", img_)
        # cv2.waitKey()
        pyramid_imgs.append(img_)

    # 构造PNET的输入图像，归一化处理
    pnet_input_imgs = []
    for img in pyramid_imgs:
        img_ = (img - 127.5) / 127.5
        pnet_input_imgs.append(img_)

    # 将图像列表转为ndarray
    pnet_input_array = np.array(pnet_input_imgs)

    # 构造PNET网络
    pnet = create_pnet()
    pnet.load_weights(".\weight_path\pnet.h5", by_name=True)
    # 推理输出
    pnet_output = pnet.predict(pnet_input_array)
    # print(len(pnet_output))
    # print(pnet_output[0].shape)
    # print(pnet_output[1].shape)

    # 处理矩形框
    pnet_threshold = 0.7
    pnet_bbx = []
    for i in range(len(scales)):
        cls_prob = pnet_output[0][i, :, :, 1]
        row, col = np.where(cls_prob > pnet_threshold)
        if row.shape[0] == 0:
            continue
        start_pt = np.array((col, row)).T
        left_top = np.fix((start_pt * 2) / scales[i])  # n*2
        right_down = np.fix(((start_pt * 2) + 11) / scales[i])  # n*2
        bbx = np.concatenate((left_top, right_down), axis=1)  # n*4
        scores = np.array(pnet_output[0][i, row, col, 1])
        scores.resize(len(row), 1)  # n*1
        offsets = pnet_output[1][i, row, col] * 12 / scales[i]
        bbx = bbx + offsets
        bbx = np.concatenate((bbx, scores), axis=1)
        for b in bbx:
            # 不合理的矩形框舍弃，这里只处理矩形框宽高为负数的情况
            if ((b[2] < b[0])
                    or (b[3] < b[1])):
                continue
            # 矩形框扩展为正方形
            w = b[2] - b[0]
            h = b[3] - b[1]
            l = max(w, h)
            b[0] = b[0] - (l - w) * 0.5
            b[2] = b[2] + (l - w) * 0.5
            b[1] = b[1] - (l - h) * 0.5
            b[3] = b[3] + (l - h) * 0.5

            # 修改矩形框的分布，使其位于图像内部
            b[0] = max(0, b[0])
            b[1] = max(0, b[1])
            b[2] = min(image.shape[1], b[2])
            b[3] = min(image.shape[0], b[3])
            # 过小的人脸舍弃掉
            if ((b[2] - b[0] < min_face_size)
                    or (b[3] - b[1] < min_face_size)):
                continue
            pnet_bbx.append(b)
    # print(np.array(pnet_bbx).shape)

    if (len(pnet_bbx) == 0):
        print("no face detected")
        sys.exit()

    # nms过滤
    pnet_bbx_nms = nms(pnet_bbx, 0.7)



    ## 之后进入RNET阶段
    # 构造rnet
    rnet = create_rnet()
    rnet.load_weights(".\weight_path\\rnet.h5", by_name=True)

    # 构造rnet输入，尺寸缩放为24X24，归一化
    rnet_input_imgs = []
    for bbx in pnet_bbx_nms:
        img = image[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
        img = cv2.resize(img, (24, 24))
        img_ = (img - 127.5)  /127.5
        rnet_input_imgs.append(img_)
    rnet_input_array = np.array(rnet_input_imgs)

    # 推理输出
    rnet_output = rnet.predict(rnet_input_array)
    # print(len(rnet_output))
    # print(len(pnet_bbx_nms))
    # print(rnet_output[0].shape)
    # print(rnet_output[1].shape)

    # 整理输出矩形框
    rnet_bbx = []
    rnet_score_threshold = 0.7
    for i in range(len(pnet_bbx_nms)):
        rnet_score = rnet_output[0][i, 1]
        offset = rnet_output[1][i]
        if(rnet_score < rnet_score_threshold):
            continue
        w = pnet_bbx_nms[i][2] - pnet_bbx_nms[i][0]
        h = pnet_bbx_nms[i][3] - pnet_bbx_nms[i][1]
        scales = np.array([w / 24, h / 24, w / 24, h /24])
        bbx = pnet_bbx_nms[i]
        bbx[0:4] = bbx[0:4] + offset * 24 * scales
        bbx[4] = rnet_score
        # 不合理的矩形框舍弃，这里只处理矩形框宽高为负数的情况
        if((bbx[2] < bbx[0])
        or (bbx[3] < bbx[1])):
            continue
        # 矩形框扩展为正方形
        w = bbx[2] - bbx[0]
        h = bbx[3] - bbx[1]
        l = max(w, h)
        bbx[0] = bbx[0] - (l - w) * 0.5
        bbx[2] = bbx[2] + (l - w) * 0.5
        bbx[1] = bbx[1] - (l - h) * 0.5
        bbx[3] = bbx[3] + (l - h) * 0.5

        # 修改矩形框的分布，使其位于图像内部
        bbx[0] = max(0, bbx[0])
        bbx[1] = max(0, bbx[1])
        bbx[2] = min(image.shape[1], bbx[2])
        bbx[3] = min(image.shape[0], bbx[3])
        # 过小的人脸舍弃掉
        if((bbx[2] - bbx[0] < min_face_size)
        or (bbx[3] - bbx[1] < min_face_size)):
            continue
        rnet_bbx.append(bbx)

    if (len(rnet_bbx) == 0):
        print("no face detected")
        sys.exit()

    # nms过滤
    rnet_bbx_nms = nms(rnet_bbx, 0.7)


    ## ONET阶段
    # 构造onet
    onet = create_onet()
    onet.load_weights(".\weight_path\onet.h5", by_name=True)

    # # 构造onet输入，尺寸缩放为48X48，归一化
    # onet_input_imgs = []
    # for bbx in rnet_bbx_nms:
    #     img = image[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
    #     img = cv2.resize(img, (48, 48))
    #     img_ = (img - 127.5)  /127.5
    #     onet_input_imgs.append(img_)
    # onet_input_array = np.array(onet_input_imgs)
    #
    # # 推理输出
    # onet_output = onet.predict(onet_input_array)
    #
    # # 整理输出矩形框
    # onet_bbx = []
    # onet_score_threshold = 0.8
    # for i in range(len(rnet_bbx_nms)):
    #     onet_score = onet_output[0][i, 1]
    #     offset = onet_output[1][i]
    #     if(onet_score < onet_score_threshold):
    #         continue
    #     w = rnet_bbx_nms[i][2] - rnet_bbx_nms[i][0]
    #     h = rnet_bbx_nms[i][3] - rnet_bbx_nms[i][1]
    #     scales = np.array([w / 48, h / 48, w / 48, h /48])
    #     bbx = rnet_bbx_nms[i]
    #     bbx[0:4] = bbx[0:4] + offset * 24 * scales
    #     bbx[4] = onet_score
    #     # 不合理的矩形框舍弃，这里只处理矩形框宽高为负数的情况
    #     if((bbx[2] < bbx[0])
    #     or (bbx[3] < bbx[1])):
    #         continue
    #     # 矩形框扩展为正方形
    #     w = bbx[2] - bbx[0]
    #     h = bbx[3] - bbx[1]
    #     l = max(w, h)
    #     bbx[0] = bbx[0] - (l - w) * 0.5
    #     bbx[2] = bbx[2] + (l - w) * 0.5
    #     bbx[1] = bbx[1] - (l - h) * 0.5
    #     bbx[3] = bbx[3] + (l - h) * 0.5
    #
    #     # 修改矩形框的分布，使其位于图像内部
    #     bbx[0] = max(0, bbx[0])
    #     bbx[1] = max(0, bbx[1])
    #     bbx[2] = min(image.shape[1], bbx[2])
    #     bbx[3] = min(image.shape[0], bbx[3])
    #     # 过小的人脸舍弃掉
    #     if((bbx[2] - bbx[0] < min_face_size)
    #     or (bbx[3] - bbx[1] < min_face_size)):
    #         continue
    #     onet_bbx.append(bbx)
    #
    #
    # # nms过滤
    # onet_bbx_nms = nms(onet_bbx, 0.7)
    x,y,w,h = rnet_bbx_nms[0][0:4]
    return x,y,w,h

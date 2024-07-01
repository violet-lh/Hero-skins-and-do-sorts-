'''
topic: 英雄联盟图片色彩分析

'''
import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd


# 皮肤图片所在的文件夹路径
skin_folder = 'E:/hanll/研究生/course/数据挖掘/skins'

# 设置目标尺寸和像素值
target_size = (32, 32)
target_mode = "RGB"

# 新建列表用于存储每个英雄的皮肤图片信息
skin_data = np.zeros((1440, 32, 32, 3), dtype=np.uint8)
# 标签 
df = pd.read_excel('labels.xlsx')
third_column = df.iloc[:, 2]
skin_label = np.array(third_column)

i = 0 # 做标记

# 遍历文件夹中的子文件夹
for subfolder in os.listdir(skin_folder):
    # 构建子文件夹完整路径
    subfolder_path = os.path.join(skin_folder, subfolder)
    # 确保是一个文件夹
    if os.path.isdir(subfolder_path):
                
        # 遍历子文件夹中的图片文件
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            # 打开图片文件并获取元数据
            img = Image.open(file_path)
                       
            # 统一图片尺寸和像素值
            img = img.resize(target_size, resample=Image.BICUBIC)
            # 转换图片模式为"RGB"
            if img.mode != target_mode:
                img = img.convert(target_mode)
            
            # 将像素值标准化到 0 到 1 的区间内
            img_array = np.array(img, dtype=np.float32)
            img_array /= 255.0

            # 将图片信息添加到英雄的列表中
            skin_data[i] = img_array

            i = i+1 # 标记+1
                       
            # 关闭图片文件
            img.close()

# 分割测试集和预测集
train_images, test_images = skin_data[0:719,], skin_data[720:1439,] 
train_labels, test_labels = skin_label[0:719], skin_label[720:1439] 


model = models.Sequential([
    layers.Conv2D(64, (3, 3),padding='same', activation='relu', input_shape=(32, 32, 3)), #卷积层1，卷积核3*3
    layers.Dropout(0.3),
    layers.MaxPooling2D((2, 2)),   #池化层1，2*2采样

    layers.Conv2D(64, (3, 3),padding='same', activation='relu'), #卷积层2，卷积核3*3
    layers.Dropout(0.3),
    layers.MaxPooling2D((2, 2)),   #池化层2，2*2采样

    layers.Conv2D(128, (3, 3),padding='same', activation='relu'),  #卷积层3，卷积核3*3
    layers.Dropout(0.3),
    layers.MaxPooling2D((2, 2)),   #池化层3，2*2采样 

    layers.Conv2D(128, (3, 3),padding='same', activation='relu'),  #卷积层4，卷积核3*3
    layers.Dropout(0.3),
    layers.MaxPooling2D((2, 2)),   #池化层4，2*2采样

    layers.Conv2D(256, (3, 3),padding='same', activation='relu'),  #卷积层5，卷积核3*3
    layers.Dropout(0.3),
    
    layers.Flatten(),                        #Flatten层，连接卷积层与全连接层
    layers.Dense(1024, activation='relu'),   #全连接层，特征进一步提取
    # layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),    #全连接层，特征进一步提取
    # layers.Dropout(0.4),
    layers.Dense(8,activation="softmax")    #输出层，输出预期结果
])

model.summary()  # 模型打印网络结构

#adam = tf.keras.optimizers.Adam(lr=0.001) 用这个优化器会报错
model.compile(optimizer = 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, batch_size = 36,
                    validation_data=(test_images, test_labels))

#model.save('model.pb')
pre = model.predict(test_images)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

'''
读写excel文件
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''python导出矢量图形式'''
path = './'  # 图片输出路径
fig = plt.figure()  # 创建画板
ax = fig.add_subplot()
x1 = np.linspace(0, 10, 50)
y1 = (x1 * x1)
ax.plot(x1, y1, label='y=x$^2$')
ax.legend()  # 添加图例
fig.savefig(path + '输出图片.svg', format='svg', dpi=300)  # 输出


def read_write_xls(basestation, basestation_end):
    # 编码方式出错时用的读取excel
    basestation = "./xx.xls"
    basestation_end = "./test_end.xls"
    data = pd.read_excel(basestation)
    data.to_excel(basestation_end)


def txt_to_excel(txt_path, excel_path):
    import numpy as np
    '''
    **读取txt文件，转换成excel**
    '''
    f = open(txt_path)
    line = f.readline()
    data_list = []

    # # 提取文档
    while line:
        num = list(map(float, line.split(',')))
        data_list.append(num)
        line = f.readline()
    f.close()

    data_array = np.array(data_list)

    # 把array写入excel
    data = pd.DataFrame(data_array)
    writer = pd.ExcelWriter(excel_path)
    data.to_excel(writer, 'sheet_1', float_format='%.5f', header=False, index=False)
    writer.save()
    writer.close()


'''
using:
import My_utils
txt_path = './temp/1.txt'
excel_path = './temp/Advertising.xls'
My_utils.txt_to_excel(txt_path, excel_path)
'''
###################################################################


'''
转类型——使用图像与运算时，一定要转
'''


def bit_wisth():
    face_mask = [1, 0, 1]
    face_mask = face_mask.astype("uint8")


'''
把False 和 True 转为0和1
'''


def TF_to_10():
    a = np.array([[True, False], [False, True]])
    print(a)
    a.astype(int)
    print(a + 0)


# 定义可视化图像函数
def look_img(img):
    '''opencv 读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


"""
将txt数据转换为xls（表格）文件，方便后面做数据分析
"""

'''
    函数：def txt_xls(filename, xlsname):
    调用：
    filename = './output_data/fft.txt'
    xlsname = './output_data/fft.xls'
    txt_xls(filename, xlsname)
'''

# -*- encoding: utf-8 -*-
import xlwt  # 需要的模块


def txt_xls(filename, xlsname):
    # 文本转换成xls的函数
    # param filename txt文本文件名称、
    # param xlsname 表示转换后的excel文件名
    try:
        # f = open(filename, encoding="UTF-8")
        f = open(filename)
        xls = xlwt.Workbook()
        # 生成excel的方法，声明excel
        sheet = xls.add_sheet('sheet1', cell_overwrite_ok=True)
        x = 0
        while True:
            # 按行循环，读取文本文件
            line = f.readline()
            if not line:
                break  # 如果没有内容，则退出循环
            for i in range(len(line.split(','))):
                item = line.split(',')[i]
                sheet.write(x, i, item)  # x单元格经度，i 单元格纬度
            x += 1  # excel另起一行
        f.close()
        xls.save(xlsname)  # 保存xls文件
    except:
        raise


# if __name__ == "__main__":
#     filename = './output_data/fft.txt'
#     xlsname = './output_data/fft.xls'
#     txt_xls(filename, xlsname)

"""
鼠标点击某点，显示该点的像素值和HSV等信息
"""

'''
    函数：def shower_click(img_path):
    调用：shower_click('./1.jpg')
'''

import cv2


def shower_click(img_path):
    def mouse_click(event, x, y, flags, para):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
            print('PIX:', x, y)
            print("BGR:", img[y, x])
            print("GRAY:", gray[y, x])
            print("HSV:", hsv[y, x])

    cv2.namedWindow("img")
    img = cv2.imread(img_path)
    cv2.setMouseCallback("img", mouse_click)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()


# shower_click('./1.jpg')


""" 计时 """
# 进行测试
import time

T1 = time.time()

# 这块儿是程序

T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))  # 程序运行时间:0.0毫秒

import math


# 另一种形式(深度训练过程)：输出几分几秒
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()
# 这块儿是程序
print(time_since(start))

""" 打包数据存到xls """


# 存一手
def vstack_and_to_excel(x1, x2, dir_path):
    # 打包数据
    D1 = np.vstack((x1.T, x2.T)).T
    # 把数据转成DataFrames类型
    data = pd.DataFrame(D1)
    writer = pd.ExcelWriter(dir_path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer, 'sheet_1', float_format='%.5f', header=False, index=False)
    writer.save()
    writer.close()


# if __name__ == '__main__':
#     """
#     可能要pip install xlwt
#     保存文件必须为xls
#     """
#     a = np.arange(4)
#     b = np.array([1, 2, 4, 5])
#     dir_path = './macro.xls'
#     vstack_and_to_excel(a, b, dir_path)


import os
import re

"""批量修改文件夹的图片名"""


def ReFileName(dirPath, pattern):
    """
    :param dirPath: 文件夹路径
    :pattern:正则
    :return:
    """
    # 对目录下的文件进行遍历

    for file in os.listdir(dirPath):
        # 判断是否是文件

        if os.path.isfile(os.path.join(dirPath, file)) == True:
            newName = re.sub(pattern, '', file)
            newFilename = file.replace(file, newName)  # 把file 改成newName
            # 重命名
            os.rename(os.path.join(dirPath, file), os.path.join(dirPath, newFilename))
    print("图片名已全部修改成功")


if __name__ == '__main__':
    dirPath = r"./114-63/"
    pattern = re.compile(r'face')  # 使用 compile 函数将正则表达式的字符串形式编译为一个 Pattern 对象   # 36.25HZ\.   [(Colour)]
    ReFileName(dirPath, pattern)

    # ddawwdawdwd
    # wdaddwadwdd

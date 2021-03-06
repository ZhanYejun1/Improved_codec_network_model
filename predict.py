'''
predict.py有几个注意点
1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要原图和分割图不混合，可以把blend参数设置成False。
4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
for c in range(self.num_classes):
    seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
    seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
    seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
'''
from pspnet import PSPNet
from PIL import Image
import cv2
import numpy as np

pspnet = PSPNet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = pspnet.detect_image(image)
        # r_image.show()

        r_image = pspnet.detect_image(image)
        r_image = np.array(r_image)
        print(r_image.shape)
        r_image = cv2.bitwise_not(r_image)  # 按位取反，白变黑，黑变白
        image = np.array(image)
        cv2.imshow("yuantu", image)
        cv2.imshow("mobilev3_psp", r_image)
        cv2.imwrite(r"C:\Users\A3\Desktop\mobilev3_psp.png",r_image)
        cv2.waitKey()

# Improved_codec_network_model
自己改进的编解码网络，适用于语义分割。编码器部分主要改进了主干特征提取网络MobileNetV3,添加了快捷链路进行最大池化压缩信息，解码器部分添加了PSP模块，编解码部分添加了跳跃链接Skip Connection，采用双线性插值法上采样恢复像素。

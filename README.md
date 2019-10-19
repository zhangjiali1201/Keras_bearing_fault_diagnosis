# Keras_bearing_fault_diagnosis
**基于CNN的轴承故障识别**

**1.简介**
本科毕设，基于凯斯西储大学公开轴承数据集建立的CNN轴承故障识别模型。

记录相关过程的博客：https://blog.csdn.net/zhangjiali12011/article/details/90523879

数据集：https://csegroups.case.edu/bearingdatacenter/home

**2.环境**
Windows+anaconda

**3.框架**
Keras

**4.依赖**
tensorflow;keras;numpy;scipy;os;sklearn;matplotlib

**5.说明**

cnn_1D.py 构建基础的CNN模型

lstm_diagnosis.py 构建基础的LSTM模型

cnn_diagnosis 论文代码

cnn_valid.py 测试不同全连接层数、卷积层数选择最优模型

preprocess.py 数据预处理（预处理代码来自https://github.com/AaronCosmos/wdcnn_bearning_fault_diagnosis ）

*images文件夹 保存模型流程图；
data文件夹 保存数据集；
logs文件夹 保存日志*

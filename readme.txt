（1）创建conda环境
conda create -n mamba_out python=3.8 -y

（2）激活环境
conda activate mamba_out

（3）在激活的环境中安装所需的库，主要包括：
torch、torchvision、ultralytics、ensemble_boxes

（4）将第二阶段测试的数据中的可见光和热红外图像分别放到根目录datasets文件夹下的RGB和TIR文件夹中

（5）分别运行分别运行根目录下ADDL/test.py、GPT_m/test.py、yolov8_tir/test.py，在output文件夹中输出各个模型的预测结果

（6）然后运行根目录下的test.py，在根目录下生成result.csv文件

说明：本代码在yolov8的基础上修改，若有yolov8的环境，只需再安装一个ensemble_boxes库即可运行

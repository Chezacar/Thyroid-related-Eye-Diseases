## 数据集是保密的
## Classification:
- 各个版本的Resnet训练代码
## FAN_location:
此文件夹中代码都使用1.0.1版的PyTorch，face_alignment库可以查阅https://github.com/1adrianb/face-alignment
所有的ipynb文件都要用jupyter notebook打开

以下是一些代码的用途，没有介绍到的代码多是一些临时测试用的代码，可以先不管

    FAN_location/batch_process_new.py
        使用开源的face_alignment库，对一个文件夹中的所有人脸定位关键点，并用白色圆点标识，结果的图片另行保存。该代码仅将结果保存成图片形式，并不保存具体坐标，所以只是用于观看的。

    classification/data_process/point_local.ipynb
        因为face_alignment库给出的坐标结果是一个numpy矩阵，这个代码确定里矩阵里哪些元素表示眼睛和眉毛部分关键点的坐标，具体见代码注释

    classification/data_process/dataset_split.ipynb
        用于训练、验证、测试集的划分，训练、验证集用 4 fold分割

    classification/imgcrop/TAO/get_points.ipynb
        对数据集中selected文件夹的每个图片提取关键点，并将结果保存成json文件和pickle文件（以.dict结尾，因为对象是一个字典）

    classification/imgcrop/TAO/batch_crop.ipynb
        读取已经提取到的关键点，并据此裁剪出每张图片眼睛周围的部分并存为新图片，作为模型的输入。

    classification/imgcrop/SCUT
        SCUT数据集已经提供了人脸关键点坐标，只不过具体的点位置和face_alignment库定义的不同，这个文件夹中的代码根据SCUT提供的关键点坐标进行裁剪，另存为图片，

    classification/exp_r18
        这个文件夹中的文件是最终训练分类模型的，其中resnet.py定义模型，TAO_loader.py定义了数据集对象，train.py是主文件，train_s1.sh是运行train.py的脚本。lr_ext1.txt定义了学习率。下面的test文件夹是用测试机测试准确率。

    classification/exp_r34/50
        和exp_r18类似，只是加深了模型层数


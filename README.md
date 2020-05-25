# Courier Competition Round1
智慧物流：新冠期间饿了么骑士行为预估比赛第一轮，预测骑手下一步行动

比赛链接：https://tianchi.aliyun.com/competition/entrance/231777/information

结果:
| name     | score | MAE | rank| 
| :--------|:------|:----|:----|
|正方形的圆| 0.825 | 106 | 1   |


# 主要思路：
1. 训练样本的组合并不是直接每一个action作为一个样本，而是2个action组成pair作为一个样本，比如一个骑手有6个需要选择的行动步（1,2,3,4,5,6）,其中选择1为正确答案，那么组成pair(1,2, positive),(2,1,negative) (1,3,positive) (3,1,negative),然后预测的时候若有6个样本，则组成30个pair，然后每个pair看那个胜出，选择那个胜出的。
2. 数据增强：对于一个训练集中的骑手，通过枚举未知时间步的数量，得到更多的样本
3. 特征方面：除了用一些常用特征之外（未知时间步和最后一个时间步距离、骑手速度、订单保证送达时间等），还使用一些聚合特征，比如一个未知时间步和其他未知时间步的距离，其他未知时间步时间聚合。

# 运行方法

进入code目录，输入命令

python main.py

可运行从原始数据到最终结果的预测过程，整个过程大概30多分钟，完成后会打印finish prediction字样

# 关于requirements.txt

requirements.txt为服务器环境自动生成的依赖，在本地的笔记本上测试，在安装anoconda之后只需确认Lightgbm版本和pandas版本一致就能成功运行

# 训练和生成所有中间变量

1. 进入feature目录，按
- 0_data_sort_out.ipynb
- 1_build_dataset.ipynb
- 2_generate_train_test_courier_feature.ipynb
- 3_part_train_mp.ipynb
- 4_generate_train_data_arugment_sample.ipynb
- 5_generate_test_data_sample.ipynb

顺序依次执行这些jupyter notebook程序

2. 进入model目录，按
- 0_makepair_gbdt.ipynb
- 1_regress_task.ipynb
 
顺序依次执行这些jupyter notebook程序

# 注意
从原始数据到最终结果的预测过程，整个过程大概30多分钟，完成后会打印finish prediction字样，请不要提前退出


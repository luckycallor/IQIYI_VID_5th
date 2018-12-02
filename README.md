# IQIYI_VID_5th
这是爱奇艺多模态视频人物识别挑战赛第五名的代码分享，比赛连接：http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e

## 运行环境
python3.6
tensorflow
sklearn

## 方法介绍
本方法完全基于比赛官方提供的人脸特征进行，未使用任何其他数据。
模型由一个两层的神经网络组成，使用整个视频的人脸特征加权平均作为输入，输出的分类考虑了噪声数据。方法的主要创新点在于对噪声数据的利用，具体方法如下：
1、对噪声数据进行扩充，具体方法是将长视频随机截取一部分，以及将短视频拼接为长视频
2、利用DBSCAN算法对噪声数据进行聚类，将得到的聚类簇作为新的人物分类，实验中得到3691个聚类簇
3、由于大部分噪声数据缺少足够多的近邻，因此不能被DBSCAN算法聚类为簇。为了处理这部分噪声数据，我们在分类层增加了额外的4934个类别（与目标人物类别数相同）。因此最终的分类层共有4934+3691+4934个类别， 其中0-4933为目标人物类别，4934-8624为聚类簇人物类别，8625-13588为额外的4934个噪声人物类别。在整个模型的训练过程中，对这些噪声的类别标签进行动态更新：当某一个噪声数据被模型分类为一个目标人物时，我们就将这个噪声的类别线性映射到额外的4934个类别中。举例来说，如果某个噪声视频被分类为l（l<4934），那个这个噪声视频的类别标记会被更新为l+8625，并用于后续的模型训练。
最终结果是通过对整个数据集进行8折划分，用其中7个作为训练集，1个作为验证集，最终训练得到8个模型，对8个模型的结果进行融合得到的。
方法细节可查看答辩ppt与代码。

## 总结
最终结果的测试成绩为0.8252
其中对噪声数据的利用大概能提升3-4个百分点
多模型融合大概能提升1-2个百分点

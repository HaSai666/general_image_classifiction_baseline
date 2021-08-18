# 此代码为科大讯飞pytorch简易版baseline，希望对大家初次接触这些东西能有一定的帮助，由于个人时间比较紧张，代码几乎没有写注释，但是只有仔细看应该都是可以看得懂的

## 本代码的conf文件夹下面给了一些示例配置文件，包括了efficientnet,resnet,densenet的示例使用,参考这些配置文件相信大家也可以很快上手其他模型以及配置

## 训练代码： python train.py --config_path ./conf/efficientnet.yaml
## 测试代码(生成提交文件)： python test.py --config_path ./conf/efficientnet.yaml

Tip:本baseline的文件组织如下，提取放入数据集即可，如果出现个别文件夹不存在，那就手动创建，另外，这套代码没有适配cpu，目前只有gou版
./pet/
├── ckpt
│   ├── efficientnet-b5
│   └── resnet50
├── code
│   ├── __pycache__
│   ├── conf
│   ├── dataset.py
│   ├── model.py
│   ├── readme.md
│   ├── test.py
│   ├── train.py
│   ├── trans.py
│   └── utils.py
├── data
│   ├── train
│   └── val
├── pretrained
│   ├── efficientnet-b5-b6417697.pth
│   ├── efficientnet-b6-c76e70fd.pth
│   ├── efficientnet-b7-dcc49843.pth
│   └── resnet50-19c8e357.pth
└── result
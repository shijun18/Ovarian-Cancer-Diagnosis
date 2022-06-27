# Ovarian-Cancer-Diagnosis
Deep learning method for automatic diagnosis of ovarian cancer using CT images and clinical information. 



### How to Run

#### Step 1: 生成训练数据

将原始数据手工划分为**训练集和独立测试集**，分别保存在指定文件夹下，如 `train_data` 和`test_data`，目录结构如下，包含两个子文件夹`r0`和`non-r0`，每个样例包含一个`dcm`的序列文件和一个`nii`的分割文件:

~~~bash
./non-r0
│   ├── ./non-r0/xxx
│   │   ├── ./non-r0/xxx/IM0
			...
│   │   ├── ./non-r0/xxx/IM42
│   └── ./non-r0/xxx_Merge.nii
└── ./r0
    ├── ./r0/xxx
    │   ├── ./r0/xxx/IM0
    		...
    │   ├── ./r0/xxx/IM42
    └── ./r0/xxx_Merge.nii
~~~

运行`convert_to_npy.py`将原始数据转存为`hdf5`格式，其中`image`和`mask`键值分别索引**图像**和**分割标注**，运行之前先指定输入路径和保存路径，如下（**提前将数据**）：

~~~python
#训练数据
input_path = '../dataset/raw_data/train_data'
save_path = '../dataset/npy_data/train_data'
convert_to_npy(input_path,save_path)

#测试数据
input_path = '../dataset/raw_data/test_data'
save_path = '../dataset/npy_data/test_data'
convert_to_npy(input_path,save_path)
~~~

#### Step 2: 生成数据索引文件

出于方便，将数据路径及其对应的标签以`csv`文件的形式保存，如下，`id`和`label`分别指向的是数据**绝对**路径及其对应的标签：

~~~python
id,label
xxxxx.hdf5,0
xxxxx.hdf5,1
~~~

运行`tools.py`，同样地，运行之前先指定输入路径和`csv`保存路径，如下：

~~~python
os.makedirs('./csv_file')

input_path = os.path.abspath('../dataset/npy_data/train_data')
csv_path = './csv_file/index.csv'
make_label_csv(input_path,csv_path)

input_path = os.path.abspath('../dataset/npy_data/test_data')
csv_path = './csv_file/test_index.csv'
make_label_csv(input_path,csv_path)
~~~

#### Step 3: 配置训练参数

修改`config.py`，默认配置如下，**注意版本号不要冲突**，最好基于某种规则来指定，方便记忆，我目前用的是模型列表的索引顺序(1起始)：

~~~python
__all__ = ['r3d_18', 'se_r3d_18','da_18','da_se_18','r3d_34','da_se_34','vgg16_3d','vgg19_3d']

NET_NAME = 'r3d_18'
VERSION = 'v1.0'
DEVICE = '1'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5


CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION,CURRENT_FOLD)

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':1e-3, 
    'n_epoch':80,
    'channels':1,
    'num_classes':3,
    'input_shape':(32,256,256),
    'crop':48,
    'scale':(-100,200),
    'use_roi':False or 'roi' in VERSION, #在data_loader.py 预留了接口，还没实现
    'batch_size':2,
    'num_workers':2,
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'weight_path':WEIGHT_PATH,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'gamma': 0.1,
    'milestones': [30,60,90],
    'T_max':5,
    'use_fp16':True
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
    'output_dir':'./ckpt/{}'.format(VERSION),
    'log_dir':'./log/{}'.format(VERSION),
    'optimizer':'AdamW',
    'loss_fun':'Cross_Entropy',
    'class_weight':None,
    'lr_scheduler':'MultiStepLR' # MultiStepLR
}

~~~

#### Step 4: 指定索引文件路径

修改`run.py`文件，写入**Step 2**生成的索引文件路径，如下：

~~~python
if 'train' in args.mode:
    csv_path = './converter/csv_file/index.csv'
...
        
elif 'inf' in args.mode:
    test_csv_path = './converter/csv_file/test_index.csv'
~~~

#### Step 5：开始训练

这一步就很简单

单折训练

~~~bash
python run.py -m train
~~~

多折（默认5折）训练

~~~bash
python run.py -m train-cross
~~~


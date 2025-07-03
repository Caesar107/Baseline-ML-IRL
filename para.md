ML-IRL 主要参数列表
1. 总体参数
obj: maxentirl_sa - 使用的算法类型，这里是最大熵IRL的状态-动作版本
IS: false - 是否使用重要性采样
seed: 23 - 随机种子
cuda: -1 - 使用CPU训练（-1）；如果是>=0的数则表示使用指定的GPU

2. 环境参数 (env)
env_name: Hopper-v3 - 使用的环境名称
T: 1000 - 每个轨迹的最大长度（时间步数）
state_indices: all - 使用状态向量中的哪些维度，"all"表示全部使用

3. IRL参数 (irl)
training_trajs: 10 - 每次训练使用的轨迹数量
n_itrs: 200 - IRL算法的迭代次数，即重复"训练SAC-更新奖励函数"的循环次数
save_interval: 0 - 保存模型的间隔（0表示不保存中间结果）
eval_episodes: 20 - 评估时使用的episode数量
expert_episodes: 1 - 使用的专家轨迹数量
resample_episodes: 1 - 重采样的episode数量

4. SAC参数 (sac) - 重点
k: 1 - 每次迭代训练SAC的次数
epochs: 5 - SAC训练的epoch数量
log_step_interval: 5000 - 日志记录的步骤间隔
update_every: 1 - 每隔多少步更新一次网络
random_explore_episodes: 1 - 使用随机策略探索的episode数量
update_num: 1 - 每次更新的批次数
batch_size: 100 - 每批的样本数量
lr: 1e-3 - 学习率
alpha: 0.2 - 熵正则化系数
automatic_alpha_tuning: false - 是否自动调整alpha
buffer_size: 1000000 - 回放缓冲区大小
num_test_episodes: 10 - 测试时的episode数量
reinitialize: false - 是否在每次迭代中重新初始化

5. 奖励函数参数 (reward)
use_bn: false - 是否使用批标准化
residual: false - 是否使用残差连接
hid_act: relu - 隐藏层激活函数
hidden_sizes: [64, 64] - 隐藏层大小
clamp_magnitude: 10 - 限制输出的最大幅度
lr: 1e-4 - 学习率
weight_decay: 1e-3 - 权重衰减
gradient_step: 1 - 每次更新的梯度步数
momentum: 0.9 - 动量

6. 判别器参数 (disc)
reinit: false - 是否重新初始化
model_type: resnet_disc - 模型类型，这里使用ResNet风格的判别器
num_layer_blocks: 3 - 层块数量
hid_dim: 128 - 隐藏层维度
hid_act: tanh - 隐藏层激活函数
batch_size: 800 - 批大小
lr: 0.0003 - 学习率
iter: 1200 - 训练迭代次数
解释与分析
SAC训练周期：

每个IRL迭代会训练SAC策略5个epochs（sac.epochs: 5）
总共会进行200次IRL迭代（irl.n_itrs: 200）
所以SAC总共会被训练1000次（5 * 200）
奖励学习：

MLP（多层感知机）奖励模型，64单元的隐藏层
学习率较小（1e-4）

评估频率：
每个IRL迭代后都会评估当前策略的性能（回报值）
评估时会进行20个episodes（irl.eval_episodes: 20）

数据使用：
仅使用1个专家轨迹（irl.expert_episodes: 1）进行训练
每次训练使用10个生成的轨迹（irl.training_trajs: 10）
这些参数设置表明这是一个典型的IRL训练配置，通过迭代优化策略和奖励函数来模仿专家行为。重点是SAC训练次数（epochs: 5）和IRL总迭代次数（n_itrs: 200），这决定了整个训练的时长和性能。


ML-IRL训练参数分析：

irl.n_itrs: 600 (IRL迭代次数)
sac.epochs: 5 (每次IRL迭代中SAC训练的epoch数)
env.T: 1000 (每个轨迹最大长度)
300w
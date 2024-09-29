#从零开始实现softmax回归

import torch
from IPython import display
from d2l import torch as d2l

#导入数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#初始化模型参数
num_inputs = 784  #图像中每个样本为28*28
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

#定义softmax操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

#定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

#定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

# 分类精度
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  #检测维度是否超过1维，且第二维是否大于1
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y         #生成布尔张量
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式，可以不设，主要是为了好的习惯，减少算梯度浪费的时间
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

#定义一个实用程序类Accumulator，用于对多个变量进行累加。
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
 #累加方法，接收任意数量的位置参数*args。这个方法通过zip函数将self.data列表和*args中的元素两两配对， 
 # #然后将每对元素相加（self.data中的元素加上*args中对应位置的元素，并将结果转换为浮点数），最后将累加后的结果重新赋值给self.data。
    def reset(self):                                                     
        self.data = [0.0] * len(self.data)
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]      

    def __getitem__(self, idx):
        return self.data[idx]
    

#训练一次
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]



#在动画中绘制数据的实用程序类Animator
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',               #设置x轴和y轴的缩放比例，如'linear'（线性）或'log'（对数）
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,          #fmts：一个元组，包含绘制线条时使用的样式
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]      #为子图做准备
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)    #定义了一个设置子图参数的函数
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):         #判断是否为列表
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):        #比如后面输入的是x=1 , y=[0.8,.07,0.6]
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]  #创建包含n个空列表的列表，第二次add时存在，故不需要创建
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)                           #依序加入，区分是因为，后面比如还会有x2，x3情形加入
                self.Y[i].append(b)                           #比如第一次X为[[1],[1],[1]] Y[[0.8],[0.7],[0.6]],第二次X[[1,2],[1,2],[1,2]],Y[[0.8,0.82],[0.7,0.72],[0.6,0.62]]
        self.axes[0].cla()                                    #清除当前子图
        for x, y, fmt in zip(self.X, self.Y, self.fmts):      #依次绘制各个子线条
            self.axes[0].plot(x, y, fmt)                      #通过调整axes[0]子图，也可以将不同线条画到不同子图上
        self.config_axes()
        display.display(self.fig)
        #d2l.plt.draw()
        d2l.plt.pause(0.001)
        display.clear_output(wait=True)


#训练多次
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))         #每次add一次
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


#定义updater
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


#正式训练

num_epochs = 10
d2l.plt.ion()
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.plt.ioff()
d2l.plt.show()



#预测
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
d2l.plt.show()

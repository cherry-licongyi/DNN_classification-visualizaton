import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward #参见资料包
import lr_utils #参见资料包

import tkinter as tk  # 使用Tkinter前需要先导入
# from tkinter import ttk
from tkinter import scrolledtext

np.random.seed(1) #设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。


# 线性回归
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.T)

# plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
# plt.title("Logistic Regression") #图标题
# plt.show()
# LR_predictions  = clf.predict(X.T) #预测结果
# print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
# 		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        "% " + "(正确标记的数据点所占的百分比)")


"""
    此函数是为了初始化多层网络参数而使用的函数。
    参数：
        layers_dims - 包含我们网络中每个图层的节点数量的列表
    
    返回：
        parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的字典：
                     W1 - 权重矩阵，维度为（layers_dims [1]，layers_dims [1-1]）
                     bl - 偏向量，维度为（layers_dims [1]，1）
"""

def initialize_parameters_deep(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    # 这里参数的初始化与浅层神经网络不同，为了避免梯度爆炸和消失，事实证明，如果此处不使用 “/ np.sqrt(layer_dims[l-1])” 会产生梯度消失
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])/np.sqrt(layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros(shape=(layers_dims[l],1))
        
        #确保我要的数据的格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))
        
    return parameters




"""
    实现前向传播的线性部分。

    参数：
        A - 来自上一层（或输入数据）的激活，维度为(上一层的节点数量，示例的数量）
        W - 权重矩阵，numpy数组，维度为（当前图层的节点数量，前一图层的节点数量）
        b - 偏向量，numpy向量，维度为（当前图层节点数量，1）

    返回：
         Z - 激活功能的输入，也称为预激活参数
         cache - 一个包含“A”，“W”和“b”的字典，存储这些变量以有效地计算后向传递
"""
def linear_forward(A,W,b):

    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
     
    return Z,cache




"""
    实现LINEAR-> ACTIVATION 这一层的前向传播

    参数：
        A_prev - 来自上一层（或输入层）的激活，维度为(上一层的节点数量，示例数）
        W - 权重矩阵，numpy数组，维度为（当前层的节点数量，前一层的大小）
        b - 偏向量，numpy阵列，维度为（当前层的节点数量，1）
        activation - 选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】

    返回：
        A - 激活函数的输出，也称为激活后的值
        cache - 一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递
"""
def linear_activation_forward(A_prev,W,b,activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    
    return A,cache



"""
多层模型的前向传播计算

    实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION
    
    参数：
        X - 数据，numpy数组，维度为（输入节点数量，示例数）
        parameters - initialize_parameters_deep（）的输出
    
    返回：
        AL - 最后的激活值
        caches - 包含以下内容的缓存列表：
                 linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
                 linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
"""
def L_model_forward(X,parameters):

    caches = []
    A = X
    L = len(parameters) // 2
    # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1,L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
  
        
    assert(AL.shape == (1,X.shape[1]))
    
    return AL,caches


"""
    实施等式（4）定义的成本函数。

    参数：
        AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
        Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

    返回：
        cost - 交叉熵成本
"""
def compute_cost(AL,Y):

    m = Y.shape[1]
    # Compute loss from aL and y
    logprobs = np.multiply(np.log(AL), Y) + np.multiply((1 - Y), np.log(1 - AL))
    cost = - np.sum(logprobs) / m
    # cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
        
    cost = np.squeeze(cost)
    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost


"""
    为单层实现反向传播的线性部分（第L层）

    参数：
         dZ - 相对于（当前第l层的）线性输出的成本梯度
         cache - 来自当前层前向传播的值的元组（A_prev，W，b）

    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度，与W的维度相同
         db - 相对于b（当前层l）的成本梯度，与b维度相同
"""

def linear_backward(dZ,cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db





"""
    实现LINEAR-> ACTIVATION层的后向传播。
    
    参数：
         dA - 当前层l的激活后的梯度值
         cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
         activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
         db - 相对于b（当前层l）的成本梯度值，与b的维度相同
"""
def linear_activation_backward(dA,cache,activation="relu"):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev,dW,db

"""
    对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播，就是多层网络的向后传播
    
    参数：
     AL - 概率向量，正向传播的输出（L_model_forward（））
     Y - 标签向量（例如：如果不是，则为0，如果是则为1），维度为（1，数量）
     caches - 包含以下内容的cache列表：
                 linear_activation_forward（"relu"）的cache，不包含输出层
                 linear_activation_forward（"sigmoid"）的cache
    
    返回：
     grads - 具有梯度值的字典
              grads [“dA”+ str（l）] = ...
              grads [“dW”+ str（l）] = ...
              grads [“db”+ str（l）] = ...
"""
def L_model_backward(AL,Y,caches):
    grads = {}
    
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # lth layer: (RELU -> LINEAR) gradients.
    # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads


"""
    使用梯度下降更新参数
    
    参数：
     parameters - 包含你的参数的字典
     grads - 包含梯度值的字典，是L_model_backward的输出
    
    返回：
     parameters - 包含更新参数的字典
                   参数[“W”+ str（l）] = ...
                   参数[“b”+ str（l）] = ...
"""
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 #整除
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters

# #测试update_parameters
# print("==============测试update_parameters==============")
# parameters, grads = testCases.update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
 
# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))







"""
    实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID。
    
    参数：
	    X - 输入的数据，维度为(n_x，例子数)
        Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
        layers_dims - 层数的向量，维度为(n_y,n_h,···,n_h,n_y)
        learning_rate - 学习率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每100次打印一次
        isPlot - 是否绘制出误差值的图谱
    
    返回：
     parameters - 模型学习的参数。 然后他们可以用来预测。
"""
def L_layer_model(X, Y, layers_dims, learning_rate=0.1, num_iterations=1000, print_cost=False,isPlot=True):
    np.random.seed(2)
    costs = []
    
    parameters = initialize_parameters_deep(layers_dims)
    
    plt.ion()    # 开启一个画图的窗口进入交互模式，用于实时更新数据
    for i in range(0,num_iterations):
        AL , caches = L_model_forward(X,parameters)

        cost = compute_cost(AL,Y)
        
        grads = L_model_backward(AL,Y,caches)
        
        parameters = update_parameters(parameters,grads,learning_rate)
        if np.sum(AL) == 0:
            print(i)
            _quit()
        #打印成本值，如果print_cost=False则忽略
        if i % 10 == 0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第", i ,"次迭代，成本值为：" ,np.squeeze(cost))
                op_txt = "第"+str(i)+ "次迭代，成本值为："+str(np.squeeze(cost))+'\n'
                textt.insert('insert',op_txt)
            
    #迭代完成，根据条件绘制图
            if isPlot:
                # plt.plot(np.squeeze(costs))
                # plt.ylabel('cost')
                # plt.xlabel('iterations (per handreds)')
                # plt.title("Learning rate =" + str(learning_rate))
                # plt.pause(0.1)         # 暂停一秒
                # plt.savefig('Figure.png')
                plt.clf()
                plot_decision_boundary(lambda X: predict(X.T,parameters), X, np.squeeze(Y))#修改
                plt.title("Decision Boundary for hidden layer size")
                plt.pause(0.05)
                plt.savefig('Figure_2.png')
    
    
    # plt.ioff()       # 关闭画图的窗口，即关闭交互模式
    # plt.show()       # 显示图片，防止闪退
    
    # plt.fiture()
    plt.clf()
    # #迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('Figure.png')
        # plt.show()
    
    image_file3 = tk.PhotoImage(file='Figure_2.png')  # 图片位置（相对路径，与.py文件同一文件夹下，也可以用绝对路径，需要给定图片具体绝对路径）
    rect = capic2.create_rectangle(0, 0, 540, 400,fill='white')# 画矩形正方形 
    image3 = capic2.create_image(280, -50,anchor='n',image=image_file3)        # 图片锚定点（n图片顶端的中间点位置）放在画布（250,0）坐标处

    plt.ioff()       # 关闭画图的窗口，即关闭交互模式
    plt.show()       # 显示图片，防止闪退
    

    return parameters

    # plot_decision_boundary(lambda X: predict(X.T,parameters), X, np.squeeze(Y))#修改
    # plt.title("Decision Boundary for hidden layer size")
    # plt.savefig('Figure_2.png')
    
    



"""
    该函数用于预测L层神经网络的结果，当然也包含两层
    
    参数：
     X - 测试集
     y - 标签
     parameters - 训练模型的参数
    
    返回：
     p - 给定数据集X的预测
"""
def predict(X, parameters):

    m = X.shape[1]
    n = len(parameters) // 2 # 神经网络的层数
    p = np.zeros((1,m))
    
    #根据参数前向传播
    probas, caches = L_model_forward(X, parameters)
    
    # for i in range(0, probas.shape[1]):
    #     if probas[0,i] > 0.5:
    #         p[0,i] = 1
    #     else:
    #         p[0,i] = 0
    
    # print("准确度为: "  + str(float(np.sum((p == y))/m)))
        
    return probas


def update_canvas(lay):
   
    rect = canvas.create_rectangle(0, 0, 450, 450,fill='white')                  # 画矩形正方形   
    x0, y0, x1, y1 = 30, 40, 70, 80
    for i in range(len(lay)):
        k = lay[i]
        text = canvas.create_text(x0, y0-15+70*i, text='第'+str(i)+'层',font=('宋体', 12),fill='black')
        for j in range(k):
            oval = canvas.create_oval(x0+60*j, y0+70*i, x1+60*j, y1+70*i, fill='yellow')  # 画圆 用黄色填充
            text = canvas.create_text(x0+60*j+20, y0+70*i+20, text='a'+str(j),fill='black')

def fram_update(layf):
    global last
    last = max(len(layf),last)
    for i in range(last):
        if i<len(layf) :
            tk.Label(frame, text='layer'+str(i),font=('黑体',11)).grid(row=i,column=0)
            # num[str(i)] = tk.StringVar()
            num[str(i)] = layf[i]
            num[str(i)] = tk.Entry(frame,width=12)
            num[str(i)].grid(row=i,column=1)
            num[str(i)].insert(0,layf[i])
        else:
            tk.Label(frame, text='-----', font=('黑体',11)).grid(row=i,column=0)
            num[str(i)] = tk.Entry(frame,width=12)
            num[str(i)].grid(row=i,column=1)
            num[str(i)].insert(0,'1')


# 第6步，触发函数，用来一定指定图形
def run_start():
    try:samt=int(layer_input.get())
    except:
        samt=2
        print ('请输入整数')
        layer_input.delete(0)
        layer_input.insert(0,'2')

    samt = samt+1
    layer1 = []
    while samt > len(layer1):
        layer1.append(1)

    for i in range(min(samt,last)):
        try:k =int( num[str(i)].get())
        except:
            k=1
            print ('请输入整数')
            layer_input.delete(0)
            layer_input.insert(0,'1')
        layer1[i] = k

    text_layer = '#layers = '+str(layer1)+' \n'

    update_canvas(layer1)
    fram_update(layer1)
    # window.update()
    try:itn=int(iter_input.get())
    except:
        itn=1000
        print ('请输入整数')
        layer_input.delete(0)
        layer_input.insert(0,'1000')
    
    text_iteration = ' #iteration = '+str(itn)+' \n'

    try:learn=float(learn_input.get())
    except:
        learn=1.2
        print ('请输入整数')
        layer_input.delete(0)
        layer_input.insert(0,'1.2')

    text_learning_rate = ' #learning_rate = ' +str(learn)
    laa.config(text=text_layer+text_iteration+text_learning_rate)

    parameters = L_layer_model(X, Y, layer1,learning_rate=learn, num_iterations = itn, print_cost = True, isPlot=True)
    
    # plt.clf()
    # plot_decision_boundary(lambda X: predict(X.T,parameters), X, np.squeeze(Y))#修改
    # plt.title("Decision Boundary for hidden layer size")
    # plt.savefig('Figure_2.png')

    # capic2 = tk.Canvas(window, bg='white',height=400, width=540)
    # capic2.grid(row=5,rowspan=6, columnspan=4)

    # rect = capic2.create_rectangle(0, 0, 500, 450,fill='white')                  # 画矩形正方形 
    
    
    # image3 = capic2.create_image(280, -80,anchor='n',image=image_file3)        # 图片锚定点（n图片顶端的中间点位置）放在画布（250,0）坐标处

    window.update()



def _quit():
    window.quit()  # 结束主循环
    window.destroy()  # 销毁窗口
    exit(0)




# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.savefig('Figure_1.png')
# shape_X = X.shape
# shape_Y = Y.shape

# layers_dims = [2, 4, 4, 2, 1] #  5-layer model
# parameters = L_layer_model(X, Y, layers_dims, num_iterations = 2000, print_cost = True,isPlot=False)

# predictions = predict(X, parameters) #训练集
# print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# # 绘制边界
# plot_decision_boundary(lambda X: predict(X.T,parameters), X, np.squeeze(Y))#修改
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.savefig('Figure_2.png')

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "noisy_circles"

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y % 2

# 第1步，实例化object，建立窗口window
window = tk.Tk()
# 第2步，给窗口的可视化起名字
window.title('My Window')
# 第3步，设定窗口的大小(长 * 宽)
window.geometry('1280x760')  # 这里的乘是小x

# 说明图片位置，并导入图片到画布上
# 定义多边形参数，然后在画布上画出指定图形

capic = tk.Canvas(window, bg='white',height=400, width=540)
capic.grid(row=0,rowspan=5,columnspan=4)

image_file = tk.PhotoImage(file=dataset+'.png')  # 图片位置（相对路径，与.py文件同一文件夹下，也可以用绝对路径，需要给定图片具体绝对路径）
image = capic.create_image(280, -50,anchor='n',image=image_file)        # 图片锚定点（n图片顶端的中间点位置）放在画布（250,0）坐标处

capic2 = tk.Canvas(window, bg='white',height=400, width=540)
capic2.grid(row=5,rowspan=6, columnspan=4)

# image_file2 = tk.PhotoImage(file='Figure.png')  # 图片位置（相对路径，与.py文件同一文件夹下，也可以用绝对路径，需要给定图片具体绝对路径）
# image2 = capic2.create_image(280, -50,anchor='n',image=image_file2)        # 图片锚定点（n图片顶端的中间点位置）放在画布（250,0）坐标处
# image_file3 = tk.PhotoImage(file='Figure.png')  # 图片位置（相对路径，与.py文件同一文件夹下，也可以用绝对路径，需要给定图片具体绝对路径）

# # 第4步，在图形界面上创建一个标签label用以显示并放置
m = Y.shape[1]  # 训练集里面的数量

text_info = "X.shape = " + str(X.shape) + "\nY.shape = "+ str(Y.shape) + "\n数据集数目：" + str(m)
ltt = tk.Label(window, bg='white', fg='black',font=('微软雅黑', 13), height=3,width=20,text=text_info)
ltt.grid(row = 0,rowspan=2,column=5)

text_layer = ' #layer = [4,5,3,2,1]   \n '
text_iteration = ' #iteration = 1000 \n'
text_learning_rate = ' #learning_rate = 1.2'
laa = tk.Label(window, bg='white', fg='black',font=('微软雅黑', 13), height=3,width=20,text=text_layer+text_iteration+text_learning_rate)
laa.grid(row = 0,rowspan=2,column=6)

scroll = tk.Scrollbar()
textt = tk.Text(window,height=10,width=50,font=('微软雅黑', 11))
scroll.config(command=textt.yview)
textt.config(yscrollcommand=scroll.set)
textt.grid(row = 9,column=5,columnspan=2)

output_txt = 'Output info:\n'
textt.insert('insert',output_txt)

# 第4步，在图形界面上创建 500 * 200 大小的画布并放置神经元数 height=400, width=500
canvas = tk.Canvas(window, bg='white',height=450, width=400)
canvas.grid(row=0,rowspan=6,column=10,columnspan=4)

# 第5步，创建一个frame，每个超参数的输入框，长在主window窗口上
input_frame = tk.Frame(window)
input_frame.grid(row=1,rowspan=3,column=5,columnspan=2)

tk.Label(input_frame,text='input #iterations:',font=('黑体',12)).grid(row=0,column=0)
iter_input=tk.Entry(input_frame)
iter_input.grid(row=0,column=1)
iter_input.insert(0,'1000')

tk.Label(input_frame,text='input #learning_rate:',font=('黑体',12)).grid(row=1,column=0)
learn_input=tk.Entry(input_frame)
learn_input.grid(row=1,column=1)
learn_input.insert(0,'1.2')

tk.Label(input_frame,text='input #layers:',font=('黑体',12)).grid(row=2,column=0)
layer_input=tk.Entry(input_frame)
layer_input.grid(row=2,column=1)
layer_input.insert(0,'2')

# 第5步，创建一个主frame，每层的神经元数量的输入框，长在主window窗口上
frame = tk.Frame(window)
frame.grid(row=2,rowspan=4,column=5,columnspan=2)

num ={}
layer = [2,3,1]
last = len(layer)
update_canvas(layer)
fram_update(layer)

b = tk.Button(window, text='  start  ',command=run_start).grid(row=5,column=5)
b = tk.Button(window, text='  quit  ',command=_quit).grid(row=5,column=6)

# 第7步，主窗口循环显示
window.mainloop()
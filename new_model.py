import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from torch.utils.data import DataLoader,TensorDataset,SubsetRandomSampler
from sklearn.metrics import classification_report
# new BiLSTM-attention model implemented by torch

# constants to load data:
GESTURE_NUM = 13
VCC = 5
time_step =14

# data preparation
def set_shuffle(setX:list,setY:list):
    """
    shuffle setX and setY, but still keep setX and setY's mapping
    return:
       setX,setY
    """
    import random
    ges_set = list(zip(setY,setX))
    keys = [i+1 for i in range(len(ges_set))]
    ges_set_dict = dict(zip(keys,ges_set))
    random.shuffle(keys)
    setX = []
    setY = []
    for i in range(len(keys)):
        ges_sample = ges_set_dict[keys[i]]
        setX.append(ges_sample[1])
        setY.append(ges_sample[0])
    return setX,setY

def splitX(dataset,time_step):
    dataX = []
    for i in range(time_step,len(dataset)+time_step,time_step):
        dataX.append(dataset[i-time_step:i,0:dataset.shape[1]])
    return np.array(dataX)

# load data
ges_dfs = [[i] for i in range(GESTURE_NUM)]
folder_path = "./gesture_data/"
for i in range(GESTURE_NUM):
    file_path = folder_path + f"gesture{i+1}.csv"
    df = pd.read_csv(file_path)
    zero_rows = (df == 0).all(axis=1)
    zero_rows_indexes = list(zero_rows[zero_rows].index)
    ges_df_list = []
    ges_dfs[i].pop()
    start = 0
    for j in range(len(zero_rows_indexes)):
        ges_df = df.iloc[start:zero_rows_indexes[j],:]
        start = zero_rows_indexes[j] + 1
        ges_df_list.append(ges_df)
    ges_dfs[i].extend(ges_df_list)

# get training set, CV set and testing set:
ges_trainingX = []
ges_trainingY = []
ges_testingX = []
ges_testingY = []


for i in range(GESTURE_NUM):
    test_split = round(len(ges_dfs[i]) * 0.10)
    ges_trainingX.extend(ges_dfs[i][:len(ges_dfs[i])-test_split]) # training set
    ges_testingX.extend(ges_dfs[i][len(ges_dfs[i])-test_split:]) # test set

    ges_trainingY.extend([i+1]*(len(ges_dfs[i])-test_split))
    ges_testingY.extend([i+1]*test_split)

# shuffle training and testing set
ges_trainingX,ges_trainingY = set_shuffle(ges_trainingX,ges_trainingY)
ges_testingX,ges_testingY = set_shuffle(ges_testingX,ges_testingY)

# shape of all sets:
print("trainX shape--","(",len(ges_trainingX),",",ges_trainingX[0].shape,")")
print("testX shape--","(",len(ges_testingX),",",ges_testingX[0].shape,")")

# concat all training data in X, and do normalization:
ges_trainingX_df = pd.concat(ges_trainingX)
ges_testingX_df = pd.concat(ges_testingX)
# #用于后续性能优化：让analog pin口读数转为电压
# ges_trainingX_df = transfrom_flex_raw(ges_trainingX_df)
# ges_testingX_df = transfrom_flex_raw(ges_testingX_df)

# transfrom flex sensor data to voltage:

scaler = MinMaxScaler(feature_range=(0,1))
ges_trainingX_scaled = scaler.fit_transform(ges_trainingX_df)
ges_testingX_scaled = scaler.fit_transform(ges_testingX_df)


ges_trainingX = splitX(ges_trainingX_scaled,time_step)
ges_trainingY = np.array(ges_trainingY) # generate an one-hot encoding for labels, this encoding is a dim=14 vector, where first element points to label=0(however, no label here =0)
ges_testingX = splitX(ges_testingX_scaled,time_step)
ges_testingY = np.array(ges_testingY)

# view shapes
print("trainX Shape-- ",ges_trainingX.shape)
print("trainY Shape-- ",ges_trainingY.shape)
print("testX Shape-- ",ges_testingX.shape)
print("testY Shape-- ",ges_testingY.shape)

# 将 NumPy 数组转换为 PyTorch Tensor
trainX_tensor = torch.tensor(ges_trainingX,dtype=torch.float32)
trainY_tensor = torch.tensor(ges_trainingY,dtype=torch.float32)
testX_tensor = torch.tensor(ges_testingX,dtype=torch.float32)
testY_tensor = torch.tensor(ges_testingY,dtype=torch.float32)

# 创建 TensorDataset
train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
test_dataset = TensorDataset(testX_tensor, testY_tensor)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 测试集通常不打乱

# training:
# constants for training:
num_layers = 1  #一层lstm
num_directions = 2  #双向lstm
lr = 1e-3 # 学习率
batch_size = 16
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classses =GESTURE_NUM+1
time_step = 14
input_size = 10
hidden_size = 100 # how many hidden layer on LSTM
folds = 5


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_directions, num_classes, dropout_prob):
        super(BiLSTMModel, self).__init__()

        self.input_size = input_size  # 数据维度
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions

        self.lstm1 = nn.LSTM(input_size, hidden_size,
                             num_layers=num_layers,batch_first=True, bidirectional=(num_directions == 2))
        self.lstm2 = nn.LSTM(hidden_size, hidden_size,
                             num_layers=num_layers,batch_first=True, bidirectional=(num_directions == 2))
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.liner = nn.Linear(hidden_size, num_classes)
        self.act_func = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x: input data with shape: [batch_size, time_step, embedding_size]
        """
        # x [batch_size, time_step, embedding_size]
        # LSTM 期望将时间步放在最前面
        # 由于数据集不一定是预先设置的batch_size的整数倍，所以用size(0)获取当前数据实际的batch
        batch_size = x.shape[0]

        # LSTM最初的前向输出，即记忆单元C和隐状图H
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

        # 第一层LSTM
        # out[seq_len, batch, num_directions * hidden_size]。多层lstm，out是每个时间步t的输出h_t
        # h_n, c_n [num_layers(1) * num_directions, batch, hidden_size]，h_n是当前层最后一个隐状态输出
        out, (h_n, c_n) = self.lstm1(x, (h_0, c_0))
        # 第一层Dropout，
        out = self.dropout(out)
        # 双向LSTM输出拆成前向和后向输出
        # 将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim=2)
        out = forward_out + backward_out  # [batch,seq_len, hidden_size]

        # 第二层LSTM
        out, (h_n, c_n) = self.lstm2(out, (h_n, c_n))
        # 第二层Dropout
        out = self.dropout(out)

        # 再次将双向LSTM的输出拆分为前向输出和后向输出，并求和
        (forward_out, backward_out) = torch.chunk(out, 2, dim=2)
        out = forward_out + backward_out  # [seq_len, batch, hidden_size]

        # print("LSTM out",out)
        # 为了使用到lstm最后一个时间步时每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  # [batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1)  # [batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  # [batch, hidden_size]
        # 得到attention层的权重
        attention_w = self.attention_weights_layer(h_n)  # [batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1)  # [batch, 1, hidden_size]
        m = nn.Tanh()(out)
        # torch.bmm: 矩阵乘法,下文m.transpose(1,2)就是将原来维度从 [batch, seq_len, hidden_size] 转换为 [batch, hidden_size, seq_len]，以方便下面进行矩阵乘法
        attention_context = torch.bmm(attention_w, m.transpose(1, 2))  # [batch, 1, seq_len]
        # print("attention_context",x,sep='\n')
        softmax_w = F.softmax(attention_context, dim=-1)  # [batch, 1, seq_len], 用softmax对刚才的权重归一化,a-> a'

        x = torch.bmm(softmax_w, out)  # [batch, 1, hidden_size] 抽取sequence内的重要信息
        x = x.squeeze(dim=1)  # [batch, hidden_size]
        # print("after squeeze:",x,sep='\n')
        x = self.liner(x)
        # x = self.act_func(x)
        return x

def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)

        preds = model(datas)
        loss = loss_func(preds, labels.long())
        loss_val += loss.item() * datas.size(0)

        # 获取预测的最大概率出现的位置
        preds = nn.Softmax(dim=1)(preds)
        preds = torch.argmax(preds, dim=1)
        # labels = torch.argmax(labels, dim=0)
        corrects += torch.sum(preds == labels).item()
    test_loss = loss_val / len(test_loader.dataset)  # 计算整个测试集的总损失
    test_acc = corrects / len(test_loader.dataset)  # 计算整个测试集的总正确率
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return test_loss, test_acc


def train(model, train_loader, optimizer, loss_func, epochs,k_fold):
    """
    define training process of upon model
    return:
    model: the model after training
    train_loss_epochs: loss for each epoch in training
    train_acc_epochs: accuracy for each epoch in training
    CV_loss_epochs: loss for each epoch in validation
    CV_acc_epochs: accuracy for each epoch in validation
    """
    k = k_fold  # k折交叉验证
    best_val_acc_list = [0.0 for i in range(k)]  # 跟踪每个折最佳验证准确率
    best_models = []  # 记录每一折训练得到的最优model，然后根据正确率得到一个全局最优model
    best_model_params = copy.deepcopy(model.state_dict())  # 创建当前模型参数的深拷贝
    kf = KFold(n_splits=k)
    average_val_acc = 0.0
    train_loss_folds = []
    train_acc_folds = []
    CV_loss_folds = []
    CV_acc_folds = []
    for fold, (train_indexes, val_indexes) in enumerate(kf.split(train_loader.dataset)):
        train_sampler = SubsetRandomSampler(
            train_indexes)  # 告诉dataloader应该加载与len(train_indexes)数量相同，与train_indexes对应的样本
        val_sampler = SubsetRandomSampler(val_indexes)
        curr_train_loader = DataLoader(train_loader.dataset, batch_size=16, sampler=train_sampler)
        val_loader = DataLoader(train_loader.dataset, batch_size=16, sampler=val_sampler)

        # 记录训练损失和正确率，用于画图：
        train_loss_epochs = []
        train_acc_epochs = []
        # 记录每次验证的损失和正确率，用于画图：
        CV_loss_epochs = []
        CV_acc_epochs = []
        for epoch in range(epochs):
            print(f"Fold{fold + 1}, epoch{epoch + 1}:...")
            model.train()  # 设置为训练模式
            loss_val = 0.0
            corrects = 0.0
            for datas, labels in curr_train_loader:
                # datas: (batch_size,input_size(14),features(10))
                # labels: (batch_size,input_size(14))
                datas = datas.to(device)
                labels = labels.to(device)

                preds = model(datas)  # 前向传播
                # print(labels.long())
                loss = loss_func(preds, labels.long())  # 计算损失,tensor大小是1

                optimizer.zero_grad()  # 清除优化器梯度（来自于上一次反向传播）
                loss.backward()  # 反向传播, 计算模型参数梯度
                optimizer.step()  # 根据计算得到的梯度，使用优化器更新模型的参数。

                loss_val += loss.item() * datas.size(0)  # 获取loss，并乘以当前批次大小

                # 获取预测的最大概率出现的位置
                preds = nn.Softmax(dim=1)(preds)
                preds = torch.argmax(preds, dim=1)
                # labels = torch.argmax(labels, dim=0)
                corrects += torch.sum(preds == labels).item()
            train_loss = loss_val / len(curr_train_loader.dataset)  # 计算整个模型的总损失
            train_acc = corrects / len(curr_train_loader.dataset)  # 计算整个模型的总正确率
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            train_loss_epochs.append(train_loss)
            train_acc_epochs.append(train_acc)
            # if(epoch % 2 == 0): 每个epoch都进行评估：
            val_loss, val_acc = test(model, val_loader, loss_func)
            if (best_val_acc_list[fold] < val_acc):  # 出现最优模型时，保存最优模型
                best_val_acc_list[fold] = val_acc
                best_model_params = copy.deepcopy(model.state_dict())
            # 更新平均accuracy指标
            average_val_acc += val_acc
            CV_loss_epochs.append(val_loss)
            CV_acc_epochs.append(val_acc)
        model.load_state_dict(best_model_params)
        best_models.append(model)
        train_loss_folds.append(train_loss_epochs)
        train_acc_folds.append(train_acc_epochs)
        CV_loss_folds.append(CV_loss_epochs)
        CV_acc_folds.append(CV_acc_epochs)
        print(len(CV_loss_folds), len(CV_acc_folds), sep=",")

    # 计算所有折的平均验证accuracy
    average_val_acc = average_val_acc / (k * epochs)
    print(f'Average Validation Accuracy: {average_val_acc:.4f}')
    # 找到所有折中最好的model作为返回model，以及当前折的loss和acc记录返回
    best_val_acc_index = np.argmax(np.array(best_val_acc_list))
    print(best_val_acc_index)
    model = best_models[best_val_acc_index]
    train_losses = train_loss_folds[best_val_acc_index]
    train_accs = train_acc_folds[best_val_acc_index]
    CV_losses = CV_loss_folds[best_val_acc_index]
    CV_accs = CV_acc_folds[best_val_acc_index]
    return model, train_losses, train_accs, CV_losses, CV_accs

# inferencing:
model = BiLSTMModel(input_size, hidden_size, num_layers, num_directions, num_classses,dropout_prob=0.1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss() # 与categorical crossEntropyLoss不同，CrossEntropyLoss期望输出是一个类索引，而不是独热编码
model,train_loss,train_acc,CV_loss,CV_acc = train(model, train_loader,optimizer, loss_func, epochs,k_fold=folds)
torch.save(model.state_dict(), 'BiLSTM_Attention_weights.pth')
# evaluate
test(model,test_loader,loss_func)

# plot performance:
# 画曲线
def plot_performance(epochs, Train_loss, Train_acc, cv_loss, cv_acc):
    xlabel = "Epoch"
    legends = ["Training", "Validation"]

    plt.figure(figsize=(20, 5))
    epochs_list = [i for i in range(epochs)]
    train_acc = Train_acc
    CV_acc = cv_acc
    epochs_list_show = [i for i in range(0, epochs, 2)]
    min_y = min(min(train_acc), min(CV_acc))
    max_y = max(max(train_acc), max(CV_acc))

    plt.subplot(121)

    plt.plot(epochs_list, train_acc)
    plt.plot(epochs_list, CV_acc)

    plt.title("Model Accuracy\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.xticks(epochs_list_show, epochs_list_show)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc="upper left")
    plt.grid()

    train_loss = Train_loss
    CV_loss = cv_loss

    min_y = min(min(train_loss), min(CV_loss))
    max_y = max(max(train_loss), max(CV_loss))

    plt.subplot(122)

    plt.plot(epochs_list, train_loss)
    plt.plot(epochs_list, CV_loss)

    plt.title("Model Loss:\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.xticks(epochs_list_show, epochs_list_show)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc="upper left")
    plt.grid()
    plt.savefig("./results/model_acc_loss_curve.jpg",format="jpg")

def plot_conf_matrix(conf_matrix):
    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(8, 6))  # 设置图形的大小
    img = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues,aspect='auto')  # 使用 'nearest' 插值方法，避免数值模糊
    plt.title('Confusion Matrix')  # 设置图形的标题
    plt.colorbar(img)  # 显示颜色条

    # 在混淆矩阵的每个单元格中添加文本标签
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),  # 'd' 表示整数格式
                    ha="center", va="center", color="red",fontsize=12)

    plt.xlabel('Predicted Label')  # 设置 x 轴的标签 , 测试集sample手势对应的索引
    plt.ylabel('True Label')  # 设置 y 轴的标签, 测试集label手势对应的索引

    # 显示图形
    plt.tight_layout()
    plt.savefig("./results/model_conf_matrix.jpg",format="jpg")

# loss curve
plot_performance(epochs,train_loss,train_acc,CV_loss,CV_acc)
# metrics: accuracy, precision, recall, F1 score
prediction_Y = model(testX_tensor)
pred_Y = torch.argmax(prediction_Y,axis=1)
test_accuracy = accuracy_score(testY_tensor,pred_Y)
test_precision = precision_score(testY_tensor,pred_Y,average='macro') #'macro'：计算每个类别的得分，然后计算它们的平均值（在这里，类别权重相等）。
test_recall = recall_score(testY_tensor,pred_Y,average='macro')
test_f1 = f1_score(testY_tensor,pred_Y,average='macro')
print(f"test accuracy:{test_accuracy:.4f}")
print(f"test_precision:{test_precision:.4f}")
print(f"test_recall:{test_recall:.4f}")
print(f"test F1 score:{test_f1:.4f}")
# classification report: 识别在哪些类别上分类表现良好，哪些表现不佳
report = classification_report(testY_tensor,pred_Y)
print(report)
# confusion matrix:
conf_matrix = confusion_matrix(testY_tensor,pred_Y)
plot_conf_matrix(conf_matrix)
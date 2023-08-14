import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from config import Config
from datasets import SSTreebankDataset
from utils import adjust_learning_rate, accuracy, save_checkpoint, AverageMeter, train, validate, testing

from my_sru import SRU
import time

class ModelConfig(object):
    '''
    模型配置参数
    '''
    # 全局配置参数
    opt = Config()

    # 数据参数
    output_folder = opt.output_folder
    data_name = opt.data_name
    SST_path  = opt.SST_path
    emb_file = opt.emb_file
    emb_format = opt.emb_format
    output_folder = opt.output_folder
    min_word_freq = opt.min_word_freq
    max_len = opt.max_len

    # 训练参数
    epochs = 100  # epoch数目，除非early stopping, 先开20个epoch不微调,再开多点epoch微调
    batch_size = 32 # batch_size
    workers = 4  # 多处理器加载数据
    lr = 1e-3  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
    weight_decay = 0 # 权重衰减率
    decay_epoch = 0 # 多少个epoch后执行学习率衰减
    improvement_epoch = 100 # 多少个epoch后执行early stopping
    is_Linux = False # 如果是Linux则设置为True,否则设置为else, 用于判断是否多处理器加载
    print_freq = 100  # 每隔print_freq个iteration打印状态
    checkpoint = None  # 模型断点所在位置, 无则None
    best_model = None # 最优模型所在位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    model_name = 'TextSRU' # 模型名
    class_num = 5 if data_name == 'SST-1' else 2 # 分类类别
    dropout = 0.5 # dropout
    embed_dim = 128 # 未使用预训练词向量的默认值
    static = True # 是否使用预训练词向量, static=True, 表示使用预训练词向量
    non_static = True # 是否微调，non_static=True,表示微调


class ModelSRU(nn.Module):
    """text classification using Young's SRU implementation, based on Kim's CNN paper"""
    def __init__(self, vocab_size, embed_dim, class_num, pretrain_embed, dropout, static, non_static):
        super().__init__()
        
        # 随机初始化词向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 使用预训练词向量
        if static:
            self.embedding = self.embedding.from_pretrained(pretrain_embed)

        hidden_dim = 128

        self.sru = SRU(embed_dim, hidden_dim, 4, False, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, class_num)
    
    def forward(self, x):
        
        x = self.embedding(x) # (N, W, I)

        x = torch.permute(x, (1,0,2)) # (W, N, I)

        h, _ = self.sru(x) # (W, N, D) (L, N, D)

        x = h[-1] # (N, D)

        x = self.dropout(x) # still (N, D)

        x = self.fc(x) # logits, (N, C)
        
        return x 



def train_eval(opt):
    '''
    训练和验证
    '''
    # 初始化best accuracy
    best_acc = 0.

    # epoch
    start_epoch = 0
    epochs = opt.epochs
    epochs_since_improvement = 0  # 跟踪训练时的验证集上的BLEU变化，每过一个epoch没提升则加1

    # 读入词表
    word_map_file = opt.output_folder +  opt.data_name + '_' + 'wordmap.json'
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # 加载预训练词向量
    embed_file = opt.output_folder + opt.data_name + '_' + 'pretrain_embed.pth'
    embed_file = torch.load(embed_file)
    pretrain_embed, embed_dim = embed_file['pretrain'], embed_file['dim']

    # 初始化/加载模型
    if opt.checkpoint is None:
        if opt.static == False: embed_dim = opt.embed_dim
        model = ModelSRU(vocab_size=len(word_map), 
                      embed_dim=embed_dim, 
                      class_num=opt.class_num,
                      pretrain_embed=pretrain_embed,
                      dropout=opt.dropout, 
                      static=opt.static, 
                      non_static=opt.non_static)
    
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=opt.lr)
        
    else:
        # 载入checkpoint
        checkpoint = torch.load(opt.checkpoint, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_acc = checkpoint['acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    # 移动到GPU
    model = model.to(opt.device)
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(opt.device)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(
                    SSTreebankDataset(opt.data_name, opt.output_folder, 'train'),
                    batch_size=opt.batch_size, 
                    shuffle=True,
                    num_workers = opt.workers if opt.is_Linux else 0,
                    pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
                    SSTreebankDataset(opt.data_name, opt.output_folder, 'dev'),
                    batch_size=opt.batch_size, 
                    shuffle=True,
                    num_workers = opt.workers if opt.is_Linux else 0,
                    pin_memory=True)
    
    # timing stuff

    start_time = time.time()

    # Epochs
    for epoch in range(start_epoch, epochs):
        
        # 一个epoch的训练
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              vocab_size=len(word_map),
              print_freq=opt.print_freq,
              device=opt.device)
        
        # 一个epoch的验证
        recent_acc = validate(val_loader=val_loader,
                              model=model,
                              criterion=criterion,
                              print_freq=opt.print_freq,
                              device=opt.device)
        
        # 检查是否有提升
        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("Epochs since last improvement:", epochs_since_improvement)
        else:
            epochs_since_improvement = 0
        
        # 保存模型
        save_checkpoint(opt.model_name, opt.data_name, epoch, epochs_since_improvement, model, optimizer, recent_acc, is_best)
        
        cur_time = time.time()
        print("Wall time since start:", cur_time-start_time)
        print()

def test(opt):

    # 载入best model
    best_model = torch.load(opt.best_model, map_location='cpu')
    model = best_model['model']

    # 移动到GPU
    model = model.to(opt.device)

    # loss function
    criterion = nn.CrossEntropyLoss().to(opt.device)

    # dataloader
    test_loader = torch.utils.data.DataLoader(
        SSTreebankDataset(opt.data_name, opt.output_folder, 'test'),
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers = opt.workers if opt.is_Linux else 0,
        pin_memory=True)
    
    # test
    testing(test_loader, model, criterion, opt.print_freq, opt.device)


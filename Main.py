import random, os
import torch
import numpy as np
import argparse
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from DataProcess import CustomDataSet,collate_fn
from Model import DGEDTI
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
def get_kfold_data(i, datasets, k=5):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(datasets) // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]  # 若不能整除，将多的case放在最后一折里
        trainset = datasets[:val_start]

    return trainset, validset


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def train(model, train_data_loader, args):

    t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.epochs#计算总的训练步数  (总批次数量 // 梯度累积步数) × 训练轮数

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)#输入要优化的可学习参数，指定学习率，创建一个优化器对象赋给变量

    #创建带预热的学习率调度器,创建学习率从 0 线性增加到初始学习率，然后再线性衰减到 0 的调度器。
    if 0 < args.warmup_steps < 1:#如果 warmup_steps是 0 到 1 之间的小数，则视为比例,将其转换为具体的步数
        args.warmup_steps = int(args.warmup_steps * t_total)

    #前 warmup_steps 步从 0 线性增加到初始学习率,warmup_steps 步之后从初始学习率线性衰减到 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    train_iterator = trange(
        0, int(args.epochs), desc="Epoch", disable=False
    )
    for epoch in train_iterator:
        model.train()
        epoch_iterator = tqdm(BackgroundGenerator(train_data_loader), total=len(train_data_loader), desc="training")
        for step, data in enumerate(epoch_iterator):
            compounds, proteins, labels = [d.cuda() for d in data]
            predicts = model.forward(compounds, proteins)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Drugbank', help='dataset name: Davis, Drugbank, KIBA')
    parser.add_argument('--data_path', type=str, default=r'/root/data1/data/DGE-DTI/Drugbank', help='data path')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup steps')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--train_batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--epochs', type=int, default=80, help='the number of epochs to train for')
    parser.add_argument('--patience', type=int, default=20, help='the number of epochs to train for')
    parser.add_argument('--k_fold', type=int, default=5, help='the number of epochs to train for')
    parser.add_argument('--seed', type=int, default=4321, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--save_path', type=str, default=r'/root/data1/data/DGE-DTI/Drugbank/result', help='save path')

    args = parser.parse_args()

    args.save_path = rf'/root/data1/data/DGE-DTI/Results/{args.dataset}/lr_{args.lr}/batchsize_{args.train_batch_size}'
    set_seed(args)
    print("Train in {}".format(args.dataset))
    with open(os.path.join(args.data_path, f"{args.dataset}.txt"), 'r') as f:
        cpi_list = f.read().strip().split('\n')
    print("load finished")
    print("data shuffle")
    dataset = shuffle_dataset(cpi_list, args.seed)
    for i_fold in range(args.k_fold):
        args.output_dir = rf"{args.save_path}/{i_fold + 1}_Fold"
        print('*' * 25)
        print('第' + str(i_fold + 1) + '折')
        print('*' * 25)
        trainset, testset = get_kfold_data(i_fold, dataset, k=args.k_fold)
        TVdataset = CustomDataSet(trainset)
        test_dataset = CustomDataSet(testset)
        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=args.num_workers, collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                        collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate_fn)
        model = DGEDTI().cuda()

        global_step, tr_loss = train(model, train_dataset_load,args)

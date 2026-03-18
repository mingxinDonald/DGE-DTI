import torch
import torch.nn as nn
class DGEDTI(nn.Module):
    def __init__(self,conv=64,char_dim=128,protein_kernel=[4, 8, 12],drug_kernel=[4, 6, 8]):
        super(DGEDTI, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.drug_kernel = drug_kernel
        self.protein_kernel = protein_kernel
        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)#padding_idx=0的作用是指定填充符号的索引是0，并将该索引对应的嵌入向量固定为零向量且不参与训练。1. 在训练时不会被更新（梯度为 0）2. 始终是零向量3. 不影响模型计算
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)#词汇表大小是65，词向量长度是128
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug)

        proteinembed = self.protein_embed(protein)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        return drugConv
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    #将源语言句子从文字转换成token id
    #[1,5]

    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position, d_model):
    
    def cal_angle(position, hid_idx):
        #position:词在序列中的位置(0,1,2,3,4...)
        #hid_idx:向量的第几个维度(0,1,2,...,511)

        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    
    def get_posi_angle_vec(position):
        #为某个位置生成完整的512维角度向量。
        # get_posi_angle_vec(1) = [
        #     cal_angle(1, 0),  #维度0的角度
        #     cal_angle(1, 1),  #维度1的角度
        #     cal_angle(1, 2),  #维度2的角度
        #     ...
        #     cal_angle(1, 511) #维度511的角度
        # ]
        #返回长度为512的列表，此时还是原始角度值，sin/cos在外面统一施加

        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    #这行代码构建了完整的位置编码矩阵
    #遍历pos_i=0,1,2,3,4,5 每个位置生成一个512维的角度向量，最终堆叠成矩阵
    # [
    #     get_posi_angle_vec(0),   #第0行：位置0的512个角度值（padding用）
    #     get_posi_angle_vec(1),   #第1行：位置1的512个角度值(ich)
    #     get_posi_angle_vec(2),   #第2行：位置2的512个角度值(mochte)
    #     get_posi_angle_vec(3),   #第3行：位置3的512个角度值(ein)
    #     get_posi_angle_vec(4),   #第4行：位置4的512个角度值(bier)
    #     get_posi_angle_vec(5)    #第5行：位置5的512个角度值（备用）
    # ]
    #np.array(...)将列表转为形状为[6,512]的numpy矩阵。

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    #对矩阵的所有偶数列统一施加sin函数

    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    #对矩阵的所有奇数列统一施加cos函数

    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    #获取seq_q的第0维和第1维
    #batch_size=1,len_q=5

    batch_size, len_k = seq_k.size()
    #batch_size=1,len_k=5

    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    #生成原始padding_mask
    #(1)seq_k.data
    #取底层数据，与seq_k等价
    #(2).eq(0),找出padding位置
    #(3).unsqueeze(1),在第1维插入新维度
    #pad_attn_mask的维度是(1,1,5)

    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k
    #将维数扩展到(1,5,5)

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    #[1,5,5]

    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    #生成上三角矩阵
    #k=1表示保留对角线以上的元素，其余置为0

    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    #这行将numpy数组转为PyTorch的byte张量

    return subsequent_mask
    #返回值subsequent_mask的维数为[1,5,5]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        #(1)K.transpose(-1,-2)转置K的最后两维
        #K   [1,8,5,64]
        #,transpose(-1,-2) #交换最后两个维度
        #K^T   [1,8,64,5]
        #(2)...
        #最后scores的维度是[1,8,5,5]

        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        #这行将padding位置的得分填充为负无穷

        attn = nn.Softmax(dim=-1)(scores)
        #沿scores的最后一个维度做softmax
        #attn的维度为[1,8,5,5]

        context = torch.matmul(attn, V)
        #用注意力权重对V做加权求和，得到最终的注意力输出。
        #context的维度为[1,8,5,64]

        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        #创建Query的投影矩阵

        self.W_K = nn.Linear(d_model, d_k * n_heads)
        #创建Key的投影矩阵

        self.W_V = nn.Linear(d_model, d_v * n_heads)
        #创建Value的投影矩阵

        self.linear = nn.Linear(n_heads * d_v, d_model)
        #创建多头注意力的输出投影矩阵

        self.layer_norm = nn.LayerNorm(d_model)
        #创建了层归一化组件

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        #residual=Q,保存输入，用于后面的残差连接
        #batch_size=Q.size(0),取Q的第0维大小

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        #对Q做投影，并拆分成多个头
        #(1)self.W_Q(Q)，线性投影
        #Q [1, 5, 512]  W_Q [512, 512]    [1, 5, 512]
        #(2).view(batch_size, -1, n_heads, d_k),重塑形状
        #[1, 5, 512]
        #, view(1, -1, 8, 64)  512=8*64 拆开最后一维
        #[1, 5, 8, 64]
        #-1由PyTorch自动推断，seq_len=5
        #(3).transpose(1,2),交换维度
        #[1,5,8,64]
        #,交换第1维和第2维
        #[1,8,5,64]
        #[batch, n_heads, seq_len, d_k]

        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        #将mask扩展到多头维度，让每一个头都有一份相同的mask
        #(1)attn_mask的初始形状
        #attn_mask [1,5,5]
        #[batch_size, len_q, len_k]
        #由get_attn_pad_mask生成
        #(2).unsqueeze(1),插入头维度
        #[1,5,5]
        #,在头上插入一个新维度
        #[1,1,5,5]
        # [batch,1个头，len_q,len_k]
        #(3).repeat(1, n_heads, 1, 1),沿头维度复制8份
        #[1,1,5,5]
        #,repeat(1, 8, 1, 1)
        #[1,8,5,5]
        #8个头各有一份完全相同的mask

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #调用缩放点积注意力
        #返回context [1,8,5,64]表示注意力加权后的输出，送入后续计算
        #返回attn [1,8,5,5] 注意力权重矩阵，用于可视化（看模型关注了哪些词）

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        #这行将多头输出拼接回单个张量。
        #(1)context.transpose(1,2),交换头维度和序列精度。
        #[1,8,5,64]
        #,交换第1维和第2维
        #[1,5,8,64]
        #(2).contiguous,使内存连续
        #(3).view(batch_size,-1,n_heads*d_v),合并头维度
        #[1,5,8,64]
        #,view(1,-1,8*64)
        #[1,5,512]

        output = self.linear(context)
        #对拼接后的多头输出做输出投影
        #它让模型学习如何融合8个头的信息，对512维向量做一次全局线性变换，使各头的输出相互影响、整合成统一表示。

        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]
        #完成残差连接 + 层归一化

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        #这行创建了FFN的第一个卷积层，实现升维。
        #in_channels=512
        #out_channels=2048

        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        #这行创建了FFN的第二个卷积层，实现降维

        self.layer_norm = nn.LayerNorm(d_model)
        #d_model=512
        #对每个token的512维向量独立做归一化。

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]

        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        #这完成FFN的第一步：升维+激活
        #(1)inputs.transpose(1,2),转置适配Conv1d
        #inputs   [1, 5, 512]
        #, transpose(1, 2)
        #         [1, 512, 5]
        #(2)self.conv1(...),升维
        #[1, 512, 5]
        #, Conv1d(512->2048,kernel=1)
        #[1, 2048, 5]
        #(3)nn.ReLU(...),激活函数
        #[1, 2048, 5]
        #, ReLU
        #[1, 2048, 5]

        output = self.conv2(output).transpose(1, 2)
        #这完成FFN的第二步：降维+转回原始格式
        #(1)self.conv2(output),降维
        #output    [1, 2048, 5]   上一行conv1+ReLU的输出
        #, Conv1d(2048->512, kernel=1)
        #          [1, 512, 5]
        #(2).transpose(1, 2), 转回Transformer标准格式
        #[1,512,5]
        #,transpose(1,2)
        #[1,5,512]

        return self.layer_norm(output + residual)
        #这行完成FFN子层的Add & Norm。

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention()
        #创建头多注意力

        self.pos_ffn = PoswiseFeedForwardNet()
        #创建逐位置前馈网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        #这行调用编码自注意力
        #ich  ->  作为Q: 我想关注哪些词？
        #ich  ->  作为K: 我能被哪些词关注？
        #ich  ->  作为V: 被关注时我能贡献哪些信息？
        #返回值enc_outputs,[1,5,512],每个词融合了全局上下文的新表示。
        #返回值attn,[1,8,5,5]注意力权重矩阵，可视化用

        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        #将注意力输出送入FFN子层

        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        #创建了Decoder的自注意力层

        self.dec_enc_attn = MultiHeadAttention()
        #创建了Decoder的交叉注意力层

        self.pos_ffn = PoswiseFeedForwardNet()
        #创建了Decoder的FFN

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        #调用Decoder的Masked自注意力
        #Q=K=V=dec_inputs
        #返回值dec_outputs的维数[1,5,512]
        #返回值dec_self_attn的维数[1,8,5,5]

        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        #调用Decoder的交叉注意力，这是Encoder和Decoder之间信息传递的核心
        #Q=dec_outputs
        #K=V=enc_outputs
        #返回值dec_outputs的维数[1,5,512]
        #返回值dec_enc_attn的维数[1,8,5,5]

        dec_outputs = self.pos_ffn(dec_outputs)
        #FFN
        #返回值dec_outputs的维数[1,5,512]

        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        #这行代码创建了源语言的词嵌入层。

        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        #创建了位置编码层[6, 512],冻结参数，不更新。

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        #创建了6个独立的EncoderLayer并用nn.ModuleList管理。

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        #enc_inputs是编码器的输入，即源语音句子经过数字化后的token ID序列
        
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        #将词嵌入和位置编码相加
        #shape [1, 5, 512]

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        #这行生成编码器自注意力的Padding Mask

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
        #返回值enc_outputs的维度是[1,5,512]
        #返回值enc_self_attns的维度：列表，包含6个[1,8,5,5]张量

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        #目标语言的词嵌入层
        #nn.Embedding(7,512)

        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        #位置编码层
        #[6,512]

        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        #目标语言嵌入+位置编码
        #[1,5,512]

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        #生成Decoder自注意力的Padding Mask
        #[1,5,5]

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        #这行生成Causal Mask(因果掩码)
        #[1,5,5]

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        #这行将两个mask合并成最终的Decoder自注意力mask
        #[1,5,5]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        #这行生成Decoder交叉注意力的Padding Mask

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        #这行代码构建了完整的编码器

        self.decoder = Decoder()
        #这行代码构建了完整的编码器

        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        #这行代码创建了Transformer的最终输出投影层
        #d_model=512
        #tgt_vocab_size=7

    def forward(self, enc_inputs, dec_inputs):
        #enc_inputs  [1,5]
        #dec_inputs  [1,5]

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        #编码器
        #enc_outputs [1,5,512]
        #enc_self_attns 长度为6的列表，每个元素[1,8,5,5]

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        #解码器
        #dec_outputs [1,5,512]
        #dec_self_attns 长度为6的列表，每个元素[1,8,5,5]
        #dec_enc_attns  长度为6的列表，每个元素[1,8,5,5]

        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        #将Decoder输出投影到词汇表空间
        #dec_logits  [1, 5, 7]

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
        #ec_logits.view(-1, dec_logits.size(-1))   维数(5,7)

def showgraph(attn):
    #输入attn为长度为6的列表，每个元素的维度为[1,8,5,5]

    attn = attn[-1].squeeze(0)[0]
    #从注意力权重列表中提取最后一层第一个头的注意力矩阵
    #(1)attn[-1],取最后一层，即第6层的注意力权重[1,8,5,5]
    #(2).squeeze(0),去掉batch维度。
    #[1,8,5,5]
    # squeeze(0)
    #[8,5,5]
    #(3)[0],取第一个头
    #[8,5,5]
    # [0]
    #[5,5]

    attn = attn.squeeze(0).data.numpy()
    #将注意力矩阵转换为numpy数组以便可视化
    #.squeeze(0),尝试去掉第0维，第0维不是1,squeeze无效，形状不变。
    #[5,5]

    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    #创建一个matplotlib画布

    ax = fig.add_subplot(1, 1, 1)
    #在画布上添加一个子图

    ax.matshow(attn, cmap='viridis')
    #将注意力矩阵绘制为热力图
    #attn,要可视化的矩阵[5,5]
    #cmap='viridis',颜色映射方案。
    #值越大，颜色越亮（黄色），高注意力。
    #值越小，颜色越暗（深紫），低注意力。

    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    #设置x轴的刻度标签（源语言词）

    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    #这行设置y轴的刻度标签（目标语言词）

    plt.show()

if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    #这是一个batch，包含encoder输入、decoder输入、decoder输出。
    #其中P表示填充。S表示开始。E表示结束。

    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    #这行代码定义了一个源语言词汇表。

    src_vocab_size = len(src_vocab)
    #获取源语言词汇表的大小
    #它被用来定义Transformer中的Embedding层。
    #nn.Embedding(src_vocab_size, d_model)
    #Embedding层本质上是一个矩阵，行数必须等于词汇表大小。

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    #这行定义了目标词汇表
    #'P' -> 0, Padding token填充用
    #‘S’ -> 5, Start token, 句子开始标记
    #'E' -> 6, End token, 句子结束标记

    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    #这行代码用字典推导式创建了一个反向词汇表。
    #用于解码阶段将模型输出的数字还原为单词。

    tgt_vocab_size = len(tgt_vocab)
    #获取目标词汇表的大小
    #在Transformer中有两个用途：
    #(1)Decoder的Embedding层，nn.Embedding(tgt_vocab_size, d_model)
    #(2)最终的线性输出层，nn.Linear(d_model, tgt_vocab_size). 输出层的维度必须等于词汇表大小，经过softmax后每一维代表对应词的概率，模型选概率最大的词作为预测结果。

    src_len = 5 # length of source
    #这行定义了源序列的固定长度
    #Transformer处理的是矩阵，批次中所有句子必须等长，短句子用P(index=0)填充到src_len长度。
    #它会影响哪些地方？
    #(1)Padding mask:生成一个mask屏蔽掉P的位置，防止注意力机制关注无意义的填充。
    #(2)Positional Encoding:位置编码矩阵的行数=src_len
    #(3)注意力矩阵:Encoder自注意力的矩阵大小src_len * src_len

    tgt_len = 5 # length of target
    #定义目标序列的固定长度
    #对应目标语言句子在训练时的结构，共5个位置
    #[S, i, want, a, beer, E]
    #但实际上decoder的输入和输出是错开一位的:
    #Decoder输入: [S, i, want, a, beer]
    #Decoder输出: [i, want, a, berr, E]
    #即输入从S开始，输出是输入整体右移一位的结果，这就是Teacher Forcing训练方式。
    #它影响的地方：
    #(1)Decoder的Padding mask: 屏蔽目标序列中的P位置。
    #(2)Causal mark(因果掩码)：大小为tgt_len * tgt_len,确保位置i只能看到位置<=i的词，防止看未来
    #(3)cross-attention:注意力矩阵大小为tgt_len*src_len,即decoder每个位置对encoder所有位置做注意力

    d_model = 512  # Embedding Size
    #这行定义了Transformer中最核心的超参数，模型维度
    #每个token被表示为一个512维的向量，这个维度贯穿整个模型始终不变。
    #它出现在模型的每个地方：
    #(1)nn.Embedding(vocab_size, 512),词向量查找表，每个词对应一个512维向量。
    #(2)Positional Encoding, 位置编码也是512维，直接与词向量相加。
    #(3)Q,K,V矩阵，注意力的查询/键/值向量维度基于512
    #(4)FFN层输入输出，前馈网络输入和输出都是512维
    #(5)nn.Linear(512, tgt_vocab_size), 最终输出层从512维映射到词汇表。
    #为什么维度要保持不变？这样残差连接才能成立：
    #output = LayerNorm(x + SubLayer(x))
    #512是原始论文中使用的值，是模型容量和计算成本之间的经典权衡。

    d_ff = 2048  # FeedForward dimension
    #这行定义了前馈网络FFN的隐藏层维度：
    #FFN的结构：
    #512  ->  2048  ->  512
    #d_model            d_model
    #        d_ff
    #具体计算：
    #FFN内部
    #x = Linear(d_model, d_ff)(x) #512->2048,升维
    #x = ReLU(x)
    #x = Linear(d_ff, d_model)(x) #2048->512,降维
    #为什么是4倍？原始论文中d_ff=4*d_model,这是一个经验设计。
    #(1)先升维到更高空间，让模型有足够容量学习复杂的非线性变换。
    #(2)再降回d_model,保持残差连接的维度一致性。
    #FFN的作用：注意力机制负责聚合信息（词与词之间的交互），FFN负责处理信息（对每个位置独立做非线性变换）。2048的宽度给了模型足够的表达能力来完成这个变换。
    #实际上FFN层的参数量占了Transformer总参数量的大头，两个线性层共有512*2048+2048*512约等于200万个参数。

    d_k = d_v = 64  # dimension of K(=Q), V
    #这行定义了注意力机制中K,Q,V向量的维度
    #为什么是64？这来自多头注意力的设计：
    #d_model / n_heads = 512 / 8 = 64
    #原始论文使用8个注意力头，每个头分到512/8=64维，所有头并行计算后拼接回512维。
    #每个头内部的计算：
    #Q = Linear(d_model, d_k)(x)  #512->64
    #K = Linear(d_model, d_k)(x)  #512->64
    #V = Linear(d_model, d_v)(x)  #512->64
    #Attention = softmax(QK^T/sqrt(d_K)) * V
    #sqrt(d_K)，缩放因子防止点积过大导致梯度消失
    #为什么d_k=d_v?Q和K必须等维才能做点积(QK^T),d_v理论上可以不同，但原始论文中统一设为相同值，简化设计。
    #直观理解：每个头用64维的小空间学习一种特定的词间关系，8个头同时从不同角度理解句子，最后合并结果。

    n_layers = 6  # number of Encoder of Decoder Layer
    #这行定义了Encoder和Decoder各自的层数

    n_heads = 8  # number of heads in Multi-Head Attention
    #这行定义了多头注意力的头数。
    #单头注意力只能从一个角度理解词间关系，多头则让模型同时从8个不同角度观察。

    model = Transformer()
    #这行代码实例化了整个Transformer模型

    criterion = nn.CrossEntropyLoss()
    #创建交叉熵损失函数

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #创建Adam优化器

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    #准备训练数据
    #enc_inputs   [1,5]
    #dec_inputs   [1,5]
    #target_batch [1,5]

    for epoch in range(100):
        optimizer.zero_grad()
        #清空上一步的梯度

        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        #outputs [5,7]
        #enc_self_attns  长度为6的列表，每个元素[1,8,5,5]
        #dec_self_attns  长度为6的列表，每个元素[1,8,5,5]
        #dec_enc_attns   长度为6的列表，每个元素[1,8,5,5]

        loss = criterion(outputs, target_batch.contiguous().view(-1))
        #计算损失
        #target_batch.contiguous().view(-1)  维度为5

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        #打印每轮训练的损失量

        loss.backward()
        #反向传播

        optimizer.step()
        #更新模型所有可训练参数

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    #predict [5,7]

    predict = predict.data.max(1, keepdim=True)[1]
    #(1).data，取底层数据，脱离计算图。
    #(2).max(1, keepdim=True),沿第1维取最大值。
    #(3)[1]取indices
    #.max(...)[0] -> values [5,1]  最大得分值
    #.max(...)[1] -> values [5.1]  最大得分的词ID

    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])
    #打印翻译结果
    #(1)predict.squeeze()，去掉大小为1的维度. [5,1] -> [5]
    #(2)n.item(),张量标量转Python整数
    #(3)number_dict[n.item()],ID转回单词 
    #

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)
    #展示编码器最后一层的第一个头的注意力权重

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)
    #展示解码器最后一层的第一个头的自注意力权重

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
    #展示解码器最后一层的第一个头的交叉注意力权重
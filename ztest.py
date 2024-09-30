# import torch
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# # # mask=torch.full((3,4),2)
# # # for i in range(3):
# # #     for j in range(4):
# # #         mask[i][j]=i+j
# # # new_mask=torch.full((3,4,7),0)
# # # new_mask_index=mask.index_select(1,torch.LongTensor([1]))
# # # print(mask)
# # # print(new_mask_index)
# # # a=mask[:,:]
# # # print(a)
# # # new_mask[1][1][mask]=1
# # # # new_mask[0][new_mask_index]=1
# # # print(new_mask)
# # shape = (3, 3)

# # # 使用torch.arange生成递增值
# # a = torch.arange(0, shape[0]).unsqueeze(1).expand(*shape)
# # print(a.reshape(-1))
# # stop
# # a = torch.zeros((2, 2, 4, 4))
# # index = (
# #     torch.LongTensor([0, 0, 1, 1]),
# #     torch.LongTensor([0, 0, 1, 1]), 
# #     torch.LongTensor([0, 2, 1, 3]),
# #     torch.LongTensor([0, 2, 1, 3]),
# # )
# # new_value = torch.Tensor([1, 2, 3, 4])

# # b=a.index_put(index, new_value)
# # # print(b)

# # bs = torch.tensor([[0,0,0,0],[1,1,1,1]])
# # a=torch.tensor([[1,2,3,3],[1,0,3,3]])
# # b=torch.tensor([[4,5,6,6],[4,0,6,6]])
# # c=torch.tensor([[7,8,9,9],[7,0,9,9]])
# # all_lens=torch.tensor([1,2])
# # batch_size=a.shape[0]

# # # a=a.view(batch_size,-1)
# # # b=b.view(batch_size,-1)
# # # c=c.view(batch_size,-1)
# # mask=torch.full((batch_size,4,7,10),0)
# # # mask=mask.view(:,:,-1)
# # mask[bs,a,b,c]=1
# # print(mask[0].equal(mask[1]))
# # print(mask)
# # stop

# # a=torch.tensor([[1,2,3],[1,2,3]])
# # b=torch.tensor([[4,5,6],[4,5,6]])
# # c=torch.tensor([[7,8,9],[7,8,9]])
# # all_lens=torch.tensor([1,2])
# # batch_size=a.shape[0]
# # a=a.view(batch_size,-1)
# # b=b.view(batch_size,-1)
# # c=c.view(batch_size,-1)
# # mask_=torch.full((batch_size,4,7,10),0)
# # # mask=mask.view(:,:,-1)
# # mask_[:,a,b,c]=1

# # print(mask.equal(mask_))
# # # mask.view(batch_size,4,7,10)
# # # print(mask)
# # # mask=torch.full((2,64,64,80),2)

# a = torch.tensor([[[
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 1]]])
# print(a.argmax(-1))
# a = a.argmax(-1)
# a.to('cuda')
# print(a.device)
# a[torch.where(a == 0)] = 120
# print(a)

# from transformers import BertForMaskedLM, BertTokenizer, BertModel
# import random
 
# # 加载预训练的BERT模型
# model = BertForMaskedLM.from_pretrained('bert-base-cased')
# # for name, param in model.named_parameters():
# #     if not param.requires_grad:
# #         print(f"Parameter {name} is frozen and will not be updated during training.")
# #     else:
# #         print(f"Parameter {name} is trainable and will be updated during training.")

 
# # 加载对应的分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
 
# # 示例句子
# text = "Hello, xiao ming wen to beijing"
 
# # 对句子进行分词
# tokenized_text = tokenizer.tokenize(text)
 
# # 掩码一部分词汇（例如：随机掩码一个词）
# # masked_index = random.randint(0, len(tokenized_text)-1)
# # tokenized_text[masked_index] = '[MASK]'
 
# # 将分词结果转换为模型需要的输入格式
# masked_input = tokenizer.convert_tokens_to_ids(tokenized_text)
# model.eval()
# # 预测掩码词的原始内容
# with torch.no_grad():
#     outputs = model(torch.tensor([masked_input]))
#     print(outputs[0].shape)
#     predictions = outputs[0][0]
#     predicted_index = torch.argmax(predictions[masked_index]).item()
#     predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# # 打印原词和预测的词
# print("Original word:", tokenized_text, tokenized_text[masked_index])
# print("Predicted word:", predicted_token)
# from transformers import BertTokenizer, BertForMaskedLM
 
# # 加载预训练模型和分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model = BertForMaskedLM.from_pretrained('bert-base-cased')
 
# # 输入文本
# text = "Hello, my dog is cute"
 
# # 对文本进行分词，并构造input_ids
# encoded_input = tokenizer(text, return_tensors='pt')
# input_ids = encoded_input['input_ids']
 
# # 通过模型得到嵌入向量
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]  # 获取最后一个隐藏层的输出
 
# # 假设我们想要获取第一个词的嵌入向量（索引为0）
# embedding = last_hidden_states[0, 0, :]
# print(input_ids[0])
# embed = model.bert.embeddings.word_embeddings.weight[input_ids[0][0]]
# print(embed)
# # 输出嵌入向量
# print(embed.shape)

# from transformers import BertModel, BertTokenizer, BertForMaskedLM
# import torch

# # 初始化BERT模型和分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model = BertModel.from_pretrained('bert-base-cased')
# mlm_model = BertForMaskedLM.from_pretrained('bert-base-cased')

# # 输入文本
# text = "This is a test sentence."
# inputs = tokenizer(text, return_tensors='pt')

# # 获取input_ids
# input_ids = inputs['input_ids']

# # 获取特征表达 (BERT模型输出)
# outputs = model(input_ids)
# last_hidden_states = outputs.last_hidden_state

# # 获取MLM模型的嵌入向量
# mlm_outputs = mlm_model.bert(input_ids)
# mlm_last_hidden_states = mlm_outputs.last_hidden_state

# # 将两者加在一起
# combined_embeddings = last_hidden_states + mlm_last_hidden_states

# # 定义一个简单的线性层用于训练测试
# linear = torch.nn.Linear(combined_embeddings.size(-1), 2)

# # 示例标签
# labels = torch.tensor([1, 0]).unsqueeze(0)  # 假设是二分类任务

# # 计算logits
# logits = linear(combined_embeddings)

# # 计算损失
# criterion = torch.nn.CrossEntropyLoss()
# loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

# # 反向传播
# loss.backward()

# # 打印损失和梯度
# print("Loss:", loss.item())
# print("Gradients:", linear.weight.grad)



import torch
from transformers import BertTokenizer, BertForMaskedLM

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

# 句子
sentence = "Span-BERT is an interesting model for NLP tasks."
tokens = tokenizer.tokenize(sentence)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# 定义span掩码函数
def mask_span(input_ids, tokenizer, span_start, span_length):
    masked_input_ids = input_ids[:]
    for i in range(span_start, span_start + span_length):
        if i < len(masked_input_ids):
            masked_input_ids[i] = tokenizer.mask_token_id
    return masked_input_ids

# 示例：掩码 "interesting model"
span_start = 3  # "interesting" 的起始位置
span_length = 2  # "interesting model" 的长度
masked_input_ids = mask_span(input_ids, tokenizer, span_start, span_length)

# 转换为张量并添加批次维度
input_ids_tensor = torch.tensor([masked_input_ids])

# 预测
model.eval()
with torch.no_grad():
    outputs = model(input_ids_tensor)
    predictions = outputs.logits

# 获取预测的token
predicted_ids = torch.argmax(predictions, dim=-1).squeeze().tolist()

# 转换回tokens
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

# 打印结果
print("原始句子: ", sentence)
print("掩码句子: ", tokenizer.convert_ids_to_tokens(masked_input_ids))
print("预测结果: ", predicted_tokens)

# 打印预测的掩码部分
predicted_span = predicted_tokens[span_start:span_start + span_length]
print("预测的span: ", predicted_span)

import torch
from transformers import BertModel, BertTokenizer

class CustomMLM:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-cased')

    def forward(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids']
        
        # Mask a token (for example, masking the first token)
        mask_token_id = self.tokenizer.mask_token_id
        masked_input_ids = input_ids.clone()
        # masked_input_ids[0, 1] = mask_token_id  # Mask the second token for illustration

        # Get hidden states
        with torch.no_grad():
            outputs = self.bert_model(masked_input_ids)
            hidden_states = outputs[0]
            aa = self.mlm_model(masked_input_ids, output_hidden_states=True)

        return hidden_states, aa

# Example usage
mlm_model = CustomMLM()
hidden_states, aa = mlm_model.forward("The quick brown fox jumps over the lazy dog.")
print(hidden_states ==  aa.hidden_states[-1],hidden_states.shape)

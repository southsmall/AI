import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertModel, BertTokenizer, AdamW
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 数据集定义
class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),  # [max_length]
            'attention_mask': encoding['attention_mask'].flatten(),  # [max_length]
            'label': torch.tensor(label, dtype=torch.long)
        }


# 2. 模型定义：融合rBERT和BiLSTM
class RBertBiLSTMFusion(nn.Module):
    def __init__(self, bert_model_name, lstm_hidden_dim, lstm_num_layers, num_labels, dropout_prob=0.3):
        super(RBertBiLSTMFusion, self).__init__()
        # 加载预训练BERT模型（可替换为rBERT）
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_output_dim = self.bert.config.hidden_size  # 通常为768

        # 双向LSTM层，输入为BERT的每个token的向量
        self.bilstm = nn.LSTM(input_size=bert_output_dim,
                              hidden_size=lstm_hidden_dim,
                              num_layers=lstm_num_layers,
                              batch_first=True,
                              bidirectional=True)
        # 融合BERT [CLS] 表示（全局）与BiLSTM输出（可以取最后一个时刻或池化）
        # 这里我们对BiLSTM输出做均值池化（沿序列维度），得到2*lstm_hidden_dim维向量
        fusion_dim = bert_output_dim + 2 * lstm_hidden_dim

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        # BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # bert_last_hidden_state: [batch_size, seq_len, hidden_size]
        bert_seq_output = bert_outputs.last_hidden_state
        # bert_cls_output: [batch_size, hidden_size]，通常取[CLS]标记表示
        bert_cls = bert_outputs.pooler_output

        # 双向LSTM对BERT输出做建模
        lstm_out, (h_n, c_n) = self.bilstm(bert_seq_output)
        # h_n: [num_layers*2, batch_size, lstm_hidden_dim]
        # 对所有层的hidden状态取平均或者只取最后一层
        # 这里将bidirectional的最后一层的输出取出来并做均值池化（也可以用max-pooling）
        lstm_out_pool = torch.mean(lstm_out, dim=1)  # [batch_size, 2*lstm_hidden_dim]

        # 融合BERT的全局[CLS]向量和LSTM池化后的特征
        fusion_features = torch.cat((bert_cls, lstm_out_pool), dim=1)
        fusion_features = self.dropout(fusion_features)
        logits = self.classifier(fusion_features)
        return logits


# 3. 示例：数据加载与训练
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


if __name__ == '__main__':
    # 示例：加载数据（请将'data.csv'替换为你的数据路径，假设其中有'text'和'label'两列）
    df = pd.read_csv('data.csv')  # 数据集需要自行准备
    texts = df['text'].tolist()
    labels = df['label'].tolist()  # 灾难推文：1，非灾难推文：0

    model_name = "bert-base-uncased"  # 可替换为你的rBERT模型名称
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 划分数据集（这里简单示例，不做严格划分）
    dataset = DisasterTweetDataset(texts, labels, tokenizer, max_length=128)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 定义模型、损失函数和优化器
    num_labels = 2
    lstm_hidden_dim = 128
    lstm_num_layers = 1
    model = RBertBiLSTMFusion(model_name, lstm_hidden_dim, lstm_num_layers, num_labels, dropout_prob=0.3)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        train_acc, train_loss = train_epoch(model, data_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")

    # 后续还可以添加验证、保存模型、评估等步骤

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置随机种子以确保可复现性
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据集类
class CTDataset(Dataset):
    def __init__(self, descriptions, labels=None):
        self.descriptions = descriptions
        self.labels = labels
        
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        desc = self.descriptions[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return desc, label
        return desc

def find_lr(model_class, model_params, train_loader, criterion, device, 
            init_lr=1e-7, final_lr=1.0, num_steps=100):
    """
    学习率查找器。

    参数:
    model_class: 模型的类 (例如 CTClassifier 或 TransformerClassifier)
    model_params: 一个字典，包含传递给模型构造函数的参数 (例如 vocab_size, embedding_dim, ...)
    train_loader: 训练数据加载器
    criterion: 损失函数
    device: 'cuda' 或 'cpu'
    init_lr: 初始学习率
    final_lr: 最终学习率
    num_steps: 在多少步内从 init_lr 增加到 final_lr
    """
    print("开始查找学习率...")
    
    model = model_class(**model_params).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=init_lr) 
    lr_multiplier = (final_lr / init_lr) ** (1 / num_steps)
    
    current_lr = init_lr
    losses = []
    lrs = []
    data_iter = iter(train_loader)
    
    actual_steps_taken = 0
    for step in tqdm(range(num_steps), desc="LR Finder Steps"):
        try:
            inputs, attention_masks, labels = next(data_iter)
        except StopIteration:
            print(f"数据加载器在 {actual_steps_taken} 步后耗尽。num_steps可能过大。")
            break
            
        inputs, attention_masks, labels = inputs.to(device), attention_masks.to(device), labels.to(device)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        outputs = model(inputs, attention_masks)
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 10 * (losses[0] if losses else 1.0) : # 增加一个损失爆炸的判断
            print(f"在学习率 {current_lr:.2e} 时损失变为 NaN/Inf 或急剧增大，停止查找。")
            break
            
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        lrs.append(current_lr)
        actual_steps_taken +=1
        
        current_lr *= lr_multiplier
        if current_lr > final_lr:
            break

    if not lrs or not losses or len(lrs) < 2: # 至少需要两点来画图或计算diff
        print("未能收集到足够的学习率和损失数据。")
        return

    # 绘制损失-学习率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, label="Loss") # 给主曲线添加label
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, which="both", ls="-")
    
    # 尝试找到一个建议的学习率点
    try:
        # 平滑损失，使用移动平均，窗口大小为5
        # 为了避免边缘效应和平滑后长度变化导致的问题，我们对平滑后的数据进行操作
        window_size = 5
        if len(losses) >= window_size:
            # 使用 'same' 模式并处理边缘，或者只对 'valid' 部分操作
            # 这里我们对平滑后的数据进行切片，以匹配梯度的计算
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            
            # 对应的学习率也需要调整，取平滑窗口中间点对应的学习率
            # (或者取平滑后每个点对应的原始lrs的中心)
            # 为了简单起见，我们使用平滑后点对应的原始lrs的子集
            # smoothed_losses 的长度是 len(losses) - window_size + 1
            # 梯度计算会再减1，所以梯度数组长度是 len(losses) - window_size
            
            if len(smoothed_losses) > 1: # 至少需要两个点来计算梯度
                # 用于计算梯度的学习率：取与 smoothed_losses[:-1] 或 smoothed_losses[1:] 对齐的 lrs 部分
                # 长度为 len(smoothed_losses) - 1
                lrs_for_grad_calc = np.array(lrs[window_size//2 : window_size//2 + len(smoothed_losses)-1])

                loss_grads = np.diff(smoothed_losses) # 长度 len(smoothed_losses) - 1
                
                # 防止除以零或非常小的数，如果学习率变化不大
                log_lr_diff = np.diff(np.log10(lrs_for_grad_calc + 1e-10)) # 加个小数防止log(0)
                
                # 确保 log_lr_diff 中的所有元素都大于一个小阈值，避免除以0
                log_lr_diff[log_lr_diff < 1e-5] = 1e-5

                # 计算归一化梯度
                normalized_grads = loss_grads / log_lr_diff
                
                # 找到梯度最小（下降最快）的点
                # 我们应该在损失开始显著下降之后，但在它开始平稳或上升之前寻找
                # 忽略前面几个点（可能还在高位震荡）
                search_start_idx = max(0, window_size // 2) # 至少从平滑窗口能覆盖的第一个点开始
                if len(normalized_grads) > search_start_idx:
                    best_grad_idx_local = np.argmin(normalized_grads[search_start_idx:])
                    best_grad_idx_global = best_grad_idx_local + search_start_idx
                    
                    # 对应的学习率应该是 lrs_for_grad_calc 中的点
                    suggested_lr_steepest = lrs_for_grad_calc[best_grad_idx_global]
                    plt.axvline(x=suggested_lr_steepest, color='r', linestyle='--', 
                                label=f'Steepest Descent LR: {suggested_lr_steepest:.2e}')
                    print(f"建议的学习率 (最陡峭下降点，启发式): {suggested_lr_steepest:.2e}")

        # 找到损失最低点
        if losses: # 确保 losses 不为空
            min_loss_idx = np.argmin(losses)
            if 0 < min_loss_idx < len(lrs) -1: # 确保最低点不是在开始或结束
                lr_at_min_loss = lrs[min_loss_idx]
                # 通常选择比最低损失点对应的学习率小3到10倍
                conservative_suggested_lr = lr_at_min_loss / 3 
                plt.axvline(x=lr_at_min_loss, color='purple', linestyle='-.', label=f'Min Loss LR: {lr_at_min_loss:.2e}')
                plt.axvline(x=conservative_suggested_lr, color='g', linestyle=':', 
                            label=f'Conservative LR (MinLossLR/3): {conservative_suggested_lr:.2e}')
                print(f"保守建议的学习率 (最低损失LR/3): {conservative_suggested_lr:.2e}")
            elif losses: # 如果最低点在边缘
                 print(f"最低损失点在学习率范围的边缘 ({lrs[min_loss_idx]:.2e})。可能需要调整 init_lr 或 final_lr。")


    except Exception as e:
        import traceback
        print(f"计算建议学习率时出错: {e}")
        traceback.print_exc()


    plt.legend() # 现在应该能找到带label的artist了
    plt.savefig('lr_finder_plot.png')
    print("学习率查找器图像已保存为 lr_finder_plot.png")
    # plt.show() # 如果在非交互环境，可以注释掉show，只保存

# 修正的数据加载与预处理函数
def load_data(train_path, test_path=None):
    # 加载训练数据，使用'|'作为分隔符
    col_names = ['report_ID', 'description', 'label']
    train_df = pd.read_csv(train_path, sep=r'\|,\|', header=None, names=col_names, engine='python', dtype=str)
    # 确认实际有3列数据（report_ID, description, label）
    train_df.columns = ['report_ID', 'description', 'label']
    
    # 处理description列，将字符串转换为整数列表
    train_descriptions = []
    for desc in train_df['description'].values:
        if pd.isna(desc) or desc.strip() == '':
            tokens = []
        else:
            tokens = list(map(int, desc.strip().split()))
        train_descriptions.append(tokens)
    
    # 处理标签列
    train_labels = np.zeros((len(train_df), 17))
    for i, label_str in enumerate(train_df['label'].values):
        if pd.notna(label_str) and label_str.strip() != '':
            valid_labels = [l for l in str(label_str).strip().split() if l]
            for label in label_str.strip().split():
                train_labels[i, int(label)] = 1
    
    # 统计数据特征
    seq_lengths = [len(desc) for desc in train_descriptions]
    max_seq_len = max(seq_lengths) if seq_lengths else 0
    vocab_size = 0
    if train_descriptions:
        all_tokens = [token for desc in train_descriptions if desc for token in desc]
        if all_tokens:
            vocab_size = max(all_tokens) + 1
        else:
            vocab_size = 1 # 避免词汇表大小为0
    else:
        vocab_size = 1
    print(f"训练集大小: {len(train_df)}")
    if seq_lengths:
        print(f"序列平均长度: {np.mean(seq_lengths):.2f}")
        print(f"序列最大长度: {max_seq_len}")
        print(f"序列最小长度: {min(seq_lengths)}")
    else:
        print("序列统计信息：数据为空或所有序列为空")
    print(f"词汇表大小: {vocab_size}")
    
    # 标签分布
    label_dist = train_labels.sum(axis=0)
    print("各区域异常标签分布:")
    for i, count in enumerate(label_dist):
        print(f"区域 {i}: {count} 例 ({count/len(train_df)*100:.2f}%)")
    
    # 加载测试数据（如果提供）
    test_descriptions = None
    test_report_ids = None
    if test_path:
        test_df = pd.read_csv(test_path, sep=r'\|,\|', header=None, names=col_names, engine='python', dtype=str)
        test_report_ids = test_df['report_ID'].values

        test_descriptions = []
        for desc in test_df['description'].values:
            if pd.isna(desc) or str(desc).strip() == '':
                tokens = []
            else:
                tokens = list(map(int, str(desc).strip().split()))
            test_descriptions.append(tokens)
        print(f"测试集大小: {len(test_df)}")
    
    return train_descriptions, train_labels, test_descriptions, test_report_ids, vocab_size, max_seq_len

# 对批次数据进行填充
def collate_fn(batch):
    if isinstance(batch[0], tuple):  # 训练模式，有标签
        descriptions = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        
        # 计算这个批次中的最大长度
        max_len = max(len(desc) for desc in descriptions) if descriptions and all(desc for desc in descriptions) else 1
        
        # 填充序列
        padded_descs = []
        attention_masks = []
        for desc in descriptions:
            padded_desc = desc + [0] * (max_len - len(desc))
            attention_mask = [1] * len(desc) + [0] * (max_len - len(desc))
            padded_descs.append(padded_desc)
            attention_masks.append(attention_mask)
            
        padded_descs = torch.tensor(padded_descs, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.float32)
        
        return padded_descs, attention_masks, labels
    else:  # 测试模式，无标签
        descriptions = batch
        max_len = max(len(desc) for desc in descriptions) if descriptions and all(desc for desc in descriptions) else 1
        
        padded_descs = []
        attention_masks = []
        for desc in descriptions:
            padded_desc = desc + [0] * (max_len - len(desc))
            attention_mask = [1] * len(desc) + [0] * (max_len - len(desc))
            padded_descs.append(padded_desc)
            attention_masks.append(attention_mask)
            
        padded_descs = torch.tensor(padded_descs, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.float32)
        
        return padded_descs, attention_masks

# 定义LSTM模型
class CTClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, num_classes):
        super(CTClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向LSTM，输出维度乘2
        
    def forward(self, x, attention_mask):
        # x: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 应用注意力掩码：将padding部分的embedding置为0
        embedded = embedded * attention_mask.unsqueeze(2)
        
        # LSTM层
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # 使用序列中的最后一个非padding元素的输出
        seq_lengths = attention_mask.sum(dim=1).long()
        batch_size = x.size(0)
        
        # 获取每个序列中最后一个非padding元素的索引
        idx = (seq_lengths - 1).view(-1, 1).expand(batch_size, lstm_out.size(2))
        idx = idx.unsqueeze(1)
        
        # 收集最后一个非padding元素的输出
        last_hidden = lstm_out.gather(1, idx).squeeze(1)
        
        # Dropout和全连接层
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        # 使用sigmoid激活函数得到每个类别的概率
        return torch.sigmoid(logits)

# 定义Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dim_feedforward, dropout_rate, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, attention_mask):
        # x: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 创建Transformer的注意力掩码（注意这里掩码的定义与之前不同）
        # 在Transformer中，1表示要遮盖的位置，0表示要保留的位置
        transformer_mask = (1.0 - attention_mask).bool()
        
        # 应用Transformer编码器
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=transformer_mask)
        
        # 计算池化后的表示
        # 这里使用注意力掩码来进行平均池化，只考虑非填充的token
        expanded_mask = attention_mask.unsqueeze(-1)
        masked_output = transformer_out * expanded_mask
        sum_embeddings = masked_output.sum(1)
        sum_mask = expanded_mask.sum(1)
        # 防止除以0
        sum_mask = torch.clamp(sum_mask, min=1.0)
        pooled_output = sum_embeddings / sum_mask
        
        # Dropout和全连接层
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)  # [batch_size, num_classes]
        
        # 使用sigmoid激活函数得到每个类别的概率
        return torch.sigmoid(logits)

# 训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, model_path):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in progress_bar:
            inputs, attention_masks, labels = batch
            inputs, attention_masks, labels = inputs.to(device), attention_masks.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs, attention_masks)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in progress_bar:
                inputs, attention_masks, labels = batch
                inputs, attention_masks, labels = inputs.to(device), attention_masks.to(device), labels.to(device)
                
                outputs = model(inputs, attention_masks)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
                all_labels.append(labels.cpu().numpy())
                all_preds.append(outputs.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 计算验证集上的AUC
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        
        # 计算每个类别的AUC
        aucs = []
        for i in range(all_labels.shape[1]):
            # 检查该类别是否有正样本和负样本
            if np.sum(all_labels[:, i]) > 0 and np.sum(all_labels[:, i]) < len(all_labels):
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
        
        # 计算平均AUC
        mean_auc = np.mean(aucs) if aucs else 0
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {mean_auc:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    
    return train_losses, val_losses

# 预测函数
def predict(model, test_loader):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            inputs, attention_masks = batch
            inputs, attention_masks = inputs.to(device), attention_masks.to(device)
            
            outputs = model(inputs, attention_masks)
            all_preds.append(outputs.cpu().numpy())
    
    return np.vstack(all_preds)

# 主函数
def main(train_path, test_path, model_type='transformer', batch_size=32, epochs=10, lr=0.001,
         embedding_dim=128, hidden_dim=128, num_layers=2, dropout_rate=0.3,
         nhead=8, dim_feedforward=1024, output_file='predictions.csv'):
    # 加载数据
    train_descriptions, train_labels, test_descriptions, test_report_ids, vocab_size, max_seq_len = load_data(train_path, test_path)
    
    # 确保词汇表大小至少为1
    vocab_size = max(vocab_size, 1)
    
    # 划分训练集和验证集
    val_size = int(0.2 * len(train_descriptions))
    indices = list(range(len(train_descriptions)))
    np.random.shuffle(indices)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_descs = [train_descriptions[i] for i in train_indices]
    train_labs = train_labels[train_indices]
    
    val_descs = [train_descriptions[i] for i in val_indices]
    val_labs = train_labels[val_indices]
    
    # 创建数据集和数据加载器
    train_dataset = CTDataset(train_descs, train_labs)
    val_dataset = CTDataset(val_descs, val_labs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    num_classes = 17
    
    # 创建模型
    
    if model_type == 'lstm':
        model = CTClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
        model_path = 'ct_lstm_model.pth'
    else:  # transformer
        model = TransformerClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
        model_path = 'ct_transformer_model.pth'
    
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    '''
    common_model_params = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'num_layers': num_layers,
        'dropout_rate': dropout_rate,
        'num_classes': num_classes
    }
    if model_type == 'lstm':
        model_class = CTClassifier
        model_specific_params = {'hidden_dim': hidden_dim}
        model_path = 'ct_lstm_model.pth'
    else:  # transformer
        model_class = TransformerClassifier
        model_specific_params = {'nhead': nhead, 'dim_feedforward': dim_feedforward}
        # 注意：Transformer 的 d_model 通常就是 embedding_dim
        # TransformerEncoderLayer 的 dropout 是参数，TransformerEncoder 本身没有
        model_path = 'ct_transformer_model.pth'
    
    current_model_params = {**common_model_params, **model_specific_params}

    criterion = nn.BCELoss() # BCELoss 用于 sigmoid 输出
    find_lr(
            model_class=model_class,
            model_params=current_model_params,
            train_loader=train_loader, # 使用完整的训练加载器
            criterion=criterion,
            device=device,
            num_steps=100 # 可以根据 len(train_loader) 调整，例如 min(100, len(train_loader) -1)
        )
    print("学习率查找完成。请查看 lr_finder_plot.png 并根据图像选择学习率。")
    print("现在将退出程序。请修改 --lr 参数并重新运行进行训练。")
    return # 查找完LR后直接退出，不进行后续训练
    '''
    # 训练模型
    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=epochs, model_path=model_path)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    
    # 在测试集上进行预测
    if test_descriptions:
        test_dataset = CTDataset(test_descriptions)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        predictions = predict(model, test_loader)
        
        # 保存预测结果
        # 1. 将17维概率向量转换为单个空格分隔的字符串
        probability_strings = []
        for i in range(predictions.shape[0]): # predictions 是一个 (n_samples, 17) 的 NumPy 数组
            # 将当前行的17个浮点数概率转换为字符串，并用空格连接
            # str(p) 会自动处理浮点数的显示，包括科学计数法（如你例子所示）
            current_prob_str = ' '.join([str(p) for p in predictions[i, :]])
            probability_strings.append(current_prob_str)
            
        # 2. 创建一个包含两列的 DataFrame：report_ID 和格式化后的概率字符串
        # test_report_ids 应该是一个列表或一维NumPy数组，包含所有测试样本的ID
        output_df = pd.DataFrame({
            'report_ID': test_report_ids,
            'probabilities': probability_strings
        })
        
        # 3. 保存到 CSV，不带表头，不带索引，使用 '|,|' 作为分隔符
        output_df.to_csv(output_file, header=False, index=False, sep='|')
        
        print(f'Predictions saved to {output_file}')

if __name__ == "__main__":
    # 命令行参数处理
    import argparse
    
    parser = argparse.ArgumentParser(description='CT影像文本多标签分类模型')
    parser.add_argument('--train', type=str, default='combined_train_data.csv', help='训练数据文件路径')
    parser.add_argument('--test', type=str, default='track1_round1_testB.csv', help='测试数据文件路径')
    parser.add_argument('--model', type=str, default='transformer', choices=['lstm', 'transformer'], 
                        help='模型类型: lstm 或 transformer')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.00141, help='学习率')
    parser.add_argument('--embedding_dim', type=int, default=256, help='嵌入层维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')
    parser.add_argument('--output', type=str, default='predictions14.csv', help='预测结果输出文件')
    # Transformer 特有参数 (确保它们存在)
    parser.add_argument('--nhead', type=int, default=4, help='Transformer nhead (Transformer only)')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Transformer dim_feedforward (Transformer only)')
    
    args = parser.parse_args()
    
    # 运行主函数
    main(args.train, args.test, model_type=args.model, batch_size=args.batch_size,
         epochs=args.epochs, lr=args.lr, embedding_dim=args.embedding_dim,
         hidden_dim=args.hidden_dim, num_layers=args.num_layers,
         dropout_rate=args.dropout, output_file=args.output, nhead=args.nhead,
         dim_feedforward=args.dim_feedforward)
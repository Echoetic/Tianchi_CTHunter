import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import logging
from pathlib import Path
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置随机种子以确保可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

class CTDataset(Dataset):
    def __init__(self, descriptions, labels=None, max_length=None):
        self.descriptions = descriptions
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        desc = self.descriptions[idx]
        # 如果设置了最大长度，进行截断
        if self.max_length and len(desc) > self.max_length:
            desc = desc[:self.max_length]
            
        if self.labels is not None:
            label = self.labels[idx]
            return desc, label
        return desc

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedCTClassifier(nn.Module):
    """改进的LSTM分类器，添加了注意力机制"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 dropout_rate, num_classes, use_attention=True):
        super(ImprovedCTClassifier, self).__init__()
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, 
                           dropout=dropout_rate if num_layers > 1 else 0)
        
        lstm_output_dim = hidden_dim * 2  # 双向LSTM
        
        if self.use_attention:
            self.attention = nn.MultiheadAttention(lstm_output_dim, num_heads=8, 
                                                 dropout=dropout_rate, batch_first=True)
            
        self.dropout = nn.Dropout(dropout_rate)
        
        # 添加批归一化
        self.batch_norm = nn.BatchNorm1d(lstm_output_dim)
        
        # 多层分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.normal_(param, mean=0, std=0.1)
            elif 'lstm' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'linear' in name or 'classifier' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        
    def forward(self, x, attention_mask):
        batch_size, seq_len = x.size()
        
        # 嵌入层
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded * attention_mask.unsqueeze(2)
        
        # LSTM层
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        if self.use_attention:
            # 自注意力机制
            # 创建key_padding_mask (True表示要被忽略的位置)
            key_padding_mask = (1.0 - attention_mask).bool()
            
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, 
                                       key_padding_mask=key_padding_mask)
            
            # 残差连接
            lstm_out = lstm_out + attn_out
        
        # 加权平均池化（考虑注意力掩码）
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(lstm_out)
        masked_output = lstm_out * mask_expanded
        sum_embeddings = masked_output.sum(1)  # [batch_size, hidden_dim*2]
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1.0)
        pooled_output = sum_embeddings / sum_mask
        
        # 批归一化
        pooled_output = self.batch_norm(pooled_output)
        
        # 分类头
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return torch.sigmoid(logits)

class ImprovedTransformerClassifier(nn.Module):
    """改进的Transformer分类器"""
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, 
                 dim_feedforward, dropout_rate, num_classes):
        super(ImprovedTransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embedding_dim) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True  # Pre-norm架构，通常更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # 多层分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.normal_(param, mean=0, std=0.1)
            elif 'linear' in name or 'classifier' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        
    def forward(self, x, attention_mask):
        batch_size, seq_len = x.size()
        
        # 嵌入 + 位置编码
        embedded = self.embedding(x)
        if seq_len <= self.pos_encoding.size(0):
            embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer注意力掩码
        transformer_mask = (1.0 - attention_mask).bool()
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=transformer_mask)
        
        # 全局平均池化
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformer_out)
        masked_output = transformer_out * mask_expanded
        sum_embeddings = masked_output.sum(1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1.0)
        pooled_output = sum_embeddings / sum_mask
        
        # 层归一化
        pooled_output = self.layer_norm(pooled_output)
        
        # 分类头
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return torch.sigmoid(logits)

def load_data(train_path, test_path=None):
    """改进的数据加载函数"""
    try:
        # 加载训练数据
        col_names = ['report_ID', 'description', 'label']
        train_df = pd.read_csv(train_path, sep=r'\|,\|', header=None, names=col_names, 
                              engine='python', dtype=str)
        
        # 处理描述列
        train_descriptions = []
        for desc in train_df['description'].values:
            if pd.isna(desc) or desc.strip() == '':
                tokens = []
            else:
                try:
                    tokens = list(map(int, desc.strip().split()))
                except ValueError:
                    tokens = []
                    logger.warning(f"无法解析描述: {desc}")
            train_descriptions.append(tokens)
        
        # 处理标签列
        train_labels = np.zeros((len(train_df), 17))
        for i, label_str in enumerate(train_df['label'].values):
            if pd.notna(label_str) and label_str.strip() != '':
                try:
                    for label in str(label_str).strip().split():
                        if label.isdigit():
                            label_idx = int(label)
                            if 0 <= label_idx < 17:
                                train_labels[i, label_idx] = 1
                except ValueError:
                    logger.warning(f"无法解析标签: {label_str}")
        
        # 统计信息
        seq_lengths = [len(desc) for desc in train_descriptions if desc]
        vocab_size = 1
        if train_descriptions:
            all_tokens = [token for desc in train_descriptions if desc for token in desc]
            if all_tokens:
                vocab_size = max(all_tokens) + 1
        
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        
        logger.info(f"训练集大小: {len(train_df)}")
        if seq_lengths:
            logger.info(f"序列长度统计 - 平均: {np.mean(seq_lengths):.2f}, "
                       f"最大: {max_seq_len}, 最小: {min(seq_lengths)}")
            logger.info(f"序列长度分位数 - 50%: {np.percentile(seq_lengths, 50):.0f}, "
                       f"75%: {np.percentile(seq_lengths, 75):.0f}, "
                       f"90%: {np.percentile(seq_lengths, 90):.0f}")
        logger.info(f"词汇表大小: {vocab_size}")
        
        # 标签分布统计
        label_dist = train_labels.sum(axis=0)
        logger.info("各区域异常标签分布:")
        for i, count in enumerate(label_dist):
            logger.info(f"区域 {i}: {count} 例 ({count/len(train_df)*100:.2f}%)")
        
        # 加载测试数据
        test_descriptions = None
        test_report_ids = None
        if test_path:
            test_df = pd.read_csv(test_path, sep=r'\|,\|', header=None, names=col_names, 
                                 engine='python', dtype=str)
            test_report_ids = test_df['report_ID'].values
            
            test_descriptions = []
            for desc in test_df['description'].values:
                if pd.isna(desc) or str(desc).strip() == '':
                    tokens = []
                else:
                    try:
                        tokens = list(map(int, str(desc).strip().split()))
                    except ValueError:
                        tokens = []
                        logger.warning(f"无法解析测试描述: {desc}")
                test_descriptions.append(tokens)
            
            logger.info(f"测试集大小: {len(test_df)}")
        
        return train_descriptions, train_labels, test_descriptions, test_report_ids, vocab_size, max_seq_len
    
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise

def collate_fn(batch, max_length=None):
    """改进的批处理函数"""
    if isinstance(batch[0], tuple):  # 训练模式
        descriptions = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    else:  # 测试模式
        descriptions = batch
        labels = None
    
    # 计算批次中的最大长度
    if descriptions and any(desc for desc in descriptions):
        max_len = max(len(desc) for desc in descriptions if desc)
    else:
        max_len = 1
    
    # 如果设置了最大长度限制
    if max_length:
        max_len = min(max_len, max_length)
    
    # 填充序列
    padded_descs = []
    attention_masks = []
    
    for desc in descriptions:
        # 截断或填充
        if len(desc) > max_len:
            desc = desc[:max_len]
        
        padded_desc = desc + [0] * (max_len - len(desc))
        attention_mask = [1] * len(desc) + [0] * (max_len - len(desc))
        
        padded_descs.append(padded_desc)
        attention_masks.append(attention_mask)
    
    padded_descs = torch.tensor(padded_descs, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.float32)
    
    if labels is not None:
        return padded_descs, attention_masks, labels
    else:
        return padded_descs, attention_masks

def compute_metrics(labels, predictions, threshold=0.5):
    """计算多种评价指标"""
    metrics = {}
    
    # AUC分数
    aucs = []
    for i in range(labels.shape[1]):
        if np.sum(labels[:, i]) > 0 and np.sum(labels[:, i]) < len(labels):
            auc = roc_auc_score(labels[:, i], predictions[:, i])
            aucs.append(auc)
    
    metrics['mean_auc'] = np.mean(aucs) if aucs else 0
    metrics['aucs'] = aucs
    
    # 基于阈值的指标
    binary_preds = (predictions > threshold).astype(int)
    
    # F1分数
    f1_scores = []
    for i in range(labels.shape[1]):
        if np.sum(labels[:, i]) > 0:  # 只计算有正样本的类别
            f1 = f1_score(labels[:, i], binary_preds[:, i], zero_division=0)
            f1_scores.append(f1)
    
    metrics['mean_f1'] = np.mean(f1_scores) if f1_scores else 0
    metrics['f1_scores'] = f1_scores
    
    # 平均精度
    avg_precisions = []
    for i in range(labels.shape[1]):
        if np.sum(labels[:, i]) > 0:
            ap = average_precision_score(labels[:, i], predictions[:, i])
            avg_precisions.append(ap)
    
    metrics['mean_ap'] = np.mean(avg_precisions) if avg_precisions else 0
    
    return metrics

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, 
                num_epochs, model_path, patience=5):
    """改进的训练函数，添加早停和更好的监控"""
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds_list = []
        train_labels_list = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in progress_bar:
            inputs, attention_masks, labels = batch
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, attention_masks)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds_list.append(outputs.detach().cpu().numpy())
            train_labels_list.append(labels.detach().cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 计算训练集指标
        train_preds = np.vstack(train_preds_list)
        train_labels = np.vstack(train_labels_list)
        train_metrics = compute_metrics(train_labels, train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds_list = []
        val_labels_list = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in progress_bar:
                inputs, attention_masks, labels = batch
                inputs = inputs.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, attention_masks)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds_list.append(outputs.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 计算验证集指标
        val_preds = np.vstack(val_preds_list)
        val_labels = np.vstack(val_labels_list)
        val_metrics = compute_metrics(val_labels, val_preds)
        val_aucs.append(val_metrics['mean_auc'])
        
        # 学习率调度
        if scheduler:
            scheduler.step(val_metrics['mean_auc'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss}, Train AUC: {train_metrics["mean_auc"]}')
        logger.info(f'  Val Loss: {val_loss}, Val AUC: {val_metrics["mean_auc"]}, Val F1: {val_metrics["mean_f1"]}')
        logger.info(f'  Current Learning Rate: {current_lr}')

        # 早停和模型保存
        if val_metrics['mean_auc'] > best_val_auc:
            best_val_auc = val_metrics['mean_auc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_val_auc,
                'val_metrics': val_metrics
            }, model_path)
            logger.info(f'新的最佳模型已保存到 {model_path}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f'验证集AUC连续{patience}轮未改善，提前停止训练')
            break
    
    logger.info(f'训练完成！最佳验证AUC: {best_val_auc} (第{best_epoch+1}轮)')
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Validation AUC')
    
    plt.subplot(1, 3, 3)
    epochs_range = range(len(train_losses))
    train_aucs = [compute_metrics(np.vstack(train_labels_list), np.vstack(train_preds_list))['mean_auc'] 
                  for _ in epochs_range]
    plt.plot(epochs_range, train_aucs, label='Train AUC')
    plt.plot(epochs_range, val_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Train vs Validation AUC')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    logger.info('训练曲线已保存到 training_curves.png')
    
    return best_val_auc

def predict(model, test_loader):
    """预测函数"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='预测中'):
            inputs, attention_masks = batch
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            
            outputs = model(inputs, attention_masks)
            all_preds.append(outputs.cpu().numpy())
    
    return np.vstack(all_preds)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='改进的CT影像文本多标签分类模型')
    parser.add_argument('--train', type=str, default='combined_train_data.csv', help='训练数据文件路径')
    parser.add_argument('--test', type=str, default='track1_round1_testB.csv', help='测试数据文件路径')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'], 
                        help='模型类型')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.00139, help='学习率')
    parser.add_argument('--embedding_dim', type=int, default=256, help='嵌入层维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')
    parser.add_argument('--output', type=str, default='predictions_improved.csv', help='预测结果输出文件')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer注意力头数')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Transformer前馈网络维度')
    parser.add_argument('--max_length', type=int, default=150, help='最大序列长度')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['bce', 'focal'], help='损失函数类型')
    parser.add_argument('--patience', type=int, default=6, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 修改为显式布尔参数
    parser.add_argument('--use_attention', type=bool, default=True, help='LSTM是否使用注意力机制 (True/False)')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='是否使用学习率调度器 (True/False)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载数据
    train_descriptions, train_labels, test_descriptions, test_report_ids, vocab_size, max_seq_len = load_data(
        args.train, args.test)
    
    # 使用设定的最大长度或数据中的最大长度
    actual_max_length = min(args.max_length, max_seq_len) if max_seq_len > 0 else args.max_length
    logger.info(f"使用最大序列长度: {actual_max_length}")
    
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
    ''''''
    # 创建数据集和数据加载器
    train_dataset = CTDataset(train_descs, train_labs, max_length=actual_max_length)
    val_dataset = CTDataset(val_descs, val_labs, max_length=actual_max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=lambda x: collate_fn(x, actual_max_length))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           collate_fn=lambda x: collate_fn(x, actual_max_length))
    ''''''
    # 创建模型
    num_classes = 17
    
    if args.model == 'lstm':
        model = ImprovedCTClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout_rate=args.dropout,
            num_classes=num_classes,
            use_attention=args.use_attention  # 使用布尔参数
        )
        # 根据注意力机制使用情况命名模型文件
        model_path = f'improved_ct_lstm_{"att" if args.use_attention else "noatt"}_model.pth'
    else:
        model = ImprovedTransformerClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout_rate=args.dropout,
            num_classes=num_classes
        )
        model_path = 'improved_ct_transformer_model.pth'
    
    model = model.to(device)
    ''''''
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'模型参数统计 - 总数: {total_params:,}, 可训练: {trainable_params:,}')
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 学习率调度器 - 根据参数决定是否使用
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                        patience=3)
        logger.info(f'使用学习率调度器: ReduceLROnPlateau')
    else:
        scheduler = None
        logger.info(f'使用固定学习率: {args.lr}')
    
    # 损失函数
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.BCELoss()
    
    logger.info(f'使用损失函数: {args.loss_type}')
    logger.info(f'模型类型: {args.model}')
    if args.model == 'lstm':
        logger.info(f'LSTM注意力机制: {"启用" if args.use_attention else "禁用"}')
    
    # 训练模型
    logger.info('开始训练模型...')
    best_val_auc = train_model(model, train_loader, val_loader, optimizer, criterion, 
                               scheduler, args.epochs, model_path, patience=args.patience)
    ''''''
    # 加载最佳模型进行预测
    if test_descriptions is not None:
        logger.info('加载最佳模型进行预测...')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建测试数据集和数据加载器
        test_dataset = CTDataset(test_descriptions, max_length=actual_max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                collate_fn=lambda x: collate_fn(x, actual_max_length))
        
        # 进行预测
        predictions = predict(model, test_loader)
        
        # 保存预测结果
        # 1. 将17维概率向量转换为单个空格分隔的字符串
        probability_strings = []
        for i in range(predictions.shape[0]):  # predictions 是一个 (n_samples, 17) 的 NumPy 数组
            # 将当前行的17个浮点数概率转换为字符串，并用空格连接
            current_prob_str = ' '.join([str(p) for p in predictions[i, :]])
            probability_strings.append(current_prob_str)
            
        # 2. 创建一个包含两列的 DataFrame：report_ID 和格式化后的概率字符串
        output_df = pd.DataFrame({
            'report_ID': test_report_ids,
            'probabilities': probability_strings
        })
        
        # 3. 保存到 CSV，不带表头，不带索引，使用 '|,|' 作为分隔符
        output_df.to_csv(args.output, header=False, index=False, sep='|')
        
        logger.info(f'预测结果已保存到 {args.output}')

if __name__ == "__main__":
    main()
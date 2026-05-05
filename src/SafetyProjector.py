import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import numpy as np
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 模型定义：增加分类头 (Dual-Head Architecture)
# ==========================================
class SafetyProjector(nn.Module):
    def __init__(self, input_dim=1536, device=None): 
        super(SafetyProjector, self).__init__()
        
        # [Head 1] Embedding Projector: 负责语义解耦
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128) # 压缩到 128 维
        )
        
        # [Head 2] Classifier Head: 负责强行划界 (0=Safe, 1=Harmful)
        # 输入是归一化后的 embedding，直接映射到概率 Logits
        self.classifier = nn.Linear(128, 1) 

        if device is not None:
            self.to(device)

    def forward(self, x):
        # 1. 计算 Embedding
        feat = self.net(x)
        # 必须归一化，因为我们需要计算 Cosine Similarity，也为了让分类头处理单位球面上的特征
        emb_norm = F.normalize(feat, p=2, dim=1)
        
        # 2. 计算分类分数 (Logits)
        # 注意：这里返回的是未经过 Sigmoid 的 logits，配合 BCEWithLogitsLoss 使用更稳定
        logits = self.classifier(emb_norm)
        
        return emb_norm, logits

# ==========================================
# 2. 训练循环：混合 Loss (Triplet + BCE)
# ==========================================
def train_on_agent_align(triplets, input_dim=384, batch_size=32, epochs=20, lr=1e-3, margin=1.0, save_path="./models/safety_projector.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # 准备数据
    print(f"📊 Preparing {len(triplets)} triplets...")
    anchors = torch.tensor([t[0] for t in triplets], dtype=torch.float32)
    positives = torch.tensor([t[1] for t in triplets], dtype=torch.float32)
    negatives = torch.tensor([t[2] for t in triplets], dtype=torch.float32)
    
    dataset = TensorDataset(anchors, positives, negatives)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    
    # 初始化模型
    model = SafetyProjector(input_dim=input_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # [关键修改] 定义两个 Loss
    criterion_triplet = nn.TripletMarginLoss(margin=margin, p=2) 
    criterion_cls = nn.BCEWithLogitsLoss() # 自带 Sigmoid，数值更稳
    
    print(f"🚀 Training Safety Projector (Hybrid Loss)...")
    
    best_loss = float('inf')
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_cls_acc = 0
        num_batches = 0
        
        for a, p, n in loader:
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播：获取 embedding 和 logits
            a_emb, a_logits = model(a) # Anchor (Harmful) -> Label 1
            p_emb, p_logits = model(p) # Positive (Harmful) -> Label 1
            n_emb, n_logits = model(n) # Negative (Benign)  -> Label 0
            
            # --- 1. 计算 Triplet Loss (拉开相对距离) ---
            loss_triplet = criterion_triplet(a_emb, p_emb, n_emb)
            
            # --- 2. 计算 Classification Loss (强行划界) ---
            # 拼接 logits
            all_logits = torch.cat([a_logits, p_logits, n_logits], dim=0)
            
            # 构造标签: Anchor(1), Positive(1), Negative(0)
            ones = torch.ones_like(a_logits)
            zeros = torch.zeros_like(n_logits)
            all_labels = torch.cat([ones, ones, zeros], dim=0)
            
            loss_cls = criterion_cls(all_logits, all_labels)
            
            # --- 3. 总 Loss (加大分类权重) ---
            # 权重 2.0 是为了强迫模型优先学会区分黑白，再优化聚类
            loss = loss_triplet + 2.0 * loss_cls
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算简单的准确率用于监控
            preds = (torch.sigmoid(all_logits) > 0.5).float()
            acc = (preds == all_labels).float().mean()
            total_cls_acc += acc.item()
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_cls_acc / num_batches if num_batches > 0 else 0
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim
            }, save_path)
        
        if (epoch+1) % 1 == 0: # 每轮都打印，监控分类准确率
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f} | Class_Acc={avg_acc*100:.1f}%")
            
    print(f"✅ Training Done. Best loss: {best_loss:.4f}")
    return model

# ==========================================
# 3. 数据挖掘逻辑 (基本保持不变，确认逻辑即可)
# ==========================================
def prepare_training_triplets(harmful_texts, benign_texts, embedding_func):
    """
    基于语义挖掘构建三元组：
    1. Hard Negative: 找语义最像的良性样本 (为了划清边界)
    2. Semantic Positive: 找语义最像的恶意样本 (为了聚类 RAG)
    """
    import numpy as np
    from tqdm import tqdm
    import random

    # 1. 预计算所有 Embedding (作为挖掘的地图)
    print("Encoding Harmful samples for mining...")
    h_embs = np.array(embedding_func(harmful_texts))
    
    print("Encoding Benign samples for mining...")
    b_embs = np.array(embedding_func(benign_texts))
    
    # 归一化 (为了算 Cosine Similarity)
    h_norm = h_embs / (np.linalg.norm(h_embs, axis=1, keepdims=True) + 1e-9)
    b_norm = b_embs / (np.linalg.norm(b_embs, axis=1, keepdims=True) + 1e-9)
    
    triplets = []
    
    print("⛏️ Mining Hard Triplets (This is where the magic happens)...")
    
    # 计算矩阵: 
    # sim_h2b: Harmful 到 Benign 的距离 (用于找 Hard Negative)
    # sim_h2h: Harmful 到 Harmful 的距离 (用于找 Semantic Positive)
    sim_h2b = np.dot(h_norm, b_norm.T) 
    sim_h2h = np.dot(h_norm, h_norm.T)
    
    # 将对角线 (自己和自己) 设为 -1，防止 Positive 选到自己
    np.fill_diagonal(sim_h2h, -1.0)

    for i in tqdm(range(len(h_embs))):
        anchor = h_embs[i] # 当前的恶意样本
        
        # --- 策略 A: 挖掘 Hard Negative (关键!) ---
        # 找到最像 Anchor 的那个 Benign 样本
        # 例如：Anchor="Hack server", 我们希望找到 Benign="Server defense"
        hard_neg_idx = np.argmax(sim_h2b[i])
        hard_negative = b_embs[hard_neg_idx]
        
        # --- 策略 B: 挖掘 Semantic Positive (为了 RAG) ---
        # 找到最像 Anchor 的另一个 Harmful 样本 (Top-1 相似，但不是自己)
        # 例如：Anchor="Make Bomb", Positive="Explosive recipe"
        # 如果只想做随机 Positive，这里可以用 random.choice
        semantic_pos_idx = np.argmax(sim_h2h[i])
        semantic_positive = h_embs[semantic_pos_idx]
        
        # --- 组合三元组 ---
        # 这里的 anchor, positive, negative 都是原始 embedding
        # Projector 的任务是把它们在"新空间"里重新排列
        triplets.append((anchor, semantic_positive, hard_negative))
        
        # [可选] Data Augmentation: 也可以加一些 Semi-Hard Negative
        # 防止模型过拟合最难的样本，偶尔加点随机负样本
        if i % 5 == 0: # 每 5 个样本加一个随机噪音
            rand_neg_idx = random.choice(range(len(b_embs)))
            triplets.append((anchor, semantic_positive, b_embs[rand_neg_idx]))

    print(f"✅ Generated {len(triplets)} semantic triplets.")
    return triplets


import json

def parse_agent_align_data(dataset):
    """
    解析 AgentAlign 格式的数据列表
    Args:
        dataset: 包含上述 JSON 对象的列表
    Returns:
        harmful_texts: list[str]
        benign_texts: list[str]
    """
    harmful_texts = []
    benign_texts = []

    for record in dataset:
        # 1. 确定标签
        # 根据 ID 前缀或 Category 字段判断
        is_harmful = False
        if 'harmful' in record.get('id', '') or record.get('category') == 'self_harm':
            is_harmful = True
        elif 'benign' in record.get('category', ''):
            is_harmful = False
        else:
            continue # 跳过无法分类的数据

        # 2. 提取用户意图 (User Intent)
        # 我们通常取第一条 User Message 作为意图 Anchor
        user_content = ""
        if 'messages' in record:
            for msg in record['messages']:
                if msg['role'] == 'user':
                    user_content = msg['content']
                    break # 找到第一个问题即可
        
        if not user_content:
            continue

        # 3. 分类存储
        if is_harmful:
            harmful_texts.append(user_content)
        else:
            benign_texts.append(user_content)

    print(f"✅ Parsed Data: {len(harmful_texts)} Harmful vs {len(benign_texts)} Benign")
    return harmful_texts, benign_texts


def main_train(data_path=None, save_path=None, epochs=20, batch_size=32, lr=1e-3, margin=0.5):
    """Train the Safety Projector from AgentAlign data.

    Args:
        data_path: path to ``agent_align_data_v3.json``. Defaults to
            ``$AGENT_ALIGN_PATH`` or ``../AgentAlign/agent_align_data_v3.json``.
        save_path: where to save the trained checkpoint. Defaults to
            ``$SAFETY_PROJECTOR_PATH`` or ``./models/safety_projector.pth``.
    """
    import os
    from sentence_transformers import SentenceTransformer

    data_path = data_path or os.environ.get(
        "AGENT_ALIGN_PATH", "../AgentAlign/agent_align_data_v3.json"
    )
    save_path = save_path or os.environ.get(
        "SAFETY_PROJECTOR_PATH", "./models/safety_projector.pth"
    )

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("   Set AGENT_ALIGN_PATH or pass --data_path.")
        return

    print(f"📂 Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    harmful_texts, benign_texts = parse_agent_align_data(data)
    
    if len(harmful_texts) == 0 or len(benign_texts) == 0:
        print("❌ No valid data found. Please check the data format.")
        return
    
    print(f"✅ Loaded {len(harmful_texts)} harmful and {len(benign_texts)} benign samples")
    
    # 2. 初始化 embedding 模型（使用与 risk_tree 相同的模型）
    print("🔧 Initializing embedding model...")
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    input_dim = embedding_model.get_sentence_embedding_dimension()  # 通常是 384
    
    def embedding_func(text_list):
        """批量编码文本"""
        return embedding_model.encode(text_list, show_progress_bar=False)
    
    # 3. 准备训练三元组
    print("🔄 Preparing training triplets...")
    triplets = prepare_training_triplets(harmful_texts, benign_texts, embedding_func)
    
    if len(triplets) == 0:
        print("❌ No triplets generated. Please check the data.")
        return
    
    print(f"✅ Generated {len(triplets)} triplets")
    
    # 4. 训练模型
    model = train_on_agent_align(
        triplets=triplets,
        input_dim=input_dim,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        margin=margin,
        save_path=save_path,
    )

    print("🎉 Training completed successfully!")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SafeHarbor's Safety Projector.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to agent_align_data_v3.json. Default: $AGENT_ALIGN_PATH.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output checkpoint path. Default: $SAFETY_PROJECTOR_PATH or "
             "./models/safety_projector.pth.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()

    main_train(
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
    )
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import SimpleITK as sitk
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from collections import defaultdict
import time

from joblib import Memory  # 新增缓存工具

# 手动定义计算特异性的函数
def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) != 0 else 0


# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# 数据预处理优化配置
# ------------------------
IMAGE_SIZE = (84, 84, 84)  # 目标尺寸
CACHE_DIR = "image_cache"  # 预处理缓存目录
memory = Memory(CACHE_DIR, verbose=0)  # 初始化缓存


# 手动定义计算特异性的函数
def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) != 0 else 0


# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------
# 1. 模型定义（包含主模型和本地适应模型）
# ------------------------

class ResidualBlock3D(nn.Module):
    """3D残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class PrimaryClassificationModel(nn.Module):
    """主分类模型（特征提取器+分类器）"""

    def __init__(self, num_classes=2):
        super(PrimaryClassificationModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            ResidualBlock3D(64, 64),
            ResidualBlock3D(64, 128, stride=2),
            ResidualBlock3D(128, 256, stride=2),
            ResidualBlock3D(256, 512, stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return features, self.classifier(features.squeeze())


class LocalAdaptationModel(nn.Module):
    """本地适应模型（特征调整层）"""

    def __init__(self, feature_dim):
        super(LocalAdaptationModel, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features):
        return self.adapter(features)


# ------------------------
# 2. 真实数据集定义（包含数据预处理）
# ------------------------

import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize  # 用于3D矩阵resize
import time


class MedicalDataset(Dataset):
    """医学图像数据集（支持.npy文件加载和3D矩阵预处理）"""

    def __init__(self, json_path, image_size=(84, 84, 84)):
        """
        :param json_path: JSON文件路径，包含图像路径和标签
        :param image_size: 目标尺寸 (depth, height, width)，默认(84, 84, 84)
        """
        self.data = []
        self.image_size = image_size  # 目标尺寸 (D, H, W)

        # 读取JSON文件，解析数据路径和标签
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            for item in json_data:
                rd_path = item['RD_path']
                img_path = item['image_path']
                label = item['label']
                self.data.append((rd_path, img_path, label))

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个样本（双通道图像+标签）"""
        rd_path, img_path, label = self.data[idx]
        rd_image = self.load_image(rd_path)
        img_image = self.load_image(img_path)
        label = torch.tensor(label, dtype=torch.long)  # 转换为PyTorch标签

        # 合并双通道：维度为 (C=2, D, H, W)
        return torch.stack([rd_image, img_image], dim=0), label

    def load_image(self, path):
        """加载.npy文件并调整尺寸为目标大小（3D最近邻插值）"""
        start_time = time.time()
        img_3d = np.load(path)  # 加载3D矩阵（形状应为 (D, H, W)）

        # 检查输入是否为3D矩阵
        if len(img_3d.shape) != 3:
            raise ValueError(f"输入矩阵必须为3D，当前形状：{img_3d.shape}")

        # # 使用最近邻插值调整尺寸（order=0对应最近邻）
        # resized_3d = resize(
        #     img_3d,  # 输入3D矩阵
        #     self.image_size,  # 目标尺寸 (D, H, W)
        #     order=0,  # 最近邻插值
        #     anti_aliasing=False  # 关闭抗锯齿（最近邻无需抗锯齿）
        # )

        end_time = time.time()
        # print(
            # f"Load & resize {path} | 原形状:{img_3d.shape} → 新形状:{resized_3d.shape} | 耗时:{end_time - start_time:.4f}s")

        return torch.from_numpy(img_3d).float()  # 转换为PyTorch张量


# ------------------------
# 3. 客户端定义（实现FCAAM交替训练）
# ------------------------

class FederatedClient:
    def __init__(self, client_id, train_json, test_json, batch_size=32):
        self.client_id = client_id
        self.train_dataset = MedicalDataset(train_json)
        self.test_dataset = MedicalDataset(test_json)
        self.image_size = self.train_dataset.image_size  # 保存目标尺寸
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                       pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                      pin_memory=True)

        self.primary_model = PrimaryClassificationModel().to(device)
        self.adaptation_model = LocalAdaptationModel(feature_dim=512).to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer_primary = optim.Adam(self.primary_model.parameters(), lr=1e-4)
        self.optimizer_adaptation = optim.Adam(self.adaptation_model.parameters(), lr=1e-4)

    def set_global_params(self, params):
        self.primary_model.load_state_dict(params)

    def get_global_params(self):
        return self.primary_model.state_dict()

    def local_train(self, epochs=1):
        self.primary_model.train()
        self.adaptation_model.train()
        for epoch in range(epochs):
            with tqdm(self.train_loader, desc=f"Client{self.client_id} Epoch{epoch + 1}") as tepoch:
                for images, labels in tepoch:
                    images, labels = images.to(device), labels.to(device)
                    # 训练逻辑（保持不变）...

    def evaluate(self, test_json=None, save_path=None):
        """支持指定任意测试集路径，保存患者级预测结果"""
        self.primary_model.eval()
        self.adaptation_model.eval()
        all_preds = []
        all_labels = []
        all_pred_labels = []

        # 创建测试数据加载器：优先使用传入的test_json，否则使用默认测试集
        if test_json:
            test_dataset = MedicalDataset(test_json, image_size=self.image_size)
            test_loader = DataLoader(test_dataset, batch_size=self.train_loader.batch_size,
                                     shuffle=False, num_workers=4, pin_memory=True)
        else:
            test_loader = self.test_loader

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features, _ = self.primary_model(images)
                adjusted_features = self.adaptation_model(features.squeeze())
                logits = self.primary_model.classifier(adjusted_features)

                preds_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred_labels = np.argmax(logits.cpu().numpy(), axis=1)

                all_preds.extend(preds_prob)
                all_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(pred_labels)

        # 生成患者级结果（建议从数据中提取真实患者ID，当前用索引代替）
        result_df = pd.DataFrame({
            'patient_index': range(len(all_labels)),  # 可替换为真实患者ID（需从数据路径解析）
            'true_label': all_labels,
            'pred_prob': all_preds,
            'pred_label': all_pred_labels
        })

        # 保存结果
        if save_path:
            result_df.to_csv(save_path, index=False)
            print(f"Client{self.client_id} {test_json or '默认测试集'} 结果已保存至：{save_path}")

        # 计算指标（处理空数据情况）
        if len(all_labels) == 0:
            return {}, result_df
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_pred_labels)
        sen = recall_score(all_labels, all_pred_labels)
        spe = specificity_score(all_labels, all_pred_labels)
        return {
            'AUC': auc,
            'Accuracy': acc,
            'Sensitivity': sen,
            'Specificity': spe
        }, result_df


# ------------------------
# 4. 服务器定义（联邦平均聚合）
# ------------------------

class FederatedServer:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.global_model = PrimaryClassificationModel().to(device)  # 仅维护主模型

    def initialize_global_params(self):
        return self.global_model.state_dict()

    def aggregate(self, client_params):
        """联邦平均聚合（仅主模型参数）"""
        start_time = time.time()
        avg_params = defaultdict(list)
        for params in client_params:
            for k, v in params.items():
                avg_params[k].append(v.float())
        result = {k: torch.mean(torch.stack(v), dim=0) for k, v in avg_params.items()}
        end_time = time.time()
        print(f"Server aggregation took {end_time - start_time:.4f} seconds")
        return result


# ------------------------
# 5. 训练与评估流程
# ------------------------

def main():
    # 初始化客户端（3个中心，各有独立的训练/默认测试集）
    clients = [
        FederatedClient(client_id=1, train_json='D1_training_dataset_npy.json', test_json='D1_test_dataset_npy.json',
                        batch_size=8),
        FederatedClient(client_id=2, train_json='D2_train_dataset_npy.json', test_json='D2_test_dataset_npy.json',
                        batch_size=8),
        FederatedClient(client_id=3, train_json='D3_train_dataset_npy.json', test_json='D3_test_dataset_npy.json',
                        batch_size=8),
    ]

    server = FederatedServer(num_clients=3)
    global_params = server.initialize_global_params()

    # 联邦训练循环（示例：1轮，可根据需求调整）
    for round_idx in range(1, 10):
        print(f"\n=== 联邦训练第 {round_idx} 轮 ===")
        selected_clients = np.random.choice(clients, size=2, replace=False)  # 随机选2个中心参与

        for client in selected_clients:
            client.set_global_params(global_params)  # 加载全局模型
            client.local_train(epochs=2)  # 本地训练2个epoch

        # 聚合模型参数
        client_params = [client.get_global_params() for client in selected_clients]
        global_params = server.aggregate(client_params)

    # 确保所有客户端使用最新的全局模型（包括未被选中的客户端）
    for client in clients:
        client.set_global_params(global_params)

    # 定义公共测试集路径（所有中心需测试的目标数据集）
    common_test_json = 'D1_test_dataset_npy.json'

    # 存储所有评估结果
    internal_metrics = []  # 各中心内部测试集指标
    common_test_metrics = []  # 公共测试集（D1）指标
    all_results = []  # 用于保存所有患者级结果路径（可选）

    # 对每个中心进行双测试集评估
    for client in clients:
        client_id = client.client_id

        # ----------------------- 1. 中心内部测试集评估 -----------------------
        internal_save_path = f"Center{client_id}_internal_test_results.csv"
        internal_metric, _ = client.evaluate(save_path=internal_save_path)
        internal_metrics.append(internal_metric)

        # ----------------------- 2. 公共测试集（D1）评估 -----------------------
        common_save_path = f"Center{client_id}_D1_test_results.csv"
        common_metric, _ = client.evaluate(test_json=common_test_json, save_path=common_save_path)
        common_test_metrics.append(common_metric)

    # ----------------------- 保存指标汇总 -----------------------
    metrics_df = pd.DataFrame(
        internal_metrics,
        index=[f'Center{i}' for i in range(1, 4)],
        columns=['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    ).rename_axis("中心").reset_index()

    common_metrics_df = pd.DataFrame(
        common_test_metrics,
        index=[f'Center{i}' for i in range(1, 4)],
        columns=['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    ).rename_axis("中心").reset_index()

    with pd.ExcelWriter('evaluation_metrics.xlsx') as writer:
        metrics_df.to_excel(writer, sheet_name='中心内部测试指标', index=False)
        common_metrics_df.to_excel(writer, sheet_name='D1公共测试集指标', index=False)

    print("训练及评估完成，所有患者级结果和指标已保存！")


if __name__ == "__main__":
    main()

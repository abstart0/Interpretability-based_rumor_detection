import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 混淆矩阵数据生成图片
# 格式: [[TN, FP], [FN, TP]]
confusion_matrices = {
    'Logistic Regression': [[203, 23], [48, 127]],
    'BiGRU': [[205, 21], [38, 137]],
    'TextCNN + GloVe': [[206, 20], [33, 142]]
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for idx, (title, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=axes[idx])
    axes[idx].set_title(title)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
    axes[idx].set_xticklabels(['Non-rumor', 'Rumor'])
    axes[idx].set_yticklabels(['Non-rumor', 'Rumor'])
plt.tight_layout()
plt.show()
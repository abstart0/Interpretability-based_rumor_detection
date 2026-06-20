import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['Logistic Regression', 'BiGRU', 'TextCNN + GloVe']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# 模型对应的指标数据
data = {
    'Logistic Regression': [0.8229, 0.8467, 0.7257, 0.7815],
    'BiGRU': [0.8529, 0.8671, 0.7829, 0.8228],
    'TextCNN + GloVe': [0.8678, 0.8765, 0.8114, 0.8427]
}

# 计算角度并绘图
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, (model, values) in enumerate(data.items()):
    values_plot = values + values[:1]
    ax.plot(angles, values_plot, linewidth=2, linestyle='solid', label=model, color=colors[idx])
    ax.fill(angles, values_plot, alpha=0.1, color=colors[idx])

# 设置刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0.6, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'], fontsize=10)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.title('Model Performance Radar Chart', fontsize=15, pad=20)
plt.tight_layout()
plt.show()
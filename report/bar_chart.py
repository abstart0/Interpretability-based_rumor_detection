import matplotlib.pyplot as plt
import numpy as np

models = ['Logistic Regression', 'BiGRU', 'TextCNN + GloVe']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
data = {
    'Logistic Regression': [0.8229, 0.8467, 0.7257, 0.7815],
    'BiGRU': [0.8529, 0.8671, 0.7829, 0.8228],
    'TextCNN + GloVe': [0.8678, 0.8765, 0.8114, 0.8427]
}

x = np.arange(len(metrics))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2c3e50', '#e67e22', '#27ae60'] # 深色系更显专业

for model, color in zip(models, colors):
    offset = width * multiplier
    rects = ax.bar(x + offset, data[model], width, label=model, color=color, edgecolor='white', linewidth=1)
    ax.bar_label(rects, fmt='{:.2%}', padding=3, fontsize=9)  # 直接在柱子上显示百分比
    multiplier += 1

ax.set_ylabel('Performance Score', fontsize=12)
ax.set_xlabel('Metrics', fontsize=12)
ax.set_title('Model Performance Comparison on Validation Set', fontsize=14)
ax.set_xticks(x + width, metrics)
ax.set_ylim(0.6, 1.0)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
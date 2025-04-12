import numpy as np
import matplotlib.pyplot as plt

def plot_task_success(task_names, method_names, success_nums):
    """
    绘制任务方法成功率的分组柱状图，不区分 valid seen 和 valid unseen，每个方法用不同颜色。
    
    Parameters:
        task_names (list of str): 任务名称列表，长度为N。
        method_names (list of str): 方法名称列表，长度为M。
        success_nums (list of list of int): 成功数列表，长度为N，每个元素是长度为M的列表，表示每个任务中每个方法的成功数。
    """
    # 配色
    # colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))  # 为每种方法自动生成颜色
    # 由浅蓝到深蓝
    # colors = plt.cm.Blues(np.linspace(0.2, 0.5, len(method_names)))
    colors = plt.cm.Blues([0.2, 0.4, 0.6])
    
    # 数据维度
    num_tasks = len(task_names)
    num_methods = len(method_names)
    
    # 坐标位置
    x = np.arange(num_tasks)
    width = 0.8 / num_methods  # 每组柱的宽度

    fig, ax = plt.subplots(figsize=(10, 6))

    # 遍历方法，在每个任务上绘制柱状图
    for i, method_name in enumerate(method_names):
        success = [success_nums[t][i] for t in range(num_tasks)]
        
        # 每个方法的柱状位置
        positions = x - (0.8 - width) / 2 + i * width
        
        # 绘制柱状图
        ax.bar(positions, success, width, label=method_name, color=colors[i])

    # 设置任务名称为x轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha='right', fontsize=15)
    
    # 标签和图例
    ax.set_ylabel('Success Count', fontsize=16)
    ax.set_title('Task Success by Method', fontsize=18)
    # ax.legend(loc='upper left', fontsize=14, bbox_to_anchor=(1, 1))
    ax.legend(fontsize=14)
    ax.set_ylim(0, max([max(task) for task in success_nums]) + 1)

    # 显示网格
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.show()
    plt.savefig('task_success.png')

# 示例数据
task_names = ['pick laptop'] #'Average']
method_names = ['ours']
success_nums = [
    [0, 4, 4],  # Examine in Light
]

# 调用绘制函数
plot_task_success(task_names, method_names, success_nums)
import numpy as np
import matplotlib.pyplot as plt

def plot_task_success(task_names, method_names, success_nums):
    # colors = plt.cm.Blues(np.linspace(0.2, 0.5, len(method_names)))
    colors = plt.cm.Blues([0.2, 0.4, 0.6])
    
    num_tasks = len(task_names)
    num_methods = len(method_names)
    
    x = np.arange(num_tasks)
    width = 0.8 / num_methods

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method_name in enumerate(method_names):
        success = [success_nums[t][i] for t in range(num_tasks)]
        
        positions = x - (0.8 - width) / 2 + i * width
        
        ax.bar(positions, success, width, label=method_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha='right', fontsize=15)
    
    ax.set_ylabel('Success Count', fontsize=16)
    ax.set_title('Task Success by Method', fontsize=18)
    # ax.legend(loc='upper left', fontsize=14, bbox_to_anchor=(1, 1))
    ax.legend(fontsize=14)
    ax.set_ylim(0, max([max(task) for task in success_nums]) + 1)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.show()
    plt.savefig('task_success.png')

task_names = ['pick laptop'] #'Average']
method_names = ['ours']
success_nums = [
    [0, 4, 4],  # Examine in Light
]

plot_task_success(task_names, method_names, success_nums)
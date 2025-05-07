import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from glob import glob

#finetune folder
DIR = 'finetuned/67_results_ABCD/'

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory.
    """
    checkpoints = glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoints.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)), reverse=True)
    return checkpoints[0].replace('\\', '/') if checkpoints else None

# Find the latest checkpoint directory
checkpoint_dir = find_latest_checkpoint(DIR)
model_id = DIR.split("/")[1]
print("Latest checkpoint directory:", checkpoint_dir)
if not checkpoint_dir:
    print("No checkpoint found. Please check the directory.")
    exit()
    
DIR = checkpoint_dir + '/'

print("Loading training log from:", DIR+'trainer_state.json')

# Load the training log
with open(DIR+'trainer_state.json', 'r') as f:
    data = json.load(f)

# Extract training and evaluation metrics
train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

for entry in data['log_history']:
    if 'loss' in entry:
        train_steps.append(entry['epoch'])
        train_losses.append(entry['loss'])
    if 'eval_loss' in entry:
        eval_steps.append(entry['epoch'])
        eval_losses.append(entry['eval_loss'])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot training loss
plt.plot(train_steps, train_losses, 
         label='Training Loss', 
         color='darkorange',
         marker='o',
         markersize=3,
         linewidth=2,
         alpha=0.8)

# Plot evaluation loss
plt.plot(eval_steps, eval_losses, 
         label='Evaluation Loss', 
         color='royalblue', 
         marker='o',
         markersize=3,
         linewidth=2,
         alpha=0.6)

# Customize the plot
plt.title('Training and Evaluation Loss Progression - ' + model_id, fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(np.arange(0, max(train_steps)+0.1, step=0.1))
plt.ylim(0, max(train_losses)*1.1)

# Add annotations for key points
plt.annotate(f'Best Eval Loss: {min(eval_losses):.2f}',
             xy=(eval_steps[np.argmin(eval_losses)], min(eval_losses)),
             xytext=(-80, -30),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='royalblue'))

plt.annotate(f'Final Train Loss: {train_losses[-1]:.2f}',
                xy=(train_steps[-1], train_losses[-1]),
                xytext=(-80, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='darkorange'))

plt.annotate(f"Best Train Loss: {min(train_losses):.2f}",
                xy=(train_steps[np.argmin(train_losses)], min(train_losses)),
                xytext=(-100, -20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='darkorange'))

# Add background gradient
plt.gca().set_facecolor('#f5f5f5')

# Save and show
plt.tight_layout()
if not os.path.exists('graphs'):
    os.makedirs('graphs')
plt.savefig('graphs/training_graph_'+model_id+'.png', dpi=300)
plt.show()
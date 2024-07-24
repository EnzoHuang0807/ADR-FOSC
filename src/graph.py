import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 
sns.set_style("darkgrid")

def extract_metric(log_dir, metric_tag):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    if metric_tag not in event_acc.Tags()['scalars']:
        print(f"No '{metric_tag}' tag found in {log_dir}")
        return [], []

    # Retrieve scalar data for the specified metric tag
    events = event_acc.Scalars(metric_tag)
    
    steps = [e.step for e in events]
    values = [e.value for e in events]
    print(steps)
    return steps, values


experiment_dir = "../cifar10_experiment/runs"
image_dir = "../img"

log_dirs = [
    "PGD-AT-AWP-ADR",
    "PGD-AT-AWP-ADR-1e-3"
]

labels = [
    "AWP + ADR",
    "AWP + modified ADR"
]

for log_dir, label in zip(log_dirs, labels):
    steps, values = extract_metric(os.path.join(experiment_dir, log_dir), "Score/adv_acc")
    start = steps.index(0)
    end = steps.index(199, start + 1)
    plt.plot(steps[start : end + 1], values[start : end + 1], label = label)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()

output_path = os.path.join(image_dir, "ADR_AWP.png")
plt.savefig(output_path)


import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Path to your TensorBoard log directory (the folder that contains `events.out.tfevents.*` files)
model_id = "TraverseNN-TenLeafTrain"
log_dir = "lightning_logs/cpu_2024-10-08/" + model_id + "/version_4/"

# Initialize the event accumulator
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()  # Load the events from the log

# Get training and validation loss from TensorBoard logs
train_loss = event_acc.Scalars('train_loss_step')  # Replace with your actual tag name
val_loss = event_acc.Scalars('val_loss')  # Replace with your actual tag name

# Extract step and value for training and validation loss
train_steps = [point.step for point in train_loss]
train_values = [point.value for point in train_loss]

val_steps = [point.step for point in val_loss]
val_values = [point.value for point in val_loss]

# Plotting the data using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_values, label='Training Loss')
plt.plot(val_steps, val_values, label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('plots/loss_' + model_id + '.png')  # Save as an image (PNG, JPEG, etc.)
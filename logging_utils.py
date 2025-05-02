import numpy as np
from transformers import TrainerCallback

def compute_perplexity(loss):
    return np.exp(loss) if loss is not None else float('inf')

# Enhanced logging function for training
def log_metrics(writer, log, step, model, validation_loss=None, validation_accuracy=None):
    loss = log.get("loss")
    lr = log.get("learning_rate")
    epoch = log.get("epoch")

    if loss is not None:
        writer.add_scalar("Loss/train", loss, step)
        writer.add_scalar("Perplexity/train", compute_perplexity(loss), step)
    if validation_loss is not None:
        writer.add_scalar("Loss/validation", validation_loss, step)
        writer.add_scalar("Perplexity/validation", compute_perplexity(validation_loss), step)
    
    if lr is not None:
        writer.add_scalar("Learning_Rate/train", lr, step)
    if epoch is not None:
        writer.add_scalar("Epoch/train", epoch, step)
    
    if validation_accuracy is not None:
        writer.add_scalar("Accuracy/validation", validation_accuracy, step)

    # Log gradient and weight norms
    log_gradient_and_weight_norms(writer, step, model)

    writer.flush()

# Log gradient and weight norms
def log_gradient_and_weight_norms(writer, step, model):
    total_grad_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item()**2
            count += 1
    if count > 0:
        writer.add_scalar("Grad_Norm", total_grad_norm**0.5, step)

    total_weight_norm = sum(p.data.norm(2).item()**2 for p in model.parameters())**0.5
    writer.add_scalar("Weight_Norm", total_weight_norm, step)


# Custom Logging Callback
class LoggingCallback(TrainerCallback):
    def __init__(self, writer, model, trainer):
        self.writer = writer
        self.model = model
        self.trainer = trainer
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step <= 0 or step % 10 != 0 or logs is None:
            return

        log_metrics(self.writer, logs, step, self.model)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            validation_loss = metrics.get("eval_loss")
            log_metrics(self.writer, metrics, state.global_step, self.model, validation_loss)

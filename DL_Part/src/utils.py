import matplotlib.pyplot as plt
import seaborn as sns

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def plot_results(results):
    """Create comprehensive plots of the classification results"""
    plt.style.use('default')
    sns.set_palette("husl")

    model_names = list(results.keys())
    accuracy_values = [results[name]['final_accuracy'] for name in model_names]
    f1_values = [results[name]['final_f1'] for name in model_names]
    precision_values = [results[name]['final_precision'] for name in model_names]
    recall_values = [results[name]['final_recall'] for name in model_names]
    training_times = [results[name]['training_time'] for name in model_names]
    param_counts = [results[name]['trainable_params'] for name in model_names]

    # Accuracy Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model_names)), accuracy_values, color='steelblue')
    plt.title('Final Validation Accuracy by Model')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')

    # Training curves for best models
    best_models = sorted(model_names, key=lambda x: results[x]['final_f1'], reverse=True)[:5]
    plt.figure(figsize=(10, 6))
    for model_name in best_models:
        epochs = range(1, len(results[model_name]['train_losses']) + 1)
        plt.plot(epochs, results[model_name]['train_losses'], label=f'{model_name} (Train)', alpha=0.7)
        plt.plot(epochs, results[model_name]['val_losses'], label=f'{model_name} (Val)', linestyle='--', alpha=0.7)
    plt.title('Training Curves - Top 5 Models (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')

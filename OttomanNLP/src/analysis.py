import matplotlib.pyplot as plt 
def plot_models(model_name,index_numbers,losses,accuracy):
    model_name = model_name.replace("/","-")
    """Plot loss values over training steps."""
    plt.figure(figsize=(10, 6))
    plt.plot(index_numbers, losses, marker='o', color='gray', linestyle='-', linewidth=2, markersize=8)
    plt.title('Loss Values Over Training Steps', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Steps', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Loss', fontsize=14, fontweight='bold', color='black')
    plt.grid(True, linestyle='--', color='darkgray', alpha=0.5)
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')

    # Add custom legend text in the upper right corner
    plt.text(max(index_numbers) * 0.05, max(losses) * 0.1, f"Test Accuracy = {accuracy}", fontsize=12, color='black',fontweight="bold",horizontalalignment='left')
    plt.text(max(index_numbers) * 0.05, max(losses) * 0.15, f"Model = {model_name}", fontsize=12, color='black',fontweight="bold",horizontalalignment='left')

    plt.tight_layout()
    plt.savefig(f'model_plots/{model_name}.png')
    plt.show()
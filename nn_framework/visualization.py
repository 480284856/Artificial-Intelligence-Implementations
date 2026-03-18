# nn_framework/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from .metrics import confusion_matrix, f1_score_from_cm, iou_from_cm, mean_iou

def visualize_architecture_comparison(model1, hist1, name1, model2, hist2, name2, X_test, y_test):
    """
    Requirement O1.1: Standardized comparison within the framework.
    Saves PNG and raw JSON metrics.
    """
    os.makedirs("results_data", exist_ok=True)
    
    # Export Raw Data (JSON)
    for h, n in [(hist1, name1), (hist2, name2)]:
        filepath = f"results_data/{n.replace(' ', '_')}_metrics.json"
        clean_hist = {k: [float(val) for val in v] for k, v in h.items()}
        with open(filepath, 'w') as f:
            json.dump(clean_hist, f, indent=4)

    fig = plt.figure(figsize=(20, 10))
    
    # Panel 1: Loss
    plt.subplot(2, 2, 1)
    plt.plot(hist1['loss'], label=name1, color='royalblue') 
    plt.plot(hist2['loss'], label=name2, color='forestgreen') 
    plt.title("O1.1: Training Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)

    # Panel 2: Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(hist1['val_acc'], label=f"{name1} (Val)", color='royalblue', linestyle='--') 
    plt.plot(hist2['val_acc'], label=f"{name2} (Val)", color='forestgreen', linestyle='--') 
    plt.title("O1.1: Accuracy Progress")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)

    # Calculation & Matrices (Panels 3 & 4)
    results = []
    for model in [model1, model2]:
        preds = np.argmax(model.forward(X_test, training=False), axis=1)
        cm = confusion_matrix(y_test, preds, num_classes=10)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        results.append((cm_norm, f1_score_from_cm(cm), mean_iou(iou_from_cm(cm))))

    for i, (cmap, name) in enumerate([('Blues', name1), ('Greens', name2)]):
        plt.subplot(2, 2, i + 3)
        im = plt.imshow(results[i][0], cmap=cmap, vmin=0, vmax=1)
        plt.title(f"{name}\nF1: {results[i][1]:.4f} | mIoU: {results[i][2]:.4f}")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted Label"); plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig("O1_1_final_architecture_benchmark.png", dpi=300)
    plt.show()


    # In nn_framework/visualization.py (ganz unten hinzufügen)

def compare_predictions_side_by_side(model1, name1, model2, name2, X_test, y_test, image_index):
    """Requirement O1: Compact side-by-side inference showcase."""
    test_image = X_test[image_index].reshape(1, -1)
    target = y_test[image_index]
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    models_to_show = [(model1, name1, axes[0]), (model2, name2, axes[1])]
    
    for model, name, ax in models_to_show:
        logits = model.forward(test_image, training=False)
        prediction = np.argmax(logits)
        ax.imshow(X_test[image_index].reshape(28, 28), cmap='gray')
        ax.set_title(f"[{name}]\nPred: {prediction} (Target: {target})", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
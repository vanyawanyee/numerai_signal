import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.scatter(results['dax']['y_test'], results['dax']['ensemble_pred'], alpha=0.5)
    ax1.plot([results['dax']['y_test'].min(), results['dax']['y_test'].max()], 
             [results['dax']['y_test'].min(), results['dax']['y_test'].max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('DAX Log Returns: Actual vs Predicted (Ensemble)')
    
    ax2.scatter(results['tesla']['y_test'], results['tesla']['ensemble_pred'], alpha=0.5)
    ax2.plot([results['tesla']['y_test'].min(), results['tesla']['y_test'].max()], 
             [results['tesla']['y_test'].min(), results['tesla']['y_test'].max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Tesla Log Returns: Actual vs Predicted (Ensemble)')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X, title):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'][:20], feature_importance['importance'][:20])
    plt.xticks(rotation=90)
    plt.title(f"Top 20 Important Features - {title}")
    plt.tight_layout()
    plt.show()
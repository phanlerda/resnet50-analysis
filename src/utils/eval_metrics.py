# src/utils/eval_metrics.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def calculate_classification_metrics(y_true, y_pred, average='macro'):
    """
    Tính toán các metrics classification cơ bản.
    Args:
        y_true (array-like): Nhãn thật.
        y_pred (array-like): Nhãn dự đoán.
        average (str): 'micro', 'macro', 'weighted', hoặc None.
                       Mặc định là 'macro' để không bị ảnh hưởng bởi class imbalance.
                       Nếu None, scores cho mỗi class được trả về.
    Returns:
        dict: Một dictionary chứa accuracy, precision, recall, f1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return metrics

def get_confusion_matrix_data(y_true, y_pred):
    """
    Trả về dữ liệu confusion matrix.
    Args:
        y_true (array-like): Nhãn thật.
        y_pred (array-like): Nhãn dự đoán.
    Returns:
        numpy.ndarray: Ma trận confusion.
    """
    return confusion_matrix(y_true, y_pred)

# Ví dụ sử dụng (có thể bỏ đi sau khi tích hợp)
if __name__ == '__main__':
    y_true_example = [0, 1, 2, 0, 1, 2, 0, 0, 1]
    y_pred_example = [0, 1, 1, 0, 1, 2, 2, 0, 1]
    
    print("Metrics (Macro Average):")
    metrics_macro = calculate_classification_metrics(y_true_example, y_pred_example, average='macro')
    for k, v in metrics_macro.items():
        print(f"{k}: {v:.4f}")

    print("\nMetrics (Micro Average):")
    metrics_micro = calculate_classification_metrics(y_true_example, y_pred_example, average='micro')
    for k, v in metrics_micro.items():
        print(f"{k}: {v:.4f}")

    print("\nMetrics per class (average=None):")
    # labels=[0, 1, 2] # Cần thiết nếu có class không xuất hiện trong y_pred hoặc y_true nhưng muốn tính
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true_example, y_pred_example, average=None, zero_division=0
    )
    print(f"Precision per class: {precision_pc}")
    print(f"Recall per class: {recall_pc}")
    print(f"F1-score per class: {f1_pc}")
    print(f"Support per class: {support_pc}")
    
    cm_data = get_confusion_matrix_data(y_true_example, y_pred_example)
    print(f"\nConfusion Matrix Data:\n{cm_data}")
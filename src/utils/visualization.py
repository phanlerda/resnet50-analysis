# src/utils/visualization.py
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Vẽ và lưu biểu đồ learning curves cho loss và accuracy.
    Args:
        train_losses (list): Danh sách training loss qua các epoch.
        val_losses (list): Danh sách validation loss qua các epoch.
        train_accs (list): Danh sách training accuracy qua các epoch.
        val_accs (list): Danh sách validation accuracy qua các epoch.
        save_path (str, optional): Đường dẫn để lưu biểu đồ. Nếu None, chỉ hiển thị.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Biểu đồ learning curves đã được lưu tại: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    Vẽ và lưu confusion matrix.
    Args:
        y_true (array-like): Nhãn thật.
        y_pred (array-like): Nhãn dự đoán.
        classes (list): Danh sách tên các class.
        normalize (bool): True để normalize confusion matrix.
        title (str): Tiêu đề biểu đồ.
        cmap (matplotlib.colors.Colormap): Bảng màu.
        save_path (str, optional): Đường dẫn để lưu biểu đồ. Nếu None, chỉ hiển thị.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm) # In ra console nếu cần

    plt.figure(figsize=(10, 10 if len(classes) > 20 else 8)) # Tăng kích thước nếu nhiều class
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if len(classes) < 30: # Chỉ hiển thị tick labels nếu số class nhỏ
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    else:
        print("Số lượng class quá lớn, tick labels sẽ không được hiển thị trên ma trận.")


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix đã được lưu tại: {save_path}")
    else:
        plt.show()
    plt.close()
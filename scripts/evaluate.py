import torch
from torch.utils.data import DataLoader
import os
import sys
import time
import numpy as np
import json
import torch.nn as nn 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.configs import base_config
from src.datasets.vggface2_dataset import VGGFace2Dataset
from src.models.vggface2_resnet50 import VGGFace2ResNet50
from src.utils.data_utils import get_default_transforms
from src.utils.eval_metrics import calculate_classification_metrics, get_confusion_matrix_data
from src.utils.visualization import plot_confusion_matrix
from tqdm import tqdm

def evaluate_model(model, dataloader, criterion, device, class_names, results_dir, model_name_prefix):
    model.eval()
    all_labels = []
    all_preds = []
    
    running_loss = 0.0
    total_inference_time = 0.0
    num_samples = 0

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            num_samples += inputs.size(0)

            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    avg_inference_time_per_sample = (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0 
    avg_inference_time_per_batch = (total_inference_time / len(dataloader)) * 1000 if len(dataloader) > 0 else 0 

    print(f"\n--- Evaluation Results for {model_name_prefix} ---")
    print(f"Average inference time per sample: {avg_inference_time_per_sample:.2f} ms")
    print(f"Average inference time per batch: {avg_inference_time_per_batch:.2f} ms")

    if criterion and num_samples > 0:
        test_loss = running_loss / num_samples
        print(f"Test Loss: {test_loss:.4f}")

    metrics = calculate_classification_metrics(all_labels, all_preds, average='macro') 
    print("Overall Metrics (Macro Average):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    metrics_micro = calculate_classification_metrics(all_labels, all_preds, average='micro')
    print("Overall Metrics (Micro Average - equivalent to accuracy for multi-class):")
    print(f"  Accuracy (Micro F1): {metrics_micro['f1_score']:.4f}")

    metrics_to_save = {
        'macro_average': metrics,
        'micro_average': metrics_micro,
        'avg_inference_time_ms_per_sample': avg_inference_time_per_sample,
        'avg_inference_time_ms_per_batch': avg_inference_time_per_batch,
    }
    if criterion and num_samples > 0:
         metrics_to_save['test_loss'] = test_loss

    metrics_filename = os.path.join(results_dir, f"{model_name_prefix}_evaluation_metrics.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Evaluation metrics saved to: {metrics_filename}")

    cm_save_path = os.path.join(results_dir, f"{model_name_prefix}_confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, classes=class_names, normalize=False,
                          title=f'Confusion Matrix for {model_name_prefix}', save_path=cm_save_path)
    
    cm_norm_save_path = os.path.join(results_dir, f"{model_name_prefix}_confusion_matrix_normalized.png")
    plot_confusion_matrix(all_labels, all_preds, classes=class_names, normalize=True,
                          title=f'Normalized Confusion Matrix for {model_name_prefix}', save_path=cm_norm_save_path)


def main():
    print(f"Sử dụng device: {base_config.DEVICE}")
    os.makedirs(base_config.RESULTS_DIR, exist_ok=True)

    # --- CONFIGURATIONS ---
    MODEL_NAME = "vggface2_resnet50_baseline"
    # Kiểm tra thư mục weights/ của bạn xem tên file chính xác là gì.
    # Nếu train.py lưu model tốt nhất dựa trên val_acc, nó thường là "_best.pth"
    # Nếu train.py không có val_loader và lưu dựa trên train_acc, nó có thể là "_best_train_acc.pth"
    MODEL_WEIGHTS_FILENAME = f"{MODEL_NAME}_best_val_acc.pth"

    # 1. Kiểm tra đường dẫn đến file weights
    model_path = os.path.join(base_config.WEIGHTS_DIR, MODEL_WEIGHTS_FILENAME)
    print(f"Đang tìm file weights model tại: {os.path.abspath(model_path)}") # In đường dẫn tuyệt đối để dễ debug

    if not os.path.exists(model_path):
        print(f"Lỗi: File weights model '{model_path}' không tìm thấy.")
        print("Vui lòng kiểm tra các điểm sau:")
        print(f"  1. `base_config.WEIGHTS_DIR` ({base_config.WEIGHTS_DIR}) có trỏ đúng đến thư mục chứa weights không?")
        print(f"  2. Tên file `MODEL_WEIGHTS_FILENAME` ({MODEL_WEIGHTS_FILENAME}) có chính xác không?")
        print(f"  3. Script `train.py` đã chạy thành công và lưu lại model chưa?")
        return

    # 2. Đường dẫn đến dữ liệu Test
    TEST_DATA_DIR = base_config.ALIGNED_TEST_DATA_DIR
    print(f"Sử dụng dữ liệu test từ: {TEST_DATA_DIR}")

    if TEST_DATA_DIR is None or not os.path.exists(TEST_DATA_DIR) or not any(os.scandir(TEST_DATA_DIR)):
        print(f"Lỗi: ALIGNED_TEST_DATA_DIR ('{TEST_DATA_DIR}') trong base_config.py không hợp lệ, trống hoặc không được định nghĩa.")
        print("Vui lòng chạy create_test_set.py, sau đó preprocess_data.py cho tập test,")
        print("và cập nhật ALIGNED_TEST_DATA_DIR trong base_config.py.")
        return
            
    # 3. Số lượng class

    NUM_CLASSES_LIMIT_EVAL = 10

    USE_ON_THE_FLY_ALIGNMENT_TEST = False # Giả định TEST_DATA_DIR đã được align

    # 4. Data Loading
    _, test_transform = get_default_transforms(image_size=base_config.IMAGE_SIZE)

    mtcnn_for_test_dataset = None

    test_dataset = VGGFace2Dataset(
        root_dir=TEST_DATA_DIR,
        transform=test_transform,
        use_face_alignment=USE_ON_THE_FLY_ALIGNMENT_TEST,
        mtcnn_model=mtcnn_for_test_dataset,
        limit=NUM_CLASSES_LIMIT_EVAL # Áp dụng limit ở đây
    )
    
    if len(test_dataset) == 0:
        print(f"Lỗi: Test dataset tại '{TEST_DATA_DIR}' (với limit={NUM_CLASSES_LIMIT_EVAL}) rỗng.")
        return

    num_model_classes = test_dataset.get_num_classes() # Số class thực tế sau khi load với limit
    print(f"Số lượng class trong test set sẽ được sử dụng để load model: {num_model_classes}")
    
    if NUM_CLASSES_LIMIT_EVAL is not None and num_model_classes != NUM_CLASSES_LIMIT_EVAL:
        print(f"Cảnh báo: NUM_CLASSES_LIMIT_EVAL ({NUM_CLASSES_LIMIT_EVAL}) khác với số class thực tế load được từ test_dataset ({num_model_classes}).")
        print("Điều này có thể gây lỗi khi load model nếu model được train với số class khác.")
        # Có thể bạn muốn dừng ở đây nếu có sự không khớp nghiêm trọng

    test_class_names = sorted(test_dataset.class_names) 

    test_loader = DataLoader(test_dataset, batch_size=base_config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Model Loading
    print(f"Đang load model với {num_model_classes} output classes.")
    model = VGGFace2ResNet50(num_classes=num_model_classes, pretrained=False) 
    model.load_state_dict(torch.load(model_path, map_location=base_config.DEVICE))
    model = model.to(base_config.DEVICE)

    # 6. Evaluation
    criterion = nn.CrossEntropyLoss() 
    
    evaluate_model(model, test_loader, criterion, base_config.DEVICE, 
                   class_names=test_class_names, 
                   results_dir=base_config.RESULTS_DIR,
                   model_name_prefix=MODEL_NAME)

if __name__ == '__main__':
    main()
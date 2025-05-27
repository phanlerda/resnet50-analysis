# scripts/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # Thêm Dataset để dùng cho TempValDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import sys
import json # Để lưu history
import numpy as np # Để chia train/val
from PIL import Image # Import ở global scope, và sẽ import lại trong TempValDataset nếu cần

# Thêm src vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.configs import base_config
from src.datasets.vggface2_dataset import VGGFace2Dataset
from src.models.vggface2_resnet50 import VGGFace2ResNet50
from src.utils.data_utils import get_default_transforms
from src.utils.visualization import plot_learning_curves
# from facenet_pytorch import MTCNN # Uncomment nếu bạn dự định sử dụng on-the-fly alignment với MTCNN

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num, num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1}/{num_epochs} [Training]", unit="batch", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
        current_acc = correct_predictions.double().item()/total_samples if total_samples > 0 else 0
        progress_bar.set_postfix(loss=loss.item(), acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    
    return epoch_loss, epoch_acc.item()

def validate_one_epoch(model, dataloader, criterion, device, epoch_num, num_epochs):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1}/{num_epochs} [Validation]", unit="batch", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            current_acc = correct_predictions.double().item()/total_samples if total_samples > 0 else 0
            progress_bar.set_postfix(loss=loss.item(), acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0

    return epoch_loss, epoch_acc.item()

def main():
    print(f"Sử dụng device: {base_config.DEVICE}")
    os.makedirs(base_config.WEIGHTS_DIR, exist_ok=True)
    os.makedirs(base_config.RESULTS_DIR, exist_ok=True)

    # --- CONFIGURATIONS ---
    MODEL_NAME = "vggface2_resnet50_baseline" 
    NUM_CLASSES_LIMIT = getattr(base_config, 'NUM_CLASSES_LIMIT_TRAIN', 10) # Lấy từ config hoặc mặc định 10
                                                                            # Đặt None để dùng toàn bộ.
    VAL_DATA_DIR = getattr(base_config, 'VAL_DATA_DIR', None) 
    USE_ON_THE_FLY_ALIGNMENT = getattr(base_config, 'USE_ON_THE_FLY_ALIGNMENT_TRAIN', False)
    
    # 1. Data Loading and Transforms
    train_transform, val_transform = get_default_transforms(image_size=base_config.IMAGE_SIZE)
    
    mtcnn_for_dataset = None
    if USE_ON_THE_FLY_ALIGNMENT:
        print("Cảnh báo: Sử dụng on-the-fly alignment. Quá trình load data sẽ chậm.")
        # from facenet_pytorch import MTCNN # Import ở đây để tránh lỗi nếu không dùng
        # mtcnn_for_dataset = MTCNN(
        #     image_size=base_config.MTCNN_IMAGE_SIZE, 
        #     margin=base_config.MTCNN_MARGIN, 
        #     keep_all=False, 
        #     device=base_config.DEVICE,
        #     select_largest=True,
        #     post_process=True # Đảm bảo output là tensor chuẩn hóa
        # )

    train_dataset_path = base_config.ALIGNED_DATA_DIR if not USE_ON_THE_FLY_ALIGNMENT else base_config.DATA_DIR
    if not os.path.exists(train_dataset_path) or (os.path.isdir(train_dataset_path) and not any(os.scandir(train_dataset_path))):
        print(f"Lỗi: Thư mục dữ liệu huấn luyện '{train_dataset_path}' không tồn tại hoặc trống.")
        if not USE_ON_THE_FLY_ALIGNMENT:
            print("Vui lòng chạy scripts/preprocess_data.py trước hoặc cung cấp dữ liệu đã align.")
        return

    train_dataset = VGGFace2Dataset(
        root_dir=train_dataset_path, 
        transform=train_transform,
        use_face_alignment=USE_ON_THE_FLY_ALIGNMENT,
        mtcnn_model=mtcnn_for_dataset,
        limit=NUM_CLASSES_LIMIT 
    )
    
    if len(train_dataset) == 0:
        print(f"Lỗi: Training dataset tại '{train_dataset_path}' rỗng hoặc không load được class nào (kiểm tra limit).")
        return

    num_actual_classes = train_dataset.get_num_classes()
    print(f"Số lượng class huấn luyện thực tế: {num_actual_classes}")

    val_dataset = None
    train_loader_dataset = train_dataset # Dataset sẽ được dùng cho train_loader

    if VAL_DATA_DIR and os.path.exists(VAL_DATA_DIR) and any(os.scandir(VAL_DATA_DIR)):
        print(f"Sử dụng tập validation riêng từ: {VAL_DATA_DIR}")
        # Giả sử VAL_DATA_DIR cũng cần cùng cài đặt alignment như train_dataset
        val_dataset_path_actual = base_config.ALIGNED_VAL_DATA_DIR if not USE_ON_THE_FLY_ALIGNMENT else VAL_DATA_DIR
        
        val_dataset = VGGFace2Dataset(
            root_dir=val_dataset_path_actual, # Sử dụng đường dẫn đã align nếu có
            transform=val_transform,
            use_face_alignment=USE_ON_THE_FLY_ALIGNMENT, 
            mtcnn_model=mtcnn_for_dataset,
            # Quan trọng: limit ở đây phải được xử lý cẩn thận để khớp với train_dataset.class_names
            # Nếu model train trên N class cụ thể, val set cũng phải chứa N class đó.
            # Tạm thời không dùng limit cho val_dataset riêng, mà sẽ lọc sau nếu cần
        )
        if len(val_dataset) == 0:
            print(f"Cảnh báo: Validation dataset tại '{val_dataset_path_actual}' rỗng. Sẽ không có validation.")
            val_dataset = None
        else:
            # Lọc val_dataset để chỉ chứa các class có trong train_dataset (nếu train_dataset bị limit)
            # Điều này quan trọng để đảm bảo num_classes của model khớp với dữ liệu val
            train_class_names_set = set(train_dataset.class_names)
            
            # Lọc image_paths và labels
            filtered_val_image_paths = []
            filtered_val_labels = []
            
            # Tạo mapping từ tên class của val_dataset sang index của train_dataset
            # Điều này cần thiết nếu val_dataset.class_to_idx khác với train_dataset.class_to_idx
            # (ví dụ, nếu val_dataset không bị limit hoặc có thứ tự class khác)
            val_class_to_train_idx = {
                cls_name: train_dataset.class_to_idx[cls_name] 
                for cls_name in val_dataset.class_names 
                if cls_name in train_class_names_set
            }

            for i, val_img_path in enumerate(val_dataset.image_paths):
                original_val_label_idx = val_dataset.labels[i]
                original_val_class_name = val_dataset.idx_to_class[original_val_label_idx]
                
                if original_val_class_name in val_class_to_train_idx:
                    filtered_val_image_paths.append(val_img_path)
                    # Map label của val_dataset về label của train_dataset
                    filtered_val_labels.append(val_class_to_train_idx[original_val_class_name])

            if not filtered_val_image_paths:
                print("Cảnh báo: Không có class chung nào giữa train_dataset (limited) và VAL_DATA_DIR. Bỏ qua validation.")
                val_dataset = None
            else:
                val_dataset.image_paths = filtered_val_image_paths
                val_dataset.labels = filtered_val_labels
                # Cập nhật lại class_names, class_to_idx, idx_to_class cho val_dataset
                # để phản ánh đúng các class được giữ lại và map với train_dataset
                val_dataset.class_names = sorted(list(val_class_to_train_idx.keys()))
                val_dataset.class_to_idx = {cls_name: i for i, cls_name in enumerate(val_dataset.class_names)}
                val_dataset.idx_to_class = {i: cls_name for cls_name, i in val_dataset.class_to_idx.items()}
                # Quan trọng: Sau khi lọc, labels trong filtered_val_labels là index của train_dataset
                # Chúng ta cần map lại về index mới của val_dataset (0 đến N-1)
                # Hoặc, đơn giản là đảm bảo num_classes của model là num_actual_classes từ train_dataset
                # và các labels trong val_dataset cũng nằm trong khoảng [0, num_actual_classes-1] và khớp đúng identity.
                # Cách đơn giản nhất là nếu NUM_CLASSES_LIMIT được dùng, thì VAL_DATA_DIR cũng phải là 1 subset
                # với đúng các class đó, và VGGFace2Dataset sẽ tự handle class_to_idx.
                # Đoạn code lọc ở trên khá phức tạp và dễ lỗi nếu không cẩn thận.
                # Cân nhắc: Nếu VAL_DATA_DIR được cung cấp, nó NÊN chứa các class giống hệt train_dataset (nếu train bị limit).
                print(f"Số lượng mẫu validation sau khi lọc (khớp với train classes): {len(val_dataset.image_paths)}")


    elif len(train_dataset) > int(len(train_dataset) * 0.1) and len(train_dataset) > 20 : # Split from train_dataset
        print("Không có VAL_DATA_DIR hợp lệ, tiến hành chia train_dataset thành train/validation subsets.")
        val_split_ratio = 0.1 
        num_train_total = len(train_dataset)
        
        # Tạo indices và shuffle
        indices = list(range(num_train_total))
        np.random.seed(42) 
        np.random.shuffle(indices)
        
        # Chia indices
        split_point = int(np.floor(val_split_ratio * num_train_total))
        if split_point == 0 and num_train_total > 1 : # Đảm bảo có ít nhất 1 mẫu val nếu có thể
            split_point = 1
        
        val_indices = indices[:split_point]
        train_indices = indices[split_point:]

        if not val_indices: # Không có mẫu nào cho validation
             print("Không đủ mẫu để tạo validation set từ training set. Bỏ qua validation.")
             val_dataset = None
        else:
            train_loader_dataset = torch.utils.data.Subset(train_dataset, train_indices) # Dùng Subset cho training
            
            # Tạo TempValDataset cho validation subset
            val_image_paths_list = [train_dataset.image_paths[i] for i in val_indices]
            val_labels_list = [train_dataset.labels[i] for i in val_indices]

            class TempValDataset(Dataset):
                def __init__(self, image_paths, labels, transform, use_alignment=False, mtcnn=None):
                    self.image_paths = image_paths
                    self.labels = labels
                    self.transform = transform
                    self.use_alignment = use_alignment # Thêm flag này
                    self.mtcnn = mtcnn                 # Thêm mtcnn model

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, idx):
                    from PIL import Image # Import ở đây cho DataLoader workers
                    # from src.utils.data_utils import align_face_mtcnn # Nếu cần on-the-fly alignment

                    img_path = self.image_paths[idx]
                    label = self.labels[idx]
                    
                    try:
                        if self.use_alignment and self.mtcnn:
                            # Cần hàm align_face_mtcnn ở đây, nó nên được import
                            # Hoặc copy logic từ VGGFace2Dataset.__getitem__
                            # image = align_face_mtcnn(img_path, self.mtcnn, output_size=base_config.IMAGE_SIZE) 
                            # if image is None:
                            #     print(f"Warning (TempValDataset): Không detect được mặt trong {img_path}. Dùng ảnh gốc.")
                            #     image = Image.open(img_path).convert('RGB')
                            # Tạm thời, giả định nếu split từ train, thì train_dataset đã xử lý alignment
                            # Hoặc train_dataset dùng ảnh đã align sẵn.
                            # Nếu train_dataset dùng on-the-fly, TempValDataset cũng nên dùng.
                            # Hiện tại, để đơn giản, không align lại trong TempValDataset.
                            image = Image.open(img_path).convert('RGB')
                        else:
                            image = Image.open(img_path).convert('RGB')
                    except IOError:
                        print(f"Lỗi đọc ảnh (TempValDataset): {img_path}. Trả về sample ngẫu nhiên.")
                        random_idx = torch.randint(0, len(self.image_paths), (1,)).item()
                        return self.__getitem__(random_idx) # Có thể nguy hiểm

                    if self.transform:
                        image = self.transform(image)
                    return image, label
            
            val_dataset = TempValDataset(
                val_image_paths_list, 
                val_labels_list, 
                val_transform,
                use_alignment=USE_ON_THE_FLY_ALIGNMENT, # Truyền trạng thái alignment
                mtcnn=mtcnn_for_dataset                # và model mtcnn nếu cần
            )
            print(f"Đã chia {len(train_loader_dataset)} mẫu cho training, {len(val_dataset)} mẫu cho validation.")
    else:
        print("Dataset quá nhỏ để chia hoặc không có VAL_DATA_DIR. Validation sẽ bị bỏ qua.")
        val_dataset = None

    # Tạo DataLoaders
    train_loader = DataLoader(train_loader_dataset, batch_size=base_config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=base_config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print("Không có validation loader hợp lệ.")


    # 2. Model
    # num_actual_classes là số class thực tế từ train_dataset (sau khi limit)
    model = VGGFace2ResNet50(num_classes=num_actual_classes, pretrained=True)
    model = model.to(base_config.DEVICE)

    # 3. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=base_config.LEARNING_RATE, weight_decay=base_config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=base_config.STEP_SIZE, gamma=base_config.GAMMA)

    # 4. Training Loop
    best_val_acc = 0.0
    best_train_acc_for_saving = 0.0 # Dùng nếu không có val
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    for epoch in range(base_config.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, base_config.DEVICE, epoch, base_config.NUM_EPOCHS)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(scheduler.get_last_lr()[0])

        epoch_summary = f"Epoch {epoch+1}/{base_config.NUM_EPOCHS} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"

        if val_loader:
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, base_config.DEVICE, epoch, base_config.NUM_EPOCHS)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            epoch_summary += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = os.path.join(base_config.WEIGHTS_DIR, f"{MODEL_NAME}_best_val_acc.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"\nModel saved to {model_save_path} (Val Acc: {best_val_acc:.4f})")
        else: 
            history['val_loss'].append(None) 
            history['val_acc'].append(None)
            # Lưu model dựa trên train_acc nếu không có val
            if train_acc > best_train_acc_for_saving : 
                best_train_acc_for_saving = train_acc
                model_save_path = os.path.join(base_config.WEIGHTS_DIR, f"{MODEL_NAME}_best_train_acc.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"\nModel saved to {model_save_path} (Train Acc: {best_train_acc_for_saving:.4f}) (No validation)")
        
        epoch_summary += f" | LR: {scheduler.get_last_lr()[0]:.6f}"
        print(epoch_summary) # In tóm tắt epoch sau khi progress bar hoàn tất

        scheduler.step()

    print("\nTraining complete.")
    if val_loader:
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    else:
        print(f"Best Training Accuracy (used for saving model): {best_train_acc_for_saving:.4f}")

    # Lưu history và plot learning curves
    history_save_path = os.path.join(base_config.RESULTS_DIR, f"{MODEL_NAME}_training_history.json")
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_save_path}")

    plot_save_path = os.path.join(base_config.RESULTS_DIR, f"{MODEL_NAME}_learning_curves.png")
    
    valid_val_loss = [x for x in history['val_loss'] if x is not None]
    valid_val_acc = [x for x in history['val_acc'] if x is not None]
    
    if not val_loader or not valid_val_loss : 
        # Nếu không có val_loader, tạo dummy data cho plot để không bị lỗi
        # Hoặc sửa plot_learning_curves để có thể chỉ nhận train data
        print("Không có dữ liệu validation hợp lệ để vẽ, biểu đồ validation sẽ không có ý nghĩa.")
        # Để plot_learning_curves chạy được, cần truyền đủ 4 list
        # Nếu không có val data, có thể truyền list rỗng hoặc list 0
        # Hiện tại, plot_learning_curves sẽ vẽ đường 0 nếu không có val data
        dummy_val_data = [0] * len(history['train_loss'])
        plot_learning_curves(
            history['train_loss'], dummy_val_data,
            history['train_acc'], dummy_val_data,
            save_path=plot_save_path
        )
    else:
        plot_learning_curves(
            history['train_loss'], valid_val_loss,
            history['train_acc'], valid_val_acc,
            save_path=plot_save_path
        )

if __name__ == '__main__':
    # Cấu hình một số tham số có thể lấy từ base_config.py
    # Ví dụ trong base_config.py:
    # NUM_CLASSES_LIMIT_TRAIN = 10 # hoặc None
    # VAL_DATA_DIR = "../data/aligned_vggface2_test_subset" # hoặc None
    # ALIGNED_VAL_DATA_DIR = "../data/aligned_vggface2_test_subset" # nếu VAL_DATA_DIR là tập gốc chưa align
    # USE_ON_THE_FLY_ALIGNMENT_TRAIN = False
    main()
# scripts/split_dataset.py
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset_per_class(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Chia dataset từ input_dir thành các tập train, val, test trong output_dir.
    Việc chia được thực hiện cho từng class một để đảm bảo mỗi class có mặt trong tất cả các tập (nếu đủ ảnh).

    Args:
        input_dir (str): Đường dẫn đến thư mục gốc chứa các class.
        output_dir (str): Đường dẫn đến thư mục sẽ chứa các tập train/val/test.
        train_ratio (float): Tỷ lệ cho tập train.
        val_ratio (float): Tỷ lệ cho tập validation.
        test_ratio (float): Tỷ lệ cho tập test.
        seed (int): Random seed để đảm bảo kết quả chia nhất quán.
    """
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Lỗi: Thư mục đầu vào '{input_dir}' không tồn tại hoặc không phải là thư mục.")
        return

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9: # Kiểm tra tổng tỷ lệ
        print("Lỗi: Tổng các tỷ lệ train, val, test phải bằng 1.0")
        return

    # Xóa thư mục output cũ nếu tồn tại để tránh lỗi
    if output_path.exists():
        print(f"Thư mục output '{output_dir}' đã tồn tại. Xóa thư mục cũ...")
        shutil.rmtree(output_path)
    
    # Tạo các thư mục con train/val/test
    train_path = output_path / "train"
    val_path = output_path / "val"
    test_path = output_path / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    print(f"Bắt đầu chia dataset từ '{input_dir}' vào '{output_dir}'...")
    print(f"Tỷ lệ: Train={train_ratio*100}%, Val={val_ratio*100}%, Test={test_ratio*100}%")

    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"Lỗi: Không tìm thấy thư mục class nào trong '{input_dir}'.")
        return

    total_files_copied = {'train': 0, 'val': 0, 'test': 0}

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        # Tạo thư mục class tương ứng trong train/val/test
        (train_path / class_name).mkdir(exist_ok=True)
        (val_path / class_name).mkdir(exist_ok=True)
        (test_path / class_name).mkdir(exist_ok=True)

        # Lấy danh sách tất cả các file ảnh trong thư mục class hiện tại
        # Lọc các file phổ biến, có thể thêm các đuôi file khác nếu cần
        image_files = [f for f in class_dir.glob('*') if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp')]
        
        if not image_files:
            print(f"Cảnh báo: Không có file ảnh nào trong class '{class_name}'. Bỏ qua class này.")
            continue

        random.shuffle(image_files) # Xáo trộn danh sách file

        num_images = len(image_files)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)
        # num_test còn lại, nhưng tính toán để đảm bảo tổng không vượt quá
        num_test = num_images - num_train - num_val 

        # Đảm bảo ít nhất 1 ảnh cho mỗi tập nếu có thể và tỷ lệ > 0
        if train_ratio > 0 and num_train == 0 and num_images > 0: num_train = 1
        if val_ratio > 0 and num_val == 0 and num_images > num_train : num_val = 1 # Chỉ gán nếu còn ảnh sau khi lấy train
        if test_ratio > 0 and num_test == 0 and num_images > num_train + num_val: num_test = 1 # Chỉ gán nếu còn ảnh sau khi lấy train+val
        
        # Điều chỉnh lại nếu tổng vượt quá (do làm tròn hoặc gán tối thiểu 1)
        # Ưu tiên train, rồi val, rồi test
        if num_train + num_val + num_test > num_images:
            if num_train > num_images: num_train = num_images; num_val = 0; num_test = 0
            elif num_train + num_val > num_images: num_val = num_images - num_train; num_test = 0
            else: num_test = num_images - num_train - num_val


        # Phân chia file
        train_files = image_files[:num_train]
        val_files = image_files[num_train : num_train + num_val]
        test_files = image_files[num_train + num_val : num_train + num_val + num_test] # Lấy đến hết phần còn lại
                                                                                      # đã được điều chỉnh ở trên

        # Copy file vào các thư mục tương ứng
        for file_path in train_files:
            shutil.copy(file_path, train_path / class_name / file_path.name)
            total_files_copied['train'] += 1
        for file_path in val_files:
            shutil.copy(file_path, val_path / class_name / file_path.name)
            total_files_copied['val'] += 1
        for file_path in test_files:
            shutil.copy(file_path, test_path / class_name / file_path.name)
            total_files_copied['test'] += 1
            
        # print(f"  Class '{class_name}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    print("\nHoàn tất việc chia dataset.")
    print("Thống kê số lượng file đã copy:")
    print(f"  Train: {total_files_copied['train']} files")
    print(f"  Val:   {total_files_copied['val']} files")
    print(f"  Test:  {total_files_copied['test']} files")
    print(f"Dữ liệu đã được chia và lưu tại: '{output_dir}'")


if __name__ == "__main__":
    # --- Cấu hình ---
    # Đường dẫn đến thư mục gốc chứa 10 class (ví dụ đã align)
    # Thay đổi đường dẫn này cho phù hợp với cấu trúc của bạn
    INPUT_DATA_FOLDER = "../data/vggface2" 
    
    # Đường dẫn đến thư mục output, nơi sẽ chứa các folder train/val/test
    OUTPUT_SPLIT_FOLDER = "../data/dataset_splits" 

    TRAIN_RATIO = 0.7  # 70% for training
    VAL_RATIO = 0.15   # 15% for validation
    TEST_RATIO = 0.15  # 15% for testing
    RANDOM_SEED = 42   # Để đảm bảo việc chia là nhất quán nếu chạy lại

    # Gọi hàm thực hiện chia
    split_dataset_per_class(
        INPUT_DATA_FOLDER,
        OUTPUT_SPLIT_FOLDER,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    # In ra thông tin để người dùng cập nhật base_config.py
    print("\n--- Gợi ý cập nhật cho src/configs/base_config.py ---")
    print(f"# Đường dẫn đến dữ liệu training đã chia và align (nếu input đã align)")
    print(f"ALIGNED_DATA_DIR = \"{Path(OUTPUT_SPLIT_FOLDER) / 'train'}\"")
    print(f"# Đường dẫn đến dữ liệu validation đã chia và align (nếu input đã align)")
    print(f"ALIGNED_VAL_DATA_DIR = \"{Path(OUTPUT_SPLIT_FOLDER) / 'val'}\"")
    print(f"# Đường dẫn đến dữ liệu test đã chia và align (nếu input đã align)")
    print(f"ALIGNED_TEST_DATA_DIR = \"{Path(OUTPUT_SPLIT_FOLDER) / 'test'}\"")
    print("--------------------------------------------------------")
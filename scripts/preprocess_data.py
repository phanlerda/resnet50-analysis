# scripts/preprocess_data.py
import os
import sys
from tqdm import tqdm
from PIL import Image

# Thêm src vào PYTHONPATH để import các module tùy chỉnh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.configs import base_config
from src.utils.data_utils import align_face_mtcnn
from facenet_pytorch import MTCNN

def preprocess_dataset(input_dir, output_dir, mtcnn_model, target_img_size=224, margin=20):
    """
    Align và crop tất cả ảnh trong input_dir và lưu vào output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    class_names = sorted(os.listdir(input_dir))
    total_images = 0
    processed_images = 0
    failed_images = 0

    for class_name in tqdm(class_names, desc="Processing classes"):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_path):
            continue
        
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path, exist_ok=True)

        image_files = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(image_files, desc=f"Class {class_name}", leave=False):
            total_images += 1
            img_input_path = os.path.join(class_input_path, img_name)
            img_output_path = os.path.join(class_output_path, img_name)

            if os.path.exists(img_output_path): # Bỏ qua nếu đã xử lý
                processed_images +=1
                continue

            aligned_face_pil = align_face_mtcnn(img_input_path, mtcnn_model, output_size=target_img_size, margin=margin)
            
            if aligned_face_pil:
                try:
                    # Resize lần cuối nếu align_face_mtcnn không resize chính xác về target_img_size
                    # (MTCNN có tham số image_size, nhưng để chắc chắn)
                    if aligned_face_pil.size != (target_img_size, target_img_size):
                         aligned_face_pil = aligned_face_pil.resize((target_img_size, target_img_size), Image.LANCZOS) # Hoặc Image.BILINEAR
                    aligned_face_pil.save(img_output_path)
                    processed_images += 1
                except Exception as e:
                    print(f"Lỗi khi lưu ảnh {img_output_path}: {e}")
                    failed_images +=1
            else:
                # print(f"Không tìm thấy khuôn mặt trong {img_input_path}")
                failed_images += 1
    
    print(f"\nHoàn tất tiền xử lý.")
    print(f"Tổng số ảnh đầu vào: {total_images}")
    print(f"Số ảnh đã xử lý và lưu thành công: {processed_images}")
    print(f"Số ảnh không xử lý được (không có mặt hoặc lỗi): {failed_images}")


if __name__ == "__main__":
    print(f"Sử dụng device: {base_config.DEVICE}")
    
    mtcnn = MTCNN(
        image_size=base_config.MTCNN_IMAGE_SIZE, # Kích thước MTCNN dùng để phát hiện
        margin=base_config.MTCNN_MARGIN,         # Margin mặc định cho MTCNN
        keep_all=False,                          # Chỉ giữ lại khuôn mặt lớn nhất
        device=base_config.DEVICE,
        select_largest=True
    )
    
    # Ví dụ: tiền xử lý tập train VGGFace2
    # Đảm bảo base_config.DATA_DIR trỏ đến thư mục gốc của VGGFace2 (chứa các thư mục nXXXXXX)
    # và base_config.ALIGNED_DATA_DIR là nơi bạn muốn lưu ảnh đã align.
    
    input_dataset_dir = base_config.DATA_DIR 
    output_aligned_dir = base_config.ALIGNED_DATA_DIR
    
    # Margin lớn hơn một chút khi align có thể tốt hơn cho model sau này
    # Kích thước output là kích thước model yêu cầu (ví dụ 224x224)
    custom_margin_for_saving = 20 
    custom_output_size_for_saving = base_config.IMAGE_SIZE 

    print(f"Bắt đầu tiền xử lý từ: {input_dataset_dir}")
    print(f"Lưu kết quả vào: {output_aligned_dir}")
    print(f"Kích thước ảnh output: {custom_output_size_for_saving}x{custom_output_size_for_saving}")
    print(f"Margin khi crop: {custom_margin_for_saving}")

    preprocess_dataset(
        input_dir=input_dataset_dir,
        output_dir=output_aligned_dir,
        mtcnn_model=mtcnn,
        target_img_size=custom_output_size_for_saving,
        margin=custom_margin_for_saving
    )
    
    # Tương tự, bạn có thể chạy cho tập test nếu có
    # input_test_dir = base_config.TEST_DATA_DIR 
    # output_aligned_test_dir = "../data/aligned_vggface2_test" (ví dụ)
    # preprocess_dataset(input_test_dir, output_aligned_test_dir, mtcnn, ...)
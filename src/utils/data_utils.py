# src/utils/data_utils.py
import torch
from torchvision import transforms
from PIL import Image
import os
from facenet_pytorch import MTCNN
import cv2 # OpenCV
import numpy as np

# Khởi tạo MTCNN
# device nên được truyền từ config hoặc script chính
# mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=config.DEVICE, select_largest=True)

def get_default_transforms(image_size=224):
    """Trả về transform mặc định cho training và validation."""
    # Thống kê ImageNet thường được dùng cho pre-trained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size + 32), # Resize lớn hơn một chút rồi crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms

def align_face_mtcnn(img_path, mtcnn_model, output_size=160, margin=20):
    """
    Phát hiện, crop và align khuôn mặt từ một ảnh sử dụng MTCNN.
    Args:
        img_path (str): Đường dẫn đến ảnh.
        mtcnn_model (MTCNN): Model MTCNN đã khởi tạo.
        output_size (int): Kích thước ảnh output (vuông).
        margin (int): Margin xung quanh bounding box của khuôn mặt.
    Returns:
        PIL.Image: Ảnh khuôn mặt đã được crop và align, hoặc None nếu không phát hiện được.
    """
    try:
        img = Image.open(img_path).convert('RGB')
    except IOError:
        print(f"Lỗi khi đọc ảnh: {img_path}")
        return None

    # Thay đổi margin của mtcnn_model tạm thời nếu cần
    original_margin = mtcnn_model.margin
    original_image_size = mtcnn_model.image_size
    mtcnn_model.margin = margin
    mtcnn_model.image_size = output_size # MTCNN dùng image_size để resize sau crop

    face_tensor = mtcnn_model(img) # Detect và crop

    # Khôi phục margin và image_size gốc
    mtcnn_model.margin = original_margin
    mtcnn_model.image_size = original_image_size

    if face_tensor is not None:
        # Chuyển tensor về PIL Image
        # face_tensor có shape (C, H, W) và giá trị trong [-1, 1]
        face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_array = (face_array * 0.5 + 0.5) * 255 # Chuyển về [0, 255]
        face_array = face_array.astype(np.uint8)
        return Image.fromarray(face_array)
    return None

def align_face_dlib(img_path, face_detector, shape_predictor, output_size=224, padding=0.25):
    """
    (Lựa chọn thay thế cho MTCNN nếu muốn dùng dlib)
    Phát hiện, crop và align khuôn mặt sử dụng dlib.
    """
    # Cần import dlib
    # Code cho dlib alignment sẽ phức tạp hơn một chút, liên quan đến việc
    # tìm facial landmarks và thực hiện affine transformation.
    # Nếu bạn muốn đi theo hướng này, tôi có thể cung cấp code sau.
    # Hiện tại, chúng ta tập trung vào MTCNN vì nó tích hợp sẵn trong facenet-pytorch.
    pass

# Ví dụ sử dụng (để trong script preprocess_data.py hoặc notebook)
# if __name__ == '__main__':
#     from src.configs import base_config
#     mtcnn = MTCNN(
#         image_size=base_config.MTCNN_IMAGE_SIZE, 
#         margin=base_config.MTCNN_MARGIN, 
#         keep_all=False, 
#         device=base_config.DEVICE, 
#         select_largest=True
#     )
#     sample_img_path = "path_to_a_sample_image.jpg" # Thay bằng đường dẫn thật
#     aligned_face = align_face_mtcnn(sample_img_path, mtcnn, output_size=224, margin=20)
#     if aligned_face:
#         aligned_face.save("aligned_sample.jpg")
#         print("Đã lưu ảnh aligned.")
#     else:
#         print("Không tìm thấy khuôn mặt.")
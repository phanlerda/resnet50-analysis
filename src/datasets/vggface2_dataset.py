# src/datasets/vggface2_dataset.py
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN # Để dùng cho on-the-fly alignment nếu cần

class VGGFace2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None, use_face_alignment=False, mtcnn_model=None):
        """
        Args:
            root_dir (string): Thư mục chứa tất cả các ảnh (đã có cấu trúc class/image.jpg).
                               Hoặc thư mục chứa ảnh đã align.
            transform (callable, optional): Transform sẽ được áp dụng trên một sample.
            limit (int, optional): Giới hạn số lượng class để load (dùng cho debug).
            use_face_alignment (bool): Nếu True, sẽ thực hiện face alignment on-the-fly.
                                       Cần cung cấp mtcnn_model.
            mtcnn_model (MTCNN object): Model MTCNN đã khởi tạo, cần nếu use_face_alignment=True.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_face_alignment = use_face_alignment
        self.mtcnn_model = mtcnn_model

        if use_face_alignment and mtcnn_model is None:
            raise ValueError("mtcnn_model phải được cung cấp nếu use_face_alignment là True.")

        self.class_names = sorted(os.listdir(root_dir))
        if limit:
            self.class_names = self.class_names[:limit]

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.class_names)}
        
        self.image_paths = []
        self.labels = []

        for cls_name in self.class_names:
            class_path = os.path.join(root_dir, cls_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[cls_name])
        
        print(f"Đã load {len(self.image_paths)} ảnh từ {len(self.class_names)} class.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            if self.use_face_alignment:
                # Thực hiện alignment on-the-fly
                # Điều này có thể chậm, nên cân nhắc pre-process trước
                # Giả sử hàm align_face_mtcnn từ data_utils đã được điều chỉnh
                # để nhận đường dẫn ảnh và trả về PIL Image
                from src.utils.data_utils import align_face_mtcnn # Cẩn thận circular import
                
                # Bạn cần đảm bảo mtcnn_model được truyền vào đây đúng cách
                # output_size nên phù hợp với transform sau đó (ví dụ 224)
                image = align_face_mtcnn(img_path, self.mtcnn_model, output_size=224, margin=20) 
                if image is None: # Nếu không detect được face, trả về None hoặc ảnh gốc
                    print(f"Warning: Không detect được mặt trong {img_path}. Dùng ảnh gốc.")
                    image = Image.open(img_path).convert('RGB')
            else:
                image = Image.open(img_path).convert('RGB')
        except IOError:
            print(f"Lỗi khi đọc ảnh: {img_path}. Trả về sample ngẫu nhiên.")
            # Xử lý lỗi: có thể trả về một ảnh placeholder hoặc sample khác
            random_idx = torch.randint(0, len(self.image_paths), (1,)).item()
            return self.__getitem__(random_idx)


        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_classes(self):
        return len(self.class_names)

# Ví dụ sử dụng (đặt trong notebook hoặc script train)
# if __name__ == '__main__':
#     from src.configs import base_config
#     from src.utils.data_utils import get_default_transforms

#     train_transform, _ = get_default_transforms(base_config.IMAGE_SIZE)
    
#     # Cách 1: Sử dụng dữ liệu đã align trước
#     # dataset = VGGFace2Dataset(root_dir=base_config.ALIGNED_DATA_DIR, transform=train_transform)

#     # Cách 2: Sử dụng alignment on-the-fly (chậm hơn)
#     # mtcnn = MTCNN(
#     #     image_size=base_config.MTCNN_IMAGE_SIZE, 
#     #     margin=base_config.MTCNN_MARGIN, 
#     #     keep_all=False, 
#     #     device=base_config.DEVICE,
#     #     select_largest=True
#     # )
#     # dataset = VGGFace2Dataset(
#     #     root_dir=base_config.DATA_DIR, # Dùng thư mục ảnh gốc
#     #     transform=train_transform,
#     #     use_face_alignment=True,
#     #     mtcnn_model=mtcnn,
#     #     limit=10 # Tải 10 class để test
#     # )

#     # print(f"Số class: {dataset.get_num_classes()}")
#     # print(f"Kích thước dataset: {len(dataset)}")
#     # img, label = dataset[0]
#     # print(f"Kích thước ảnh: {img.shape}, Label: {label}")
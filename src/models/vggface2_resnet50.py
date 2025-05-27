# src/models/vggface2_resnet50.py
import torch
import torch.nn as nn
from torchvision import models

class VGGFace2ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_dim=None):
        """
        Args:
            num_classes (int): Số lượng class output cho classification.
            pretrained (bool): Sử dụng weights ImageNet pre-trained hay không.
            embedding_dim (int, optional): Nếu được cung cấp, lớp fc cuối cùng
                                           sẽ được thay thế bằng một lớp tạo embedding.
                                           Nếu None, mô hình dùng cho classification.
        """
        super(VGGFace2ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # Lấy số features của lớp avant-dernier (trước lớp fc)
        num_ftrs = self.resnet50.fc.in_features
        
        if embedding_dim:
            # Thay thế lớp fc để tạo embeddings
            # Thường thì chúng ta sẽ lấy features từ lớp trước fc
            # và có thể thêm một lớp fc mới để chiếu xuống embedding_dim
            self.resnet50.fc = nn.Linear(num_ftrs, embedding_dim)
            self.is_embedding_model = True
            print(f"Model VGGFace2 (ResNet-50) được cấu hình cho embedding, output_dim: {embedding_dim}")
        else:
            # Thay thế lớp fc cho classification
            self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
            self.is_embedding_model = False
            print(f"Model VGGFace2 (ResNet-50) được cấu hình cho classification, num_classes: {num_classes}")

    def forward(self, x):
        return self.resnet50(x)

    def get_embedding(self, x):
        """
        Lấy embedding features. Hàm này hữu ích nếu model được huấn luyện
        cho classification nhưng bạn muốn trích xuất features từ lớp avant-dernier.
        Hoặc nếu model đã được cấu hình cho embedding (embedding_dim is not None).
        """
        if self.is_embedding_model:
            return self.forward(x) # Nếu đã là embedding model, chỉ cần forward
        else:
            # Lấy output của lớp avant-dernier
            # Đây là cách phổ biến để trích xuất features từ model classification
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)

            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)

            x = self.resnet50.avgpool(x)
            x = torch.flatten(x, 1)
            # x bây giờ là features trước lớp fc cuối cùng
            return x


# Ví dụ sử dụng (đặt trong notebook hoặc script train)
# if __name__ == '__main__':
#     from src.configs import base_config
    
#     # Model cho classification
#     model_cls = VGGFace2ResNet50(num_classes=base_config.NUM_CLASSES_VGGFACE2, pretrained=True)
#     dummy_input_cls = torch.randn(2, 3, base_config.IMAGE_SIZE, base_config.IMAGE_SIZE)
#     output_cls = model_cls(dummy_input_cls)
#     print(f"Output shape (classification): {output_cls.shape}") # Mong đợi: [2, num_classes]

#     # Lấy embedding từ model classification
#     embedding_from_cls = model_cls.get_embedding(dummy_input_cls)
#     print(f"Embedding shape (from classification model): {embedding_from_cls.shape}") # Mong đợi: [2, 2048] (ResNet-50 default)

#     # Model cho embedding trực tiếp
#     model_emb = VGGFace2ResNet50(num_classes=None, pretrained=True, embedding_dim=base_config.EMBEDDING_DIM)
#     dummy_input_emb = torch.randn(2, 3, base_config.IMAGE_SIZE, base_config.IMAGE_SIZE)
#     output_emb = model_emb(dummy_input_emb) # Hoặc model_emb.get_embedding(dummy_input_emb)
#     print(f"Output shape (embedding model): {output_emb.shape}") # Mong đợi: [2, embedding_dim]

# ResNet50-VGGFace2-FaceRecognition

Dự án phân tích và tối ưu hóa mô hình ResNet-50 cho nhận diện khuôn mặt trên một tập con 10 danh tính từ VGGFace2.

## 🌟 Tổng quan

*   **Mục tiêu:** Xây dựng, huấn luyện và đánh giá chi tiết ResNet-50 cho nhận diện khuôn mặt.
*   **Dữ liệu:** Subset 10 danh tính từ VGGFace2.
*   **Công nghệ:** Python, PyTorch, Torchvision, Scikit-learn.

## 🚀 Cài đặt & Chạy

1.  **Clone repo:**
    ```bash
    git clone https://github.com/phanlerda/resnet50-analysis.git
    cd resnet50-analysis
    ```
2.  **Tạo môi trường ảo và cài đặt thư viện:** (Khuyến nghị)
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Chuẩn bị dữ liệu:** Đảm bảo dữ liệu nằm trong thư mục `data/`.
4.  **Huấn luyện:**
    ```bash
    python src/train.py
    ```
    Mô hình tốt nhất được lưu tại `results/best_model.pth`.
5.  **Đánh giá:**
    ```bash
    python src/evaluate.py --model_path results/best_model.pth
    ```
    Kết quả được lưu trong thư mục `results/`.
6.  **Xem chi tiết:** Mở notebook `face_recognition_report_detailed.ipynb`.

## 📊 Kết quả Nổi bật

*   **Accuracy (Test):** ~93.4%
*   **F1-Score (Macro):** ~92.8%
*   **Inference Time:** ~0.91 ms/mẫu

Xem chi tiết trong `results/metrics.json` và báo cáo (nếu có).

## 퓨 Phát triển Tương lai
*   Mở rộng với nhiều danh tính hơn.
*   Thử nghiệm kiến trúc/hàm mất mát mới.
*   Tối ưu hóa siêu tham số.

---
*Báo cáo chi tiết của dự án này được soạn thảo riêng.*

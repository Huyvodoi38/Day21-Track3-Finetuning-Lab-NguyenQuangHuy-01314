# Custom Dataset Specification — E-Commerce Customer Support Triage (Vietnamese)

> Tài liệu mô tả tập dữ liệu miền riêng theo tiêu chuẩn **Bonus B2 (Deck §13)** của Lab 21.

---

## 1. Tổng quan tập dữ liệu

* **Miền nghiệp vụ**: Chăm sóc khách hàng & Triage ticket sàn Thương mại điện tử (E-commerce / Logistics) tại Việt Nam.
* **Quy mô**: 250 mẫu dữ liệu chất lượng cao (225 train / 25 val theo seed 42).
* **Định dạng đầu vào**: Câu hội thoại / ticket phản ánh ngắn gọn từ người mua hàng bằng tiếng Việt tự nhiên.
* **Định dạng đầu ra**: JSON cấu trúc 4 trường chuẩn:
  ```json
  {
    "intent": "doi_tra | hoan_tien | san_pham_loi | van_chuyen | hoi_thong_tin",
    "urgency": "thap | trung_binh | cao",
    "product": "<tên sản phẩm>",
    "sentiment": "tich_cuc | trung_tinh | tieu_cuc"
  }
  ```

---

## 2. Nguồn dữ liệu & Phương pháp thu thập

1. **Nguồn dữ liệu gốc**: 
   - Tổng hợp và ẩn danh hóa từ các luồng chat CSKH thực tế của các shop kinh doanh đồ công nghệ, gia dụng và phụ kiện trên Shopee/Lazada/TikTok Shop.
   - Loại bỏ toàn bộ thông tin định danh cá nhân (PII) như tên thật, số điện thoại, địa chỉ cụ thể; mã đơn hàng được chuẩn hóa theo định dạng giả lập (`VNxxxxxx`, `DHxxxxxx`, `ODxxxxxx`).

2. **Tính mới về phân phối (Out-of-Distribution — OOD so với Base Model)**:
   - Các mô hình nền tảng đa ngôn ngữ (như Qwen 3.5) thường được pre-train trên văn bản báo chí, bách khoa toàn thư hoặc web crawl chuẩn ngữ pháp.
   - Tập dữ liệu này chứa các đặc trưng ngôn ngữ hội thoại TMĐT thực tế tại Việt Nam:
     - Từ lóng và viết tắt: *ship cod, check inbox, bom hàng, vỡ màn, xả hàng, lỗi nguồn, đổi size, hoàn xu*.
     - Cấu trúc câu ngắn, thiếu chủ ngữ/vị ngữ, mang cảm xúc dồn dập khi gặp sự cố giao nhận.

---

## 3. Quy trình khử nhiễm (Decontamination Pipeline)

Để đảm bảo phép so sánh và đánh giá tại NB5 hoàn toàn khách quan, quy trình khử nhiễm nghiêm ngặt được áp dụng:

1. **N-gram Overlap Check**:
   - Sử dụng thuật toán so khớp 13-gram giữa tập huấn luyện (`train_seed.jsonl`) và tập kiểm thử (`eval_target.jsonl`).
   - Loại bỏ bất kỳ mẫu nào có độ tương đồng Jaccard similarity $> 0.7$ đối với phần input của câu hỏi.

2. **Chia tách độc lập (Stratified Split)**:
   - Tập train (225 mẫu) và tập val (25 mẫu) được phân tách bằng seed 42 cố định.
   - Tập đánh giá mốc (`eval_target.jsonl` 50 mẫu và `eval_regression.jsonl` 15 mẫu) được đóng băng hoàn toàn trước khi bất kỳ tiến trình huấn luyện nào diễn ra.

---

## 4. Thống kê phân phối nhãn

* **Intent**:
  - `doi_tra`: 22%
  - `hoan_tien`: 20%
  - `san_pham_loi`: 24%
  - `van_chuyen`: 18%
  - `hoi_thong_tin`: 16%
* **Urgency**: `thap` (30%), `trung_binh` (45%), `cao` (25%).
* **Sentiment**: `tieu_cuc` (40%), `trung_tinh` (45%), `tich_cuc` (15%).
* **Độ dài Token (p95)**: 98 tokens, phù hợp với `max_length = 256` của tier T4.

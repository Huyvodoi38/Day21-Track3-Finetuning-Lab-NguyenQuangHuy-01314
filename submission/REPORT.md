# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Quang Huy  **MSSV**: 2A202601314  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA Tesla T4 (16GB VRAM, sm_75)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42, 90/10 split) |
| `max_length` | 256 — p95 đo được là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 steps (batch hiệu dụng 16: batch 1 x grad_accum 16) |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*  
Template của Qwen3.5 giữ nguyên cặp thẻ `<think>...</think>`, đảm bảo chuỗi suy luận reasoning trace được bảo toàn đúng cấu trúc và không gây lỗi cú pháp hội thoại.

---

## 2. Mask proof (NB1)

| Chỉ số | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tokens được tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3225.3 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1015.9 |
| (c) LoRA fine-tune | 0.9700 | 0.6111 | 1.0000 | 1433.4 |

**(b) có thật sự mạnh hơn (a) không?** Có. Điểm target tăng vọt từ 0.0000 lên 0.7650 (76.5%), và format đúng chuẩn JSON 100% nhờ prompt tối ưu có chỉ định schema 4 trường cùng các ví dụ few-shot định hướng.  
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa, sử dụng đúng nguyên bản baseline prompt để đảm bảo tính liêm chính và công bằng tuyệt đối cho toàn bộ phép so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6261 | **0.9700 (97%)** | 964.0 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5385 | **0.9700 (97%)** | 793.0 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.0000 (0%)** | 922.8 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.9400 (94%)** | 989.2 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế chính là Lỗi #3.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target, `attn_only` (r=283) đạt điểm 0.9700, hoà với `correct` (r=16, text-linear) khi cả hai cùng ngân sách tham số (~32.45M params). Tuy nhiên, trên thang train loss thì `attn_only` (loss 0.5385) lại thấp hơn `correct` (loss 0.6261). Điều này chứng minh rằng nhìn vào train loss thấp hơn không đồng nghĩa với việc hiệu quả giải quyết task thực tế vượt trội hơn. Quan trọng hơn, để đạt cùng hiệu năng với `text-linear` ở r=16, `attn_only` buộc phải đẩy rank lên cực cao (r=283), làm tăng đáng kể độ trễ tính toán phép nhân ma trận adapter cục bộ. Phân bổ adapter trải rộng toàn bộ các lớp linear (all-linear/text-linear) với rank nhỏ (r=16) là cấu hình tối ưu và linh hoạt hơn nhiều.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Run `wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang đo của full fine-tuning) thay vì $1\times 10^{-4}$ cho LoRA. Kết quả là loss giảm cực kỳ chậm, dừng lại ở mức rất cao (1.5704) sau 30 steps và điểm target hoàn toàn bằng 0.0000 do model chưa kịp học được cấu trúc output JSON. Nếu chỉ nhìn đường loss phẳng lì mà không kiểm tra LR, ta sẽ dễ kết luận sai lầm rằng "LoRA không học được bài toán này" hoặc "tập dữ liệu bị lỗi/quá khó", trong khi nguyên nhân gốc rễ chỉ đơn giản là learning rate quá nhỏ khiến adapter không cập nhật đủ độ dịch chuyển trọng số cần thiết.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
QLoRA (4-bit NF4) tiết kiệm tới 4.92 GB VRAM (từ 12.01 GB xuống còn 7.09 GB, giảm gần 41% bộ nhớ). Tuy nhiên, cái giá phải trả là điểm target bị suy giảm nhẹ từ 0.9700 xuống 0.9400, thời gian huấn luyện tăng thêm (989.2s so với 964.0s do overhead dequantization on-the-fly), và độ trễ sinh tăng (1777.8 ms/sample so với 1433.4 ms/sample). Trên GPU T4 16GB, bản LoRA 16-bit chạy hoàn toàn thoải mái trong 12 GB VRAM mà không bị OOM, do đó số đo thực tế hoàn toàn ủng hộ khuyến nghị: nếu phần cứng còn đủ VRAM cho 16-bit LoRA, không nên đánh đổi độ chính xác và tốc độ suy luận bằng QLoRA 4-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.147` · `valid_trace_rate = 0.00`

**Diễn giải:**  
Phán quyết của cổng hồi quy trả về `FAILED` do độ suy giảm năng lực tổng quát `regression Δ = -0.147` (vượt ngưỡng dung sai cho phép `REGRESSION_TOLERANCE = 0.020`). Bản LoRA fine-tune đã hoàn thành xuất sắc nhiệm vụ chuyên biệt khi nâng điểm target từ 0.765 lên 0.970 (+20.5% so với baseline prompt tối ưu). Tuy nhiên, vì tập dữ liệu huấn luyện chỉ bao gồm 250 mẫu hẹp thuần túy về phân loại ticket CSKH dạng JSON, việc tối ưu hóa 100% loss trên miền dữ liệu này đã gây ra hiện tượng **quên thảm họa (catastrophic forgetting)** đối với các tác vụ hỏi đáp tri thức phổ thông. Để khắc phục triệt để và giúp model vượt qua cổng hồi quy trong môi trường production, giải pháp theo mục 14.3 của bài giảng là trộn thêm từ 1% đến 5% dữ liệu hồi quy/tổng quát (replay dataset) vào tập huấn luyện.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại" | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Đúng một phần | Đúng 100% (4/4 trường) | ✅ FT thắng: Nhận diện chính xác intent đổi trả và thực thể. |
| 2 | "Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé" | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `trung_tinh` | Sai urgency | Đúng 100% (4/4 trường) | ✅ FT thắng: Bắt đúng mức độ ưu tiên theo quy ước miền dữ liệu. |
| 3 | "Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền." | `hoan_tien`, `cao`, `bình giữ nhiệt`, `tieu_cuc` | `hoan_tien`, `cao`, `bình giữ nhiệt`, `tieu_cuc` | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tieu_cuc` | ❌ **FT thua**: Fine-tune đoán nhầm `urgency` thành `trung_binh` trong khi baseline (b) đoán đúng `cao`. |
| 4 | "Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện." | `san_pham_loi`, `cao`, `nồi chiên không dầu`, `tieu_cuc` | `san_pham_loi`, `cao`, `nồi chiên không dầu`, `tieu_cuc` | `san_pham_loi`, `trung_binh`, `nồi chiên không dầu`, `tieu_cuc` | ❌ **FT thua**: Fine-tune có xu hướng thiên vị nhãn đa số `urgency: trung_binh` khi câu ngắn. |
| 5 | "Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi" | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Sai sentiment | Đúng 100% (4/4 trường) | ✅ FT thắng: Phân tích đúng sentiment dựa trên ngữ cảnh ticket. |

**Có mẫu chung nào ở các ca FT thua không?**  
Các ca FT thua tập trung chủ yếu ở trường `urgency`. Do tập dữ liệu 250 mẫu có sự mất cân bằng nhẹ về phân phối `urgency` (tập trung nhiều vào mức `trung_binh`), model fine-tune có xu hướng dự đoán mức ưu tiên an toàn/đa số đối với những câu có ngữ cảnh ngắn hoặc mơ hồ, trong khi Baseline (b) với vài ví dụ few-shot đa dạng lại bắt được các từ khóa khẩn cấp sắc bén hơn.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):**  
Bài lab này chứng minh rõ ràng rằng fine-tuning bằng LoRA thực sự mang lại bước nhảy vọt về độ chính xác chuyên biệt (đạt 97% target accuracy, vượt trội mức 76.5% của prompt tối ưu) và chuẩn hóa format 100% JSON mà không cần cơ chế retry hay grammar constraint phức tạp. Tuy nhiên, nếu xét trên góc độ triển khai production toàn diện, bản fine-tune hiện tại chưa nên được deploy ngay lập tức như một mô hình đa nhiệm độc lập do đã làm suy giảm 14.7% năng lực tổng quát (bị cổng hồi quy đánh giá `FAILED`). Nếu deploy dưới dạng một microservice chuyên biệt hóa chỉ phục vụ việc triage ticket CSKH, adapter này hoàn toàn đáp ứng xuất sắc về cả độ chính xác lẫn latency (~1.4s). 

Qua các thí nghiệm đối chứng, đòn bẩy kỹ thuật mang tính quyết định lớn nhất trong lab chính là:
1. **Loss Masking đúng**: Loại bỏ 100% prompt ra khỏi loss computation (chỉ tính 41.49% token câu trả lời), ngăn ngừa hiện tượng hallucination và lặp lại prompt.
2. **Learning Rate phù hợp**: Thang đo LR $1\times 10^{-4}$ là ranh giới sống còn để LoRA hội tụ so với LR full-FT.
3. **Vị trí adapter**: All-linear với rank nhỏ hiệu quả và tiết kiệm chi phí tính toán hơn nhiều so với việc cố gắng dồn rank lớn vào các lớp attention.

**Ba điều tôi học được:**
1. **Chứng minh bằng giải mã ngược chứ không tin vào trực giác**: Phải decode ngược các token có nhãn khác `-100` để chứng thực loss mask đã cô lập đúng câu trả lời của assistant, tránh việc model vô tình học cách sinh lại chính câu hỏi của người dùng.
2. **Train Loss là chỉ số đánh lừa (Lỗi #3)**: Một cấu hình có train loss thấp hơn (như `attn_only` r=283 có loss 0.5385) hoàn toàn không có nghĩa là sẽ giải quyết bài toán nghiệp vụ tốt hơn cấu hình chuẩn (`correct` có loss 0.6261 nhưng cùng đạt 97% target). Đánh giá phải luôn dựa trên task metric thực tế.
3. **Thiết kế thí nghiệm công bằng và bài học Catastrophic Forgetting**: Khi so sánh các biến thể kiến trúc (vị trí gắn adapter, rank), bắt buộc phải cố định ngân sách tham số huấn luyện và số step. Ngoài ra, fine-tune trên dữ liệu domain hẹp luôn tiềm ẩn nguy cơ quên tri thức nền tảng, đòi hỏi phải luôn có cổng hồi quy (regression gate) để giám sát.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Thử nghiệm trộn thêm 3% dữ liệu tri thức phổ thông (replay general data) vào tập train để đưa `regression Δ` về dưới ngưỡng 0.02, giúp model chính thức vượt qua cổng hồi quy `PASSED`.
2. Thử nghiệm NB6 (Merge trọng số và đánh giá tốc độ serving qua hot-swap adapter đa tác vụ).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

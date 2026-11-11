# LAB 08 · Pretrained Transformers and BERT Fine-Tuning

## Chạy code

1. Mở thư mục `lab08` trong Jupyter và chọn kernel Python có `torch`, `transformers`, `datasets`, `huggingface_hub`, `scikit-learn`, `numpy`, `pandas`, `matplotlib` và `psutil`. Lần tải đầu cần truy cập Hugging Face. Notebook tự chọn CUDA khi có.
2. Chạy và lưu [tokenization.ipynb](tokenization.ipynb), rồi chạy toàn bộ [fine_tuning.ipynb](fine_tuning.ipynb) theo thứ tự và lưu kèm output. Mục 13 gọi [pretrained_classification.py](pretrained_classification.py). Script cũng chạy độc lập bằng `python pretrained_classification.py`.
3. Nộp các file theo Mục 25. Ba hình và bảng kết quả tổng hợp được lưu ở gốc `lab08`. Checkpoint, log từng epoch, logits, dự đoán, bảng chi tiết, cấu hình và danh sách mẫu nằm trong `outputs/`.

## Thiết lập đã dùng

- BERT-Tiny với cùng kiến trúc và tokenizer ở hai nhánh. IMDb có 22.500 mẫu train, 2.500 mẫu validation và 25.000 mẫu test chính thức. Các tập con lồng nhau và giữ cân bằng nhãn với chênh lệch tối đa một mẫu. Bốn mức có 2.250, 5.625, 11.250 và 22.500 mẫu.
- So sánh domain dùng cùng checkpoint IMDb với 25.000 mẫu test IMDb và 25.000 mẫu test Yelp. Yelp dùng 1–2 sao cho nhãn negative, 4–5 sao cho positive, bỏ 3 sao. Hai tập test có cùng số mẫu ở mỗi nhãn.
- Các lần chạy dùng hai epoch và lưu checkpoint có F1 validation cao nhất. Precision, recall và F1 lấy nhãn positive làm nhãn đích. Tập test dùng để báo cáo.
- Thời gian huấn luyện tính trong vòng lặp train. Bộ nhớ GPU là mức cấp phát đỉnh của PyTorch. Bộ nhớ CPU là RSS của tiến trình được lấy mẫu trong khi train. Ngưỡng confidence cao là 0,80.

## Các mục có code

| Mục W8 | Cell | Kết quả lưu |
| --- | --- | --- |
| 12.1 Tokenization | 1 | `outputs/tokenization.json` |
| 13 Classification Head | 2 | `outputs/classification_head.json` |
| 14 So sánh khởi tạo | 3 | `outputs/experiment1_initialization.csv`, checkpoint và log |
| 15 Frozen và fine-tuning | 4 | `outputs/experiment2_freezing.csv`, checkpoint và log |
| 16 Lượng dữ liệu | 5 | `data_efficiency.png`, `outputs/experiment3_data_efficiency.csv`, manifest, checkpoint và log |
| 17 Độ dài chuỗi | 6 | `outputs/experiment4_sequence_length.csv` có thời gian và bộ nhớ, checkpoint và log |
| 18 Domain shift | 7 | `outputs/experiment5_domain_shift.csv`, manifest và logits của Yelp |
| 19 Phân loại cảm xúc | 8 | `confusion_matrix.png`, `outputs/confusion_matrix_counts.csv`, `outputs/application_metrics.csv` |
| 20 Chuẩn bị dữ liệu phân tích lỗi | 9 | Bảng mẫu 3 FP, 3 FN và 2 ca sai có confidence cao. Lưu văn bản đầy đủ trong `outputs/error_examples.csv`, kèm `outputs/error_candidates.csv` và `outputs/error_counts.csv`. |
| 21 Confidence | 10 | `confidence_analysis.png`, `outputs/confidence_summary.csv` |
| 22 Ablation | 11 | `outputs/ablation_frozen_vs_finetuned.csv` |
| 23 Tái lập | 12 | `outputs/reproducibility.json` |
| 25 Tổng hợp | 13 | `results.csv` |

Mục 1–8 cung cấp kiến thức nền. Mục 26 là rubric và Mục 30 dẫn sang LAB 09.

## Các phần tự làm và nguồn đối chiếu

Tự viết nội dung, scan PDF, rồi lưu đúng tên dưới thư mục `scans/` để các liên kết tương đối trong file Markdown mở được. Có thể thay đường dẫn nếu dùng tên scan khác. Mục 25 nêu các file Markdown. Các liên kết PDF trong gói này dành cho bản scan bạn bổ sung.

| Mục W8 | Nội dung bạn tự làm | Dùng output? | File chứa liên kết bản scan |
| --- | --- | --- | --- |
| 9 | Tính MLM loss bằng tay và trả lời ba câu đi kèm | Không. Dùng xác suất trong đề. | [calculations.md](calculations.md) |
| 10 | Ghi token, ID, special token và mask, rồi tự đếm và so sánh | Có. Tự nhập câu đề bài vào biến `texts` ở Cell 1 và chạy tokenizer. | [calculations.md](calculations.md) |
| 11 | Tự ghi cả ba dự đoán theo yêu cầu trước thí nghiệm | Không. Viết trước khi mở bảng và hình kết quả. | [prediction.md](prediction.md) |
| 12.1 và 13 | Giải thích `input_ids`, `attention_mask` và vai trò của classification head | 12.1 dùng output Cell 1 để đối chiếu. Mục 13 có thể tham chiếu thông tin head ở Cell 2. | [calculations.md](calculations.md) |
| 14 | Trả lời bốn câu hỏi, đối chiếu dự đoán | Có. Dùng bảng Experiment 1 và dự đoán đã viết. | [error_analysis.md](error_analysis.md) |
| 15 | Trả lời hai câu về frozen và fine-tuning | Không bắt buộc cho phần giải thích. Dùng bảng Experiment 2 khi đối chiếu chênh lệch. | [error_analysis.md](error_analysis.md) |
| 16 | Trả lời câu hỏi về hiệu quả sử dụng dữ liệu | Có. Dùng cả đường cong và bảng Experiment 3. | [error_analysis.md](error_analysis.md) |
| 17 | Trả lời câu hỏi về độ dài chuỗi, liên hệ chi phí attention | Có. Dùng Accuracy, F1, thời gian và bộ nhớ ở Experiment 4. | [error_analysis.md](error_analysis.md) |
| 18 | Trả lời câu hỏi về domain shift | Có. So sánh kết quả IMDb và Yelp của cùng checkpoint. | [error_analysis.md](error_analysis.md) |
| 20 | Tự chọn ít nhất 3 FP, 3 FN và 2 ca rất tự tin nhưng sai. Tự điền loại lỗi và giả thuyết. | Có. Dùng bảng mẫu ở Cell 9 hoặc chọn ca khác từ `outputs/error_candidates.csv`. Văn bản đầy đủ của tám ca gợi ý nằm trong `outputs/error_examples.csv`. | [error_analysis.md](error_analysis.md) |
| 21 | Trả lời câu hỏi về confidence và correctness | Có. Dùng biểu đồ, số mẫu và độ chính xác từng nhóm. | [error_analysis.md](error_analysis.md) |
| 22 | Tự nhận xét ablation đã chọn | Có. Dùng bảng frozen và fine-tuned. | [error_analysis.md](error_analysis.md) |
| 23–24 | Ghi cấu hình tái lập và tự khai báo phần được AI hỗ trợ, phần bạn sửa, phần bạn xác minh | Mục 23 dùng `outputs/reproducibility.json`. Mục 24 dựa trên công việc bạn thực sự làm. | README này |
| 27 | Tự trả lời 12 câu Individual Learning Check bằng lời của mình | Không bắt buộc. Có thể dùng output làm ví dụ nếu cần. | README này |
| 28 | Tự trả lời Q1–Q6, tách quan sát, bằng chứng và diễn giải | Có. Q1 dùng Experiment 3, Q2 dùng Experiment 1, Q3–Q4 dùng Experiment 2, Q5 dùng Experiment 5 và Q6 dùng Experiment 4. | [error_analysis.md](error_analysis.md) |
| 29 | Tự viết reflection 300–500 từ theo sáu ý của đề | Không bắt buộc. Có thể dẫn kết quả bạn đã kiểm tra. | [reflection.md](reflection.md) |

## Mục 23 và 24

- [Cấu hình, phần cứng và thời gian đo được](outputs/reproducibility.json)
- [Bản scan khai báo AI Usage](scans/ai_usage.pdf)
  - `AI-assisted code`
  - `What was generated`
  - `What was modified`
  - `What was verified`

## Mục 27

- [Bản scan 12 câu Individual Learning Check](scans/learning_check.pdf)

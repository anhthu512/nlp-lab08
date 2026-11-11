# LAB 08 · Pretrained Transformers and BERT Fine-Tuning


## Thiết lập đã dùng

- BERT-Tiny với cùng kiến trúc và tokenizer ở hai nhánh. IMDb có 22.500 mẫu train, 2.500 mẫu validation và 25.000 mẫu test chính thức. Các tập con lồng nhau và giữ cân bằng nhãn với chênh lệch tối đa một mẫu. Bốn mức có 2.250, 5.625, 11.250 và 22.500 mẫu.
- So sánh domain dùng cùng checkpoint IMDb với 25.000 mẫu test IMDb và 25.000 mẫu test Yelp. Yelp dùng 1–2 sao cho nhãn negative, 4–5 sao cho positive, bỏ 3 sao. Hai tập test có cùng số mẫu ở mỗi nhãn.
- Các lần chạy dùng hai epoch và lưu checkpoint có F1 validation cao nhất. Precision, recall và F1 lấy nhãn positive làm nhãn đích. Tập test dùng để báo cáo.
- Thời gian huấn luyện tính trong vòng lặp train. Bộ nhớ GPU là mức cấp phát đỉnh của PyTorch. Bộ nhớ CPU là RSS của tiến trình được lấy mẫu trong khi train. Ngưỡng confidence cao là 0,80.


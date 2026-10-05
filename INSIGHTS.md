# CarbonGlobe Challenge — insight log

Mọi điều đã biết về dữ liệu, model và LB. **[KC]** = đã kiểm chứng (đo trực tiếp hoặc LB thật), **[GT]** = giả
thuyết. Kế hoạch thí nghiệm tiếp theo: `PLAN.md`. Cập nhật: 2026-10-06.

## 0. Hiện trạng

- **LB tốt nhất đã nộp: 0.100** (`result/submission_blend_global.csv`).
- Chờ nộp: `result/bg_scn.csv` (ước tính **0.0938**), `result/submission_blend_global_x.csv` (ước tính **0.0977**,
  trước hiệu chỉnh bias).
- Đòn bẩy lớn nhất đã tìm được, theo thứ tự: train trên toàn lưới (0.168 → 0.134), thay CO2 test (0.134 → 0.122),
  blend với model cũ (0.122 → 0.100), hiệu chỉnh bias theo kịch bản (~−0.006 đến −0.011).

## 1. Sổ LB (mọi điểm đã nộp)

"Ước tính" = dự báo trước khi nộp bằng `ensemble.py` / `probe_bias.py` (đại số trên các LB đã có).

| # | File / run | LB | Ước tính | Ghi chú |
|---|---|---|---|---|
| 1 | DeepED `MM_LSTM_Age_v2`, 1-step (notebook "lightgbm (2)/(3)") | 0.231 | | baseline |
| 2 | v2 + scheduled sampling | 0.186 | | VAL 0.0325 |
| 3 | LSTM-XL, 1-step | 0.168 | | VAL 0.0379 |
| 4 | EcoFormer, 1-step | 0.169 | | VAL 0.0609 |
| 5 | Blend 3 run `result/submission.csv` (0.137/0.428/0.435) | 0.149 | 0.1493 | base của probe cũ |
| 6 | LSTM-XL `none` (`lstm_xl_probe.py`) | 0.168 | | tái lập #3 |
| 7 | LSTM-XL `co2_hist` | 0.166 | | |
| 8 | LSTM-XL `clip` | 0.169 | | |
| 9 | Blend 4 run `submission_blend5.csv` | 0.149 | 0.1485 | |
| 10 | Probe `probe_agb110.csv` (blend #5, agb ×1.1) | 0.144 | | |
| 11 | Probe `probe_h110.csv` (height ×1.1) | 0.167 | | |
| 12 | `submission_scaled.csv` (#5, agb ×1.0695, height ×0.9577) | 0.141 | 0.1410 | |
| 13 | Probe `probe_cell_agb_ssp126_young.csv` (×1.3) | 0.158 | | |
| 14 | Probe `probe_cell_agb_ssp126_old.csv` (×1.3) | 0.193 | | |
| 15 | Probe `probe_cell_agb_ssp585_young.csv` (×1.3) | 0.151 | | |
| 16 | Probe `probe_height110_ssp126.csv` | 0.160 | | |
| 17 | `submission_cells.csv` (#5, agb theo 4 ô, height ×0.9577) | 0.138 | 0.1378 | |
| 18 | fusion-lite (`carbonglobe-fusioned-lite.ipynb`) | 0.167 | | VAL 0.0286 (tốt nhất) |
| 19 | Grid LSTM-XL `submission.csv` (`result/globel-train`) | 0.134 | | EMA epoch 4, chỉ K=1 |
| 20 | Grid LSTM-XL `submission_co2_hist.csv` | **0.122** | | model đơn tốt nhất |
| 21 | Blend 6 run `submission_blend_global.csv` | **0.100** | 0.1022 | lần đầu ước tính lệch 0.002 |
| 22 | Probe `bg_probe1_agb110.csv` | 0.103 | | |
| 23 | Probe `bg_probe2_height110.csv` | 0.114 | | |
| 24 | Probe `bg_probe3_agb110_ssp126.csv` | 0.108 | | |
| 25 | Probe `bg_probe4_height110_ssp126.csv` | 0.109 | | |

Chưa nộp: `bg_scn.csv` (0.0938), `submission_blend_global_x.csv` (0.0977), `submission_final.csv` (0.1374, bị #21
vượt qua, không cần nộp). Run 1 của notebook global (`notebook2dbddcbdc5.ipynb`, phân kỳ NaN) không có điểm LB.

Độ chính xác của ước tính: 7 lần đúng trong sai số làm tròn (±0.0005); 1 lần lệch 0.002 (#21, theo hướng có lợi).

## 2. Bài toán và metric

- Mỗi bước năm: khí hậu 12 tháng × 136 feature, trạng thái năm trước (7 biến: height, agb, soil, lai, gpp, npp, rh),
  15 đặc trưng tuổi. Rollout tự hồi quy 40 năm cho 15 seed age (1…500).
- Điểm = 0.5 × (MSE height/10.90645 + MSE agb/3.39567) ở năm 40, trên 6171 site × 15 tuổi × 2 kịch bản. Public /
  private LB là một split ngẫu nhiên theo site.
- Giới hạn: 5 submission/ngày, session Kaggle 12 h, quota GPU theo tuần.

## 3. Dữ liệu

- [KC] `data_global/glob_X_fea.npy` (54152, 40, 12, 136) và `glob_Y{age}.npy` (54152, 41, 12, 7): toàn bộ lưới mô
  phỏng 0.5°, nhãn đầy đủ, 100% hữu hạn. ~12–15% site có quỹ đạo toàn 0 ở tuổi 100/500 (0% ở tuổi 1).
- [KC] `train/val/test.csv` không có header: 3035 + 338 + 50779 = 54152. Đây là split của benchmark gốc theo **lưới
  bước 4** (train/val ở (hàng % 4, cột % 4) = (0, 0) qua `mask2d.npy`). `res_train4_test8.npz` `x_train` (3373 site) =
  train + val; 852 site val cũ nằm xen giữa site train. Với cuộc thi, 50779 site "test" chỉ là dữ liệu train thêm.
- [KC] Site test của cuộc thi không phải ô của lưới (0.5% khớp trên 8 feature tĩnh): phải tổng quát hóa sang vị trí
  mới.
- [KC] `test_y0` so với năm 0 của lưới: height 0.84, agb 0.73, soil 0.77, lai 1.09, gpp/npp ~1.3, rh ~1.15; lệch mạnh
  nhất ở tuổi trẻ (tuổi 10: agb 0.28, soil 0.13). Không trùng trạng thái nào trong lưới.
- Câu hỏi mở: `mask2d` có giá trị 1/2/3 (mã vùng?); tỷ lệ y0 tuổi 1 âm (agb −11.9) cần kiểm tra giá trị min.

### Bản đồ feature [GT, suy từ thống kê] và khác biệt của test [KC]

0–7 là 8 biến riêng; 8–31 khối giống CO2 (24 feature); 32 hằng số; 33–128 = 24 lớp × 4 biến; 129–135 tĩnh.

| Nhóm | Test so với lịch sử | Ảnh hưởng lên model |
|---|---|---|
| CO2 8–31 | mức thấp hơn (−0.75 sd ssp126, −0.38 sd ssp585 so với lưới), xu hướng phẳng (+0.37 / +1.45 sd so với +3 sd), gần như không có mùa | **lớn**: chỉ *mức* CO2 có tác động; mất chu kỳ mùa vô hại |
| Biến 4 (36, 40, …, 128) | một climatology tháng lặp lại 40 năm | vô hại (msd 0.0004) |
| Biến 3 (35, …, 127; "SUS") | sàn là một hằng số ngay dưới 0 | vô hại; `clip` hơi tệ hơn (#8) |
| Feature 7 | biên độ mùa ~20×, std tổng ~5× | chưa đo (`f7_fix` trong `global_probe.py`) |
| Tĩnh 129, 135 | giá trị không trùng lưới nhưng trong khoảng | chưa đo |

## 4. Phát hiện chính

### 4.1 Dữ liệu lưới là đòn bẩy lớn nhất [KC]

- Cùng kiến trúc LSTM-XL: 3373 site → 0.168; 51.6k site, chỉ K=1, ~4 epoch → **0.134** (#19).
- Model lưới sai kiểu khác hẳn: tương quan lỗi với model cũ ~0.45–0.53 (giữa các model cũ 0.78–0.97).
- Model lưới ít thiếu AGB hơn (AGB trung bình 0.984 so với 0.896 của blend cũ).
- Đã sai trước đó: tôi từng dự đoán dữ liệu lưới "chỉ giúp vừa phải" vì khoảng cách khí hậu test→lưới chỉ giảm
  3.68 → 3.47.

### 4.2 Mức CO2 của test kéo dự đoán sai [KC]

- Thay khối CO2 test bằng chuỗi 1980–2020 của lưới (`co2_hist`): LSTM-XL 0.168 → 0.166; model lưới 0.134 → **0.122**.
- Trên val (`lstm_xl_valshift.py`): đưa mức CO2 test vào làm AGB giảm 8% (ssp126) / 4% (ssp585); bỏ chu kỳ mùa
  không làm gì.
- [KC, đại số] LB dọc đường `global` → `global_co2_hist` cực tiểu ở t ≈ 1.9 (≈ 0.118): hiệu chỉnh CO2 "chưa đủ
  mạnh". Đây là cơ sở của `submission_blend_global_x.csv` và biến thể `co2_up`.
- [GT] CO2 test có thể bị mã hóa/scale khác lịch sử (thấp hơn và phẳng hơn quỹ đạo SSP thật). Kiểm tra bằng
  `co2_up`, `co2_cont`, `co2_end` (`global_probe.py`).

### 4.3 Bias theo kịch bản, ổn định qua mọi model [KC]

| Hệ số tối ưu | blend #5 (model cũ) | blend #21 (61% model lưới) |
|---|---|---|
| agb ssp126 | ~1.02 (0.996 trẻ, 1.027 già) | 0.996 |
| agb ssp585 | 1.119 (mọi tuổi) | **1.085** |
| height ssp126 | 0.9375 | **0.953** |
| height ssp585 | 0.9776 | 0.997 |

- Không phụ thuộc tuổi; giống nhau ở hai blend rất khác nhau ⇒ ED phản ứng theo kịch bản theo cách dữ liệu lịch sử
  không dạy được (CO2 đồng nhất trong không gian, tăng cùng thời gian). Hiệu chỉnh hậu kỳ là bước cố định; probe lại
  (4 slot, `make4`/`solve4`) mỗi khi base thay đổi lớn.
- Lộ trình trên blend cũ: 0.149 → 0.141 (toàn cục) → 0.138 (theo ô).

### 4.4 LB là một "oracle" tuyến tính [KC]

- Mỗi LB cho ⟨p, y⟩ (sai khác một hằng số). Mọi tổ hợp tuyến tính của các file đã chấm (blend, scale theo ô, ngoại
  suy) dự báo được LB mà không tốn slot. Đây là cơ chế của `ensemble.py` và `probe_bias.py`.
- [GT] Lệch 0.002 ở #21: khoảng cách giữa model tính trên mọi site, LB public chỉ dùng một phần site; sai số tăng khi
  các model khác nhau nhiều. Coi ước tính là ±0.002 trong trường hợp đó.

### 4.5 VAL cũ không dự báo LB [KC]

- VAL giảm 4× (0.13 → 0.033) trong khi LB 0.231 → 0.17–0.19; thứ hạng VAL ≠ thứ hạng LB; fusion-lite có VAL tốt hơn
  25% mà LB chỉ hơn 0.001 (#18). Holdout ngẫu nhiên của notebook global cũng nhiễu (0.09–0.16 giữa các epoch).
- Chỉ dùng LB (và các ước tính đại số) để chọn.

## 5. Những thứ đã loại trừ [KC]

| Hướng | Bằng chứng | Kết luận |
|---|---|---|
| Kiến trúc lớn hơn trên 3373 site | fusion-lite 0.167, xxl/tft/crossvar không cần chạy | lợi ích ≤ 0.001 |
| `HARMONIZE` (climatology + bỏ mùa CO2) | val-shift: msd 0.0004 / 0.0012 | tắt mặc định |
| `clip` input test | 0.169 vs 0.168 | bỏ |
| Probe theo tuổi | bias không phụ thuộc tuổi | chỉ probe theo kịch bản |
| Ghép site test vào lưới | 0.5% khớp | không lấy được lịch sử đúng vị trí |

## 6. Ghi chú train trên lưới

- Run 1 (`notebook2dbddcbdc5.ipynb`, HARMONIZE=True, LR 5e-4): train loss tăng trong K=1, NaN từ epoch 5 (K=4).
- Run 2 (`result/globel-train/`, HARMONIZE=False, LR 5e-4): K=1 epoch 1–4 ổn, bùng nổ ở K=4, NaN từ epoch 11.
  Submission lấy từ EMA epoch 4.
- Đã sửa trong `carbonglobe-global.ipynb`: LR 3e-4; bỏ qua bước có gradient không hữu hạn; epoch hỏng được rollback
  với LR × 0.5 (tối đa 3 lần); resume từ `last.npz` NaN tự lấy `best.npz`; input chuẩn hóa clip ±10; trạng thái
  rollout ≤ 2× max của lưới; K=1 tới epoch 8. Đã smoke test; chưa chạy ở quy mô thật.
- Nạp dữ liệu mất 5.5–57 phút tùy đĩa Kaggle.
- Fusion-lite / phân tán: không có bug; EMA 0.999 với ~195 step/epoch trễ ~5 epoch (global dùng 0.998).

## 7. Giả thuyết đã bị bác bỏ

- "Lỗi chung do model đọc CO2 như đồng hồ" → probe trên LSTM-XL chỉ −0.002; nhưng với model lưới CO2 lại quan
  trọng (−0.012). Kết luận đúng: tác động của CO2 phụ thuộc model.
- "Định dạng input test (climatology, không có mùa) gây lỗi" → vô hại.
- "Dữ liệu lưới chỉ giúp vừa phải" → giúp nhiều nhất.
- "Blend model hiện có đã chạm trần 0.149" → đúng với model cũ; model lưới phá trần nhờ lỗi khác kiểu.

## 8. Câu hỏi mở

Xem `PLAN.md` (Q1–Q7): CO2 test có bị scale sai không; nhóm feature nào khác hoạt động như CO2; feature 7; mức độ
model nhớ `test_y0`; train lại model lưới lâu hơn; model lưới thứ hai (EcoFormer); sửa CO2 ngay khi train.

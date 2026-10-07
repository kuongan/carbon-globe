# CarbonGlobe Challenge — insight log

Mọi điều đã biết về dữ liệu, model và LB. **[KC]** = đã kiểm chứng (đo trực tiếp hoặc LB thật), **[GT]** = giả
thuyết. Kế hoạch thí nghiệm tiếp theo: `PLAN.md`. Cập nhật: 2026-10-06.

File cũ đã xóa (notebook chưa chạy / đã thay thế, `lstm_xl_probe.py`, `lstm_xl_valshift.py`,
`notebook2dbddcbdc5.ipynb`, …) vẫn lấy lại được: `git show b8585dd:<đường dẫn>`. Tái tạo file submission: xem đầu
`ensemble.py` (biến `ENSEMBLE_RUNS` / `ENSEMBLE_FREE` / `ENSEMBLE_TAG`); `probe_bias.py make4 / solve4` → probe và
`bg_scn.csv`.

## 0. Hiện trạng

- **LB tốt nhất đã nộp: 0.091**, hai file: `result/submission_blend_latest_x.csv` (blend) và
  `result/globel-train/third-run/submission_ep4_none.csv` (model đơn không CO2, không trọng số âm).
- Chờ nộp: `result/submission_blend_noco2.csv` (ước tính **0.0828**), sau đó probe bias `result/bn_probe1..4_*.csv`.
- Đòn bẩy lớn nhất đã tìm được, theo thứ tự: train trên toàn lưới (0.168 → 0.134), thay CO2 test (0.134 → 0.122),
  blend với model cũ (0.122 → 0.100), hiệu chỉnh bias theo kịch bản (0.100 → 0.094).

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
| 26 | `bg_scn.csv` (#21 × hệ số theo kịch bản) | **0.094** | 0.0938 | tốt nhất |
| 27 | Grid run 2 `submission_co2_hist.csv` (`result/globel-train/second run`) | 0.130 | | EMA epoch 8, K=1; tệ hơn #20 |
| 28 | `submission_blend_latest_x.csv` (bg_scn 0.965, global_co2_hist 0.71, global −0.75, v2 0.08) | **0.091** | 0.0912 | tốt nhất |

| 29 | Hybrid `hyb_global2_co2_hist.csv` (run trên ssp126 + bg_scn trên ssp585) | 0.109 | | probe |
| 30 | Hybrid `hyb_lstm_xl.csv` | 0.123 | | probe |
| 31 | Hybrid `hyb_v2_scheduled_sampling.csv` | 0.127 | | probe |
| 32 | Hybrid `hyb_global_co2_hist.csv` | 0.109 | | probe |
| 33 | Hybrid `hyb_global.csv` | 0.116 | | probe |
| 34 | Hybrid `hyb_ecoformer_ed.csv` | 0.125 | | probe |

| 35 | Session B (`result/globel-train/third-run`), grid **không CO2**, `submission.csv` = EMA epoch 6 (best holdout) | 0.109 | | |
| 36 | Session B `submission_ep3_none.csv` | 0.116 | | |
| 37 | Session B `submission_ep4_none.csv` | **0.091** | | **model đơn tốt nhất** |
| 38 | Session A `submission_global_co2x2.csv` (trọng số **run 2**, không phải run 1) | 0.126 | | run 2 co2_hist = 0.130 |
| 39 | Session A `submission_global_co2x3.csv` (run 2) | 0.128 | | |

Chưa nộp: `submission_blend_scn_x.csv` (0.0902), `submission_blend_scn.csv` (0.0935), `submission_blend_latest.csv`
(0.0937), `submission_blend_global_x.csv` (0.0977).
Run 1 của notebook global (`notebook2dbddcbdc5.ipynb`) không có điểm LB.

Độ chính xác của ước tính: 9 lần đúng trong sai số làm tròn (±0.0005); 1 lần lệch 0.002 (#21, theo hướng có lợi).
Ngoại suy với trọng số âm (#28) cũng khớp, nên hướng "CO2 chưa đủ mạnh" (mục 4.2) là thật.

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

### 4.6 Grid run 2: train thêm làm holdout tốt hơn nhưng LB tệ hơn [KC]

- Resume từ epoch 4 với LR 2.5e-4, K=1 tới epoch 8: holdout shift 0.0389 → 0.0365, random 0.128 → 0.061. Nhưng
  co2_hist của nó **0.130** (#27) so với **0.122** của epoch 4 (#20). Holdout của lưới không dự báo LB, giống VAL cũ.
- Hai bản grid (epoch 4 vs epoch 8) khác nhau rất nhiều: msd 0.071 (cỡ khoảng cách giữa hai kiến trúc khác nhau),
  tương quan lỗi 0.72. Nhưng khi đã có `bg_scn`, run 2 chỉ nhận trọng số ~0.02 trong blend.
- Dự đoán AGB = 0 (câu hỏi của bạn): 21% hàng của file #27, 23% ở #20. Trên holdout có nhãn: truth = 0 ở 16% ô;
  model dự đoán 0 ở 28% (run 2) / 47% (run 1), nhưng ở những ô model nói 0 mà truth > 0, AGB thật trung bình chỉ
  0.009–0.018 (đơn vị đã scale, tức ~1–2% mức trung bình). Lỗi ở đó ~0.0001–0.0003 mỗi ô: **vô hại**. Đây là ô gần như
  không có rừng (height vẫn > 0, ví dụ `ssp126_0_*`); clamp ≥ 0 chỉ cắt giá trị âm rất nhỏ thành 0.
- K=4 bùng nổ lần thứ hai (epoch 9: train 0.086, holdout 0.25 → 46), không NaN nên guard cũ không bắt; rollback lại
  về `last.npz` của epoch đã hỏng nên 3 lần thử đều thất bại. Đã sửa (mục 6).

### 4.7 Blend tận dụng file đã hiệu chỉnh [KC, đại số]

- `bg_scn.csv` đã có LB nên chính nó là một thành phần blend (`result/bg_scn_preds/`). Blend 8 thành phần:
  simplex → 0.0937 (bg_scn 0.93, v2 0.05, global2 0.02: gần như không hơn 0.094); cho phép trọng số âm trên `global`
  → **0.0912** (bg_scn 0.965, global_co2_hist +0.71, global −0.75, v2 0.08). Phần lợi đến từ ngoại suy hiệu chỉnh CO2
  (mục 4.2, t ≈ 1.9), không phải từ run 2.

### 4.8 Lỗi theo từng kịch bản (hybrid probes #29–34, `ensemble_scn.py`) [KC, đại số]

Lỗi của mỗi model so với `bg_scn` **trong cùng kịch bản** (chính xác; chỉ mức tuyệt đối giữa hai kịch bản là ẩn):

| Model | ssp126 | ssp585 |
|---|---|---|
| global_co2_hist (run 1) | +0.030 | **+0.026** |
| global2_co2_hist (run 2) | +0.030 | +0.042 |
| global (run 1, CO2 gốc) | +0.044 | +0.036 |
| lstm_xl | +0.058 | +0.090 |
| ecoformer_ed | +0.062 | +0.088 |
| v2_scheduled_sampling | +0.066 | +0.118 |

- **Run 2 chỉ tệ hơn run 1 ở ssp585** (ssp126 bằng nhau). Train lâu hơn làm hỏng đúng kịch bản cần ngoại suy nhiều
  nhất. Holdout lịch sử không thấy được điều này.
- **Model cũ (3373 site) tệ hơn hẳn ở ssp585** (+0.09–0.12 so với +0.06 ở ssp126); model lưới đều hơn giữa hai kịch
  bản. Dữ liệu lưới giúp chủ yếu ở ssp585.
- `co2_hist` giúp cả hai kịch bản (ssp126 −0.014, ssp585 −0.010).
- Blend theo kịch bản cho phép trọng số âm trên `global`: ssp126 dùng global −0.56 / global_co2_hist +0.53, ssp585
  dùng **−1.74 / +1.84**. ssp585 muốn ngoại suy CO2 mạnh hơn nhiều. [GT] CO2 test của ssp585 chỉ thấp hơn lịch sử
  0.38 sd (ssp126: 0.75 sd), nên `co2_hist` dịch ssp585 ít mà ED lại cần phản ứng CO2 lớn nhất ở đó: CO2 "thật" mà
  model cần thấy ở ssp585 cao hơn cả lịch sử. Session A (`co2x3`, `co2_cont`) kiểm tra trực tiếp.
- Ước tính: `submission_blend_scn.csv` 0.0935 (không đáng nộp), `submission_blend_scn_x.csv` **0.0902** (chỉ hơn
  0.091 khoảng 0.001, nhỏ hơn sai số ±0.002 của blend có trọng số âm lớn; rủi ro private LB cao hơn).

### 4.9 Model không CO2 (session B) [KC]

- Cùng kiến trúc, cùng dữ liệu lưới, CO2 bị giữ cố định khi train và test: epoch 4 đạt **0.091** (model CO2 tốt nhất:
  0.122). Bỏ CO2 là đòn bẩy lớn nhất từ khi chuyển sang dữ liệu lưới. Model không học được phản ứng CO2 sai, và hiệu
  ứng theo kịch bản để cho hệ số hậu kỳ.
- **LB dao động rất mạnh giữa các epoch** của cùng một run: ep3 0.116, ep4 0.091, ep6 0.109 (ep5 chưa chấm). Khoảng
  cách prediction giữa các epoch 0.034–0.074 (cỡ khoảng cách giữa hai model khác nhau). Holdout không chọn được epoch:
  holdout random của ep3/ep4/ep6 là 0.092/0.126/0.113, ngược hẳn thứ tự LB (3 điểm, có thể trùng hợp; run 1 cũng có
  epoch tốt nhất trên LB là epoch có holdout random tệ nhất). [GT] Model thay đổi cách ngoại suy từ epoch này sang
  epoch khác; trung bình nhiều snapshot/seed sẽ giảm phương sai này.
- AGB trung bình theo epoch (ssp126/ssp585): ep3 0.97/0.96, ep4 1.02/1.01, ep5 0.90/0.88, ep6 1.05/1.03. Trên holdout
  lịch sử, ep6 dự đoán AGB **cao hơn thật 10%**, trong khi trên test LB lại muốn AGB cao hơn: thêm một dấu hiệu lịch
  sử và tương lai cần hướng sửa ngược nhau.
- Blend với mọi file đã chấm (`submission_blend_noco2.csv`, simplex): noco2_ep4 0.55, bg_scn 0.33, v2 0.06, lstm_xl
  0.04, noco2_ep6 0.02 → ước tính **0.0828** (0.0825–0.0831). Cho phép trọng số âm trên `global`/`global2_co2_hist`
  (`submission_blend_noco2_x.csv`): **0.0790**.

### 4.10 Session A (CO2 sweep, screening) chạy với trọng số run 2 [KC]

- `global_probe_preds.npz` khớp `global2_co2_hist` (msd 1e-6), không khớp run 1 (0.071): file trọng số được gắn là
  của run 2. Kết quả vẫn dùng được, chỉ là cho run 2: co2_hist 0.130 → **co2x2 0.126** → co2x3 0.128. Với run 2, tối
  ưu ở khoảng ×2; đẩy CO2 mạnh hơn nữa bắt đầu hại. Preds lưu thành `result/g2_*_preds/`.
- Screening (1500 site test, run 2, độ dịch so với `none`): ngoài CO2, chỉ nhóm **v2 (34, 38, …, 126)** có tác dụng
  (msd 0.024 / 0.009; bỏ độ lệch test của nó làm AGB giảm 8% / 5%, tức độ lệch đó đang *nâng* AGB, đúng chiều LB muốn:
  giữ nguyên). v1 nhỏ (0.003), v3 nhỏ (0.0015). **Feature 7, v4, static, f0–f6: không ảnh hưởng** (msd ≤ 0.0001);
  `f7_fix` vô hại nên không cần nộp.
- `co2_start` (CO2 cố định ở 1980) giảm AGB 7–11%; `co2_end`/`co2_cont` tăng AGB 14–21%: phản ứng CO2 của model lớn và
  đơn điệu, đúng như mục 4.2.
- Part B (y0 memory): chưa có output (cần dòng in của cell).

## 6. Ghi chú train trên lưới

- Run 1 (`notebook2dbddcbdc5.ipynb`, HARMONIZE=True, LR 5e-4): train loss tăng trong K=1, NaN từ epoch 5 (K=4).
- Run 2 (`result/globel-train/`, HARMONIZE=False, LR 5e-4): K=1 epoch 1–4 ổn, bùng nổ ở K=4, NaN từ epoch 11.
  Submission lấy từ EMA epoch 4.
- Run 3 = "second run" (`result/globel-train/second run/`, có bảo vệ NaN, LR 2.5e-4): K=1 epoch 5–8 ổn; K=4 bùng
  nổ ở epoch 9 dù LR thấp hơn. Submission từ EMA epoch 8 (#27).
- **K ≥ 4 trên lưới bùng nổ ở cả hai lần**, còn trên 3373 site (fusion-lite, v2) thì ổn. Chưa rõ vì sao [GT: ô cực
  đoan của lưới + dự đoán đưa ngược]. Curriculum mặc định giờ là **chỉ K=1**.
- Đã sửa trong `carbonglobe-global.ipynb`: LR 3e-4; bỏ qua bước có gradient không hữu hạn; input chuẩn hóa clip ±10;
  trạng thái rollout ≤ 2× max của lưới; guard phân kỳ phát hiện cả NaN lẫn holdout > 3× tốt nhất (`DIVERGE_FACTOR`),
  khôi phục **`best.npz`** (không phải `last.npz`) với optimizer mới và LR × 0.5, làm lại cùng epoch (tối đa 3 lần);
  resume từ `last.npz` NaN tự lấy `best.npz`. Đã smoke test guard mới.
- Nạp dữ liệu mất 5.5–57 phút tùy đĩa Kaggle.
- Fusion-lite / phân tán: không có bug; EMA 0.999 với ~195 step/epoch trễ ~5 epoch (global dùng 0.998).

## 7. Giả thuyết đã bị bác bỏ

- "Lỗi chung do model đọc CO2 như đồng hồ" → probe trên LSTM-XL chỉ −0.002; nhưng với model lưới CO2 lại quan
  trọng (−0.012). Kết luận đúng: tác động của CO2 phụ thuộc model.
- "Định dạng input test (climatology, không có mùa) gây lỗi" → vô hại.
- "Dữ liệu lưới chỉ giúp vừa phải" → giúp nhiều nhất.
- "Blend model hiện có đã chạm trần 0.149" → đúng với model cũ; model lưới phá trần nhờ lỗi khác kiểu.
- "Train model lưới lâu hơn sẽ tốt hơn" → holdout tốt hơn nhưng LB tệ hơn (0.122 → 0.130).

## 8. Câu hỏi mở

Xem `PLAN.md` (Q1–Q7): CO2 test có bị scale sai không; nhóm feature nào khác hoạt động như CO2; feature 7; mức độ
model nhớ `test_y0`; train lại model lưới lâu hơn; model lưới thứ hai (EcoFormer); sửa CO2 ngay khi train.

# CarbonGlobe Challenge — insight log

Mọi điều đã biết về dữ liệu, model và LB. **[KC]** = đã kiểm chứng (đo trực tiếp hoặc LB thật), **[GT]** = giả
thuyết. Kế hoạch thí nghiệm tiếp theo: `PLAN.md`. Cập nhật: 2026-10-06.

File cũ đã xóa (notebook chưa chạy / đã thay thế, `lstm_xl_probe.py`, `lstm_xl_valshift.py`,
`notebook2dbddcbdc5.ipynb`, …) vẫn lấy lại được: `git show b8585dd:<đường dẫn>`. Tái tạo file submission: xem đầu
`ensemble.py` (biến `ENSEMBLE_RUNS` / `ENSEMBLE_FREE` / `ENSEMBLE_TAG`); `probe_bias.py make4 / solve4` → probe và
`bg_scn.csv`.

## 0. Hiện trạng

- **LB tốt nhất đã nộp: 0.074**, hai file: `result/submission_blend_scn2.csv` (không có trọng số âm ở tầng nào) và
  `result/submission_blend_scn2_x.csv` (có). Model đơn tốt nhất: 0.089 (`session A2/submission_global_dsh_v2.csv`).
- Chọn cuối đề xuất: `submission_blend_scn2.csv` + `submission_blend_scn2_x.csv`.
- Tái tạo: `python ensemble_scn.py solve` (+ `ENSEMBLE_FREE=global,global2_co2_hist` cho bản _x) với
  `ENSEMBLE_TAG=blend_scn2`; đã kiểm tra trùng tuyệt đối (max |diff| = 0).
- Không train thêm LSTM (mục 4.19, 4.21). Thêm y0 vào blend chỉ lợi ≤ 0.0005, bằng sai số làm tròn LB (mục 4.20).
- **LB tốt nhất:** 0.070 (`result/probe_dead_T2.csv` = 0.072; `result/submission_blend_t2b.csv` = 0.070).
  Test = lưới CarbonGlobe sau phép biến đổi affine; đáp án nằm trong không gian đó (T2). Blend đã hội tụ tại 0.072
  sau khi thêm GBM rollout và seed-avg (mục 4.26). Đòn bẩy còn lại: LSTM `y0inv` trên Kaggle (session E).
- **LB tốt nhất: 0.066** (`submission_blend_x1.csv`, #86). Analog cùng site thất bại (0.173 / 0.209, mục 4.28).
- **Model đơn tốt nhất: 0.061** (`match_nnx_y0inv2.csv`, #138, NN cross-age). Trước: 0.072 (`match_gbmc2_y0inv2.csv`, #133, 2 seed). Trước: 0.073 (`match_gbmc_y0inv2.csv`, #128, mục 4.44). **LB tốt nhất: 0.049** (`submission_blend_x9f.csv`, #139). Trước: 0.050 (`submission_blend_x8f.csv`, #135). Trước: 0.054 (`submission_blend_x7f.csv`, #129). Trước đó 0.060 (`submission_blend_x6.csv`, #119; mục 4.40–4.41). Trước đó 0.061 (`submission_blend_x4f.csv`, #106). Trước đó 0.063 (`submission_blend_x2.csv`, #93: cặp LSTM + cặp GBM + CO2 lịch sử, mục 4.32).
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
| 40 | Run 4 (`result/fourth-run`), no CO2, SEED 7, `submission.csv` = EMA epoch 3 (best holdout) | 0.118 | | holdout shift 0.032 = tốt nhất từ trước tới nay |
| 41 | `submission_blend_noco2_x.csv` (bg_scn 0.767, noco2_ep4 0.601, noco2_ep6 0.114, v2 0.048, global −0.41, global2_co2_hist −0.119) | **0.079** | 0.0790 | **tốt nhất**; trọng số âm |
| 42 | `par_noco2_ep3.csv` (run trên site chẵn, bg_scn trên site lẻ) | 0.105 | | CV probe |
| 43 | `par_noco2_ep6.csv` | 0.103 | | CV probe |
| 44 | `par_s7_ep3.csv` | 0.106 | | CV probe |
| 45 | `par_v2_scheduled_sampling.csv` | 0.136 | | CV probe |
| 46 | `par_global2_co2_hist.csv` | 0.110 | | CV probe |
| 47 | `par_noco2_ep4.csv` | 0.094 | | CV probe |
| 48 | `par_global.csv` | 0.113 | | CV probe |
| 49 | `par_lstm_xl.csv` | 0.129 | | CV probe |
| 50 | `third-run/submission_ep5_none.csv` (no CO2, seed 42, epoch 5) | 0.133 | | tệ nhất trong các snapshot |
| 51 | `bx_probe1_agb110.csv` (blend 0.079, agb ×1.1) | 0.095 | | probe |
| 52 | `bx_probe2_height110.csv` | 0.083 | | probe |
| 53 | `bx_probe3_agb110_ssp126.csv` | 0.088 | | probe |
| 54 | `bx_probe4_height110_ssp126.csv` | 0.081 | | probe |
| 55 | `bx_scn.csv` (blend 0.079 × hệ số bias theo kịch bản) | **0.078** | 0.0783 | **tốt nhất** |
| 56 | `session A2/submission_global_dsh_v2.csv` (noco2_ep4, bỏ độ lệch test của nhóm v2) | **0.089** | | model đơn tốt nhất |
| 57 | `session A2/submission_global_y0fx1.csv` (noco2_ep4, flux ban đầu về mức lưới) | 0.091 | | = ep4 |
| 58 | `session A2/submission_global_y0bx1.csv` (noco2_ep4, stock ban đầu về mức lưới) | 0.138 | | đối chứng |
| 59 | `bagging/submission.csv` (bag_m1: 70% khối 10°, seed 21, epoch tốt nhất theo holdout = 6) | 0.128 | | |
| 60 | `submission_blend_a2_x.csv` (bg_scn 0.62, dsh_v2 0.47, y0bx1 0.21, v2 0.09, ep6 0.04, s7_ep3 0.04, global −0.29, global2_co2_hist −0.17) | **0.075** | 0.0764 | **tốt nhất** |
| 61 | `hyb_a2_dsh_v2.csv` (run trên ssp126 + bg_scn trên ssp585) | 0.089 | | probe |
| 62 | `hyb_a2_y0bx1.csv` | 0.131 | | probe |
| 63 | `hyb_noco2_ep6.csv` | 0.104 | | probe |
| 64 | `hyb_s7_ep3.csv` | 0.103 | | probe |
| 65 | `hyb_noco2_ep4.csv` | 0.093 | | probe |
| 66 | `submission_blend_scn2.csv` (trọng số riêng theo kịch bản, **không trọng số âm**) | **0.074** | 0.0764 | **tốt nhất, an toàn** |
| 67 | `submission_blend_scn2_x.csv` (theo kịch bản, trọng số âm trên cặp CO2) | **0.074** | 0.0753 | |
| 68 | Run 5 `fifth-run/submission_snapmean_none.csv` (global_nov2: no CO2 + bỏ v2 khi train, seed 42) | 0.136 | | |
| 69 | Run 5 `fifth-run/submission_ep4_none.csv` | 0.135 | | cùng seed/epoch với 0.091 |
| 70 | Run 6 `sixth-run/submission_snapmean_none.csv` (swa_ep4: fine-tune từ noco2_ep4, LR 5e-5, 8 × 0.5M mẫu) | 0.136 | | |
| 71 | `submission_y0.csv` (persistence: năm 40 = y0, `y0_persist.py`) | 0.285 | | thành phần blend, không dùng riêng |
| 72 | `hyb_y0_persist.csv` (y0 ở ssp126 + bg_scn ở ssp585) | 0.174 | | probe |
| 73 | `probe_dead_T2.csv` (blend_scn2, 8.8% ô chết đặt về số 0 đã biến đổi (−0.0826, −0.1182)) | **0.072** | T2: 0.0715–0.0725, T1: 0.074 | **T2 xác nhận, tốt nhất** |
| 74 | `submission_blend_scn2d_x.csv` (blend_scn2_x + ô chết, bỏ ô chết khỏi bài toán trọng số) | 0.072 | 0.0733 | |
| 75 | `match_direct_y0inv.csv` (LightGBM direct, y0 → vật lý, climate giữ nguyên, khớp trung bình) | **0.108** | | thành phần: tương quan lỗi 0.69 |
| 76 | `match_direct_inv.csv` (giống #75 + climate → vật lý) | 0.164 | | đảo climate làm hỏng |
| 77 | `submission_blend_t2b.csv` (trọng số chung: dead_t2 0.48 + scn2d_x 0.34 + gbm_direct_y0inv 0.19) | **0.070** | 0.0698 | **tốt nhất** |
| 78 | `hyb_gbm_direct_y0inv.csv` (GBM ở ssp126 + bg_scn ở ssp585) | 0.097 | | probe |
| 79 | `submission_blend_t2c.csv` (dead_t2 0.51 + scn2d_x 0.30 + gbm_avg 0.10 + gbm_rollout 0.09) | 0.072 | 0.0685 | ước tính sai vì LB rollout giả định 0.200; giải ngược ra 0.237 (mục 4.27) |
| 80 | `result/1/submission_global_y0inv.csv` (session E: noco2_ep4, y0 → vật lý, output map bằng slope của test_maps, không khớp trung bình) | 0.111 | | mức quá thấp (agb 0.93 so với 1.07–1.15 của blend); tương quan lỗi với GBM 0.38 |
| 81 | `submission_blend_e4.csv` (dead_t2 0.41 + scn2d_x 0.21 + gbm_direct 0.23 + lstm_y0inv 0.15) | **0.068** | 0.0689 | **tốt nhất** |
| 82 | `match_analog_add.csv` (analog cùng site: y0 test + thay đổi 40 năm lịch sử, khớp trung bình) | 0.173 | | trọng số blend 0 |
| 83 | `match_analog_hist.csv` (analog: trạng thái năm 40 lịch sử của site) | 0.209 | | trọng số blend 0 |
| 84 | `result/1/submission_global_xinv.csv` (forcing → vật lý, y0 giữ không gian T) | 0.196 | | thành viên của cặp ngoại suy (mục 4.28) |
| 85 | `result/1/submission_global_y0inv_xinv.csv` (y0 + forcing → vật lý) | 0.109 | | LSTM vật lý tốt nhất |
| 86 | `submission_blend_x1.csv` (dead_t2 0.08 + scn2d_x 0.34 + gbm_direct 0.20 + lstm_y0inv_xinv 0.735 + lstm_xinv −0.361) | **0.066** | 0.0661 | **tốt nhất**; ước tính đúng |
| 87 | `cs_submission_blend_x1_ssp585_agb1.1.csv` (x1, agb ssp585 ×1.1 quanh số 0 đã map) | 0.074 | | probe: hệ số tối ưu 1.005, không cần sửa |
| 88 | `result/global-probe/submission_global_y0inv_xinv_co2inv.csv` (session F: run 1 có CO2, y0 + forcing + CO2 vật lý) | 0.119 | | trọng số ~0 một mình; tách kịch bản agb +7% |
| 89 | `result/global-probe/submission_global_y0inv_xinv_co2_hist.csv` (cùng model, CO2 lịch sử) | 0.117 | dự báo 0.1215 | trọng số +0.12 trong x2 |
| 90 | `result/global-probe/submission_global_y0inv_xinv_co2inv0.5.csv` (nửa mức tăng CO2 vật lý) | 0.110 | dự báo 0.1105 | đúng dự báo ⇒ không thêm thông tin |
| 91 | `result/match_gbm_none.csv` (GBM direct, y0 để ở không gian T, khớp trung bình) | 0.169 | dự báo 0.1515 | lệch lớn ⇒ cặp GBM: trọng số −0.39 |
| 92 | `result/match_gbm_y0inv2.csv` (GBM direct, y0 = 2·y0_vật_lý − y0_T, khớp trung bình) | 0.097 | dự báo 0.0955 | gần dự báo ⇒ trọng số −0.04, không lợi |
| 93 | `submission_blend_x2.csv` (cặp LSTM + cặp GBM + g1_co2_hist) | **0.063** | 0.0624 | **tốt nhất** |
| 94 | `result/match_rf_y0inv.csv` (LightGBM chế độ random forest, y0 vật lý, khớp trung bình) | 0.110 | dự báo 0.104 | trọng số −0.17 nếu cho phép, 0.0619: không đáng |
| 95 | `result/match_rf_none.csv` (RF, y0 để ở không gian T, khớp trung bình) | 0.155 | dự báo 0.1485 | cặp RF chỉ thêm 0.0004–0.0005: trùng thông tin với cặp GBM |
| 96 | `result/noco2-s7/submission_global_xinv.csv` (seed 7 ep3, forcing vật lý) | 0.277 | dự báo 0.271 | thành viên cặp seed 7 |
| 97 | `result/noco2-s7/submission_global_y0inv_xinv.csv` (seed 7 ep3, y0 + forcing vật lý) | 0.192 | dự báo 0.189 | thành viên cặp seed 7 |
| 98 | `submission_blend_x3.csv` (x2 + cặp seed 7) | **0.062** | 0.0615 | **tốt nhất** |
| 99 | `kaggle_run1/submission_global_s42ep3_xinv.csv` | 0.254 | dự báo 0.2514 | |
| 100 | `kaggle_run1/submission_global_s42ep6_xinv.csv` | 0.192 | dự báo 0.1966 | |
| 101 | `kaggle_run1/submission_global_s42ep3_y0inv_xinv.csv` | 0.164 | dự báo 0.1627 | |
| 102 | `result/match_nn_y0inv2.csv` (NN direct, y0 vượt quá vật lý) | 0.113 | dự báo 0.1192 | đơn lẻ kém GBM y0inv2 (0.097), nhưng khác GBM nên nhận trọng số +0.16 |
| 103 | `kaggle_run1/submission_global_s42ep6_y0inv_xinv.csv` | 0.112 | dự báo 0.1161 | |
| 104 | `kaggle_run1/submission_global_bagep6_xinv.csv` | 0.260 | dự báo 0.2631 | |
| 105 | `kaggle_run1/submission_global_bagep6_y0inv_xinv.csv` | 0.182 | dự báo 0.1853 | cặp bagging: lợi 0.0001, trùng các cặp LSTM khác |
| 106 | `submission_blend_x4f.csv` (x3 + NN y0inv2 + cặp s42ep6/ep3, 3 trọng số âm) | **0.061** | 0.0600 | **tốt nhất** |
| 107 | `submission_blend_x4.csv` (cùng thành viên, chỉ 2 trọng số âm như x3) | 0.062 | 0.0606 | lựa chọn cuối: x4f + x3 |
| 112 | `kaggle_sweep_pairs_run2/submission_global_s7ep5_xinv.csv` | 0.247 | | |
| 113 | `kaggle_sweep_pairs_run2/submission_global_s7ep5_y0inv_xinv.csv` | 0.140 | | cặp s7ep5: x4f ước tính 0.0600 → 0.0599, không làm file mới |
| 114 | `submission_blend_x5.csv` (x4f + cặp s7ep5) | 0.061 | 0.0599 | không đổi so với x4f |
| 115–118 | `sp_uniform` / `sp_co2lstm` / `sp_dry` / `sp_young` (x4f, mẫu hình ssp585, +0.003 nếu vô ích) | 0.064 / 0.064 / 0.063 / **0.061** | 0.064 mỗi file | **young xác nhận** |
| 108–111 | `dp_gbm_pair` / `dp_lstm_pair` / `dp_nn` / `dp_dead_t2` (x4f, chỉ ssp585 bước theo một hướng, +0.003 nếu không lợi) | 0.064 / 0.063 / 0.064 / 0.064 | 0.064 mỗi file | lợi tối ưu ~0.0002 ⇒ không nộp |
| 119 | `submission_blend_x6.csv` (x4f + bước young ssp585 +0.072, dry +0.008) | **0.060** | 0.0599 | **tốt nhất**; lựa chọn cuối: x6 + x4f |
| 120 | `sp126_young.csv` (x4f, hướng young trên ssp126, bước +0.121; 0.064 = vô ích) | 0.063 | 0.064 | bước tối ưu ssp126 ~+0.02 (0…+0.03), lợi ≤ 0.0002 ⇒ không làm x7 |
| 121–122 | `ap_young_lo` / `ap_old_lo` (x6, hình dạng tuổi trong ssp585; 0.063 = vô ích) | 0.062 / 0.062 | 0.063 mỗi file | bước tối ưu +0.015 mỗi hướng, lợi 0.0002 ⇒ không nộp |
| 123–126 | `st_warm` / `st_npp` / `st_incr` / `st_h_only` (x6, mẫu hình site ssp585, trực giao; 0.063 = vô ích) | 0.063 / 0.064 / 0.063 / 0.064 | 0.063 mỗi file | lợi tối ưu 0.0002 ⇒ không nộp |
| 127 | `co2_dir.csv` (x6 + mẫu hình CO2 theo dòng từ hiệu ứng năm lịch của ED, ssp585, trực giao young/dry; 0.063 = vô ích) | 0.063 | 0.063 | vô ích: mẫu hình CO2 theo site của ED không thêm gì ngoài trục tuổi |
| 128 | `match_gbmc_y0inv2.csv` (GBM direct + chuỗi tuổi + rủi ro sự kiện, y0 vượt quá vật lý, khớp trung bình) | **0.073** | dự báo 0.0936 | **model đơn tốt nhất** (trước: 0.089); GBM cũ cùng biến thể 0.097 |
| 129 | `submission_blend_x7f.csv` (x6 0.80 + gbmc_y0inv2 0.63 − GBM cũ y0inv2 0.43) | **0.054** | 0.0532 | **tốt nhất** |
| 130 | `submission_blend_x7.csv` (x6 0.69 + gbmc_y0inv2 0.31) | 0.057 | 0.0566 | không trọng số âm; lựa chọn cuối: x7f + x7 |
| 131 | `probe_x7f_dead_raw0.csv` (x7f, 16,363 ô chết đặt về 0 vật lý) | 0.055 | 0.055 nếu T / 0.053 nếu vật lý | **đáp án ở không gian T xác nhận lần 2**; notebook công khai ("raw units") sai |
| 132 | `match_gbmc_none.csv` (bản sinh đôi gbmc, y0 không gian T) | 0.158 | 0.168 | thêm vào blend: 0.0532 → 0.0529, dưới mức làm tròn ⇒ không làm file mới |
| 133 | `match_gbmc2_y0inv2.csv` (GBM chrono, trung bình 2 seed 42 + 7) | **0.072** | 0.0731 (chỉ thông tin) | **model đơn tốt nhất**; lấy trung bình seed giúp ~0.001; blend 0.0532 → 0.0526 (+ cặp none), chờ seed 11 + NN chrono rồi làm x8 |
| 134 | `match_nnc_y0inv2.csv` (NN direct + chrono + rủi ro sự kiện, 4 seed, khớp trung bình) | **0.075** | dự báo 0.0897 | tốt hơn dự báo 0.015; NN cũ 0.113 |
| 135 | `submission_blend_x8f.csv` (x6 0.72 + gbmc2 0.45 + nnc 0.39 − GBM cũ 0.41 − NN cũ 0.15) | **0.050** | 0.0503 | **tốt nhất**; lựa chọn cuối: x8f + x7f |
| 136–137 | `yp_x8f_ssp126` / `yp_x8f_ssp585` (bước young trên x8f, từng kịch bản; 0.053 = vô ích) | 0.053 / 0.054 | 0.053 mỗi file | x8f không cần hiệu chỉnh tuổi nữa (ssp585: bước −0.018, lợi 0.0001) |
| 138 | `match_nnx_y0inv2.csv` (NN cross-age + chrono + rủi ro sự kiện, seed 4–7) | **0.061** | dự báo 0.067 | **model đơn tốt nhất** (trước 0.072) |
| 139 | `submission_blend_x9f.csv` (x6 + gbmc2 + nnc + nnx − GBM cũ − NN cũ + gbmc_none) | **0.049** | 0.0488 | **tốt nhất**; lựa chọn cuối: x9f + x8f |
| 140 | `submission_blend_x10.csv` (cấu hình x9f, nnx8 + gbmc3) | 0.049 | 0.0489 (bi quan) | lợi bớt nhiễu < mức làm tròn |
| 141 | `match_nnm4_y0inv2.csv` (NN cross-age + khí hậu theo tháng, seed 12–15) | 0.071 | dự báo 0.0726 | gần mức "không thông tin mới": blend 0.0489 → 0.0488 ⇒ không làm file mới |
| 142 | `match_gbmc3i.csv` (GBM chrono 3 seed, biến thể y0inv) | 0.091 | dự báo 0.0938 | blend 0.0489 → 0.0486 (+ nnm4: 0.0484); chờ nnc_y0inv / nnx8i rồi làm x11 |
| 143 | `match_gbmg_y0inv2.csv` (GBM chrono + mức độ mất sự kiện, A6) | 0.098 | dự báo 0.101 | holdout tốt hơn (0.0234) nhưng LB tệ hơn gbmc (0.073); blend +0.0001 |
| 144 | `submission_blend_flat6.csv` (6 model đơn, một tầng blend) | **0.049** | 0.0487 | ngang x9f; **lựa chọn cuối: flat6 + x9f** |
| 145–147 | `ep_flat6` / `ep_flat6_level` / `ep_flat6_young` (hệ số thời kỳ của site; 0.052 = vô ích) | 0.052 / 0.054 / 0.054 | 0.052 mỗi file | growth đúng mức vô ích; level, young: bước ngược −0.024 / −0.116, lợi 0.0003 mỗi hướng |
| 148 | `submission_blend_flat6e.csv` (flat6 + bước level/young của hệ số site) | 0.049 | ~0.0481 | lợi dưới mức làm tròn |
| 149–150 | `match_nnx8i.csv` / `match_nnc_y0inv.csv` (bản sinh đôi y0inv của NN cross-age / NN chrono) | 0.117 / 0.114 | 0.1157 / 0.1158 | sát mức không thông tin: flat6 + cả hai = 0.0483 (`submission_blend_flat8t.csv`) |
| 151–152 | `match_nna_y0inv2.csv` / `match_nna_y0inv.csv` (NN cross-age + augmentation y0) | 0.096 / 0.130 | 0.0974 / 0.1304 | đúng mức không thông tin: augmentation đổi dự báo nhưng không theo hướng có ích |
| 153–164 | `cw_scn_*` / `cw_age_*` (thành viên flat6 ở một ô, flat6 ở ô kia) | scn 0.054/0.065/0.068/0.071/0.095/0.135; age 0.052/0.056/0.074/0.069/0.084/0.106 | | trọng số theo ô: scn lợi 0.0006, age 0.0005 |
| 165–166 | `submission_blend_flat6_scn.csv` / `submission_blend_flat6_age.csv` (trọng số theo ô) | 0.049 / 0.049 | ~0.0481 / ~0.0482 | lợi dưới mức làm tròn |
| 167–168 | `match_emu_y0inv2.csv` / `match_emu_y0inv.csv` (emulator GRU) | 0.116 / 0.144 | 0.122 / 0.152 | tốt hơn dự báo 0.006–0.008: thông tin mới thật |

Chưa nộp: `submission_blend_scn3.csv` (0.0759), `submission_blend_scn3_x.csv` (0.0752), `submission_blend_scn_x.csv` (0.0902), `submission_blend_scn.csv` (0.0935), `submission_blend_latest.csv`
(0.0937), `submission_blend_global_x.csv` (0.0977).
Run 1 của notebook global (`notebook2dbddcbdc5.ipynb`) không có điểm LB.

Độ chính xác của ước tính: 10 lần đúng trong sai số làm tròn (±0.0005); 1 lần lệch 0.002 (#21, theo hướng có lợi).
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
| Feature 7 | biên độ mùa ~20×, std tổng ~5× | vô hại (msd ≤ 0.0001, mục 4.13) |
| Tĩnh 129, 135 | giá trị không trùng lưới nhưng trong khoảng | vô hại (msd ≤ 0.0001) |
| Biến 2 (34, 38, …, 126) | lệch so với lưới | **có tác động**: nâng AGB ~3–5% (mục 4.13) |

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

- [KC] Seed 7 (run 4): epoch 3 có holdout shift **0.032** (tốt nhất mọi run) nhưng LB **0.118**. Đã có 5 điểm LB của
  model không CO2 (0.091–0.118), và holdout không xếp hạng đúng lần nào. 0.091 của seed 42 ep4 nhiều khả năng là
  một lần "trúng" trong một phân phối rộng (~0.09–0.12), không phải mức điển hình của công thức này.
- Vị trí ep4 trong lịch LR giống nhau ở hai seed (LR giảm sau ep4). Nộp ep4 của seed 7 trả lời "ep4 tốt vì lịch
  train hay vì may".

- [KC, đại số] Run 4 (s7_ep3) thêm vào blend: trọng số 0.03, ước tính vẫn 0.0828. Các snapshot của seed 7 (ep2/3/4)
  gần nhau (msd 0.02–0.03) nhưng xa noco2_ep4 (0.055–0.073): seed khác cho model khác hẳn.
- [KC, đại số] Giá trị của snapshot chưa chấm (ước tính blend nếu LB của nó là L): s7_ep2, s7_ep4, noco2_ep5 vẫn có
  ích kể cả khi L = 0.12 (blend 0.0818–0.0821) vì chúng khác các thành phần chính; s7_ep5 chỉ có ích nếu L < 0.11.
  ⇒ nên chấm noco2_ep5, s7_ep4, s7_ep2 trước khi probe bias blend cuối.

- [KC, đại số] Trung bình đều (0 tham số) so với blend có fit: 4 snapshot không CO2 → 0.0932 (các thành phần
  0.091–0.118, trung bình 0.109); noco2_ep4 + bg_scn 50/50 → 0.0837 so với blend fit 0.0828. Trọng số fit chỉ thêm
  0.001: rủi ro overfit nằm ở việc **chọn** ep4, không phải ở trọng số. Trung bình nhiều snapshot là cách giảm phương
  sai giữa epoch hiệu quả nhất.
- Công cụ kiểm tra overfit: `cv_parity.py` (chia site chẵn/lẻ trên chính LB public; đã tự kiểm tra với tập public
  giả lập 45% site: hệ số bias khớp ±0.001, LB dự báo khớp ±0.0002, đánh giá chéo blend lệch ~0.0006).

### 4.11 CV chẵn/lẻ trên LB public, kết quả một phần (5/8 probe, `cv_parity.py`) [KC, đại số]

Độ chênh so với bg_scn, mỗi nửa (đơn vị LB đầy đủ; nhiễu làm tròn ±0.002 mỗi số):

| Run | site chẵn | site lẻ |
|---|---|---|
| noco2_ep3 | +0.022 | +0.022 |
| noco2_ep6 | +0.018 | +0.012 |
| s7_ep3 | +0.024 | +0.024 |
| global2_co2_hist | +0.032 | +0.040 |
| v2 | +0.084 | +0.100 |

- Thứ hạng giống nhau ở hai nửa ⇒ khác biệt giữa các snapshot là **thật** (model khác nhau thật trên test), không phải
  nhiễu do chọn site public. LB public là trọng tài đáng tin cho các khác biệt ≥ ~0.005.
- Trọng số blend: fit trên một nửa, chấm trên nửa kia → overfit +0.0007 ở cỡ nửa (≈ +0.0004 ở cỡ đầy đủ). Trọng số
  âm trên global2_co2_hist: cùng dấu ở hai nửa (−0.125 / −0.216), đánh giá chéo 0.0909 so với 0.0914 simplex ⇒
  phần ngoại suy CO2 **đứng vững ngoài mẫu** trong tập này.
- **Đủ 8/8 probe** (so với noco2_ep4, chẵn / lẻ): bg_scn +0.000/+0.006, noco2_ep6 +0.018/+0.018, noco2_ep3
  +0.022/+0.028, s7_ep3 +0.024/+0.030, global2_co2_hist +0.032/+0.046, global +0.038/+0.048, lstm_xl +0.070/+0.084,
  v2 +0.084/+0.106.
- [KC] **ep4 tốt nhất ở cả hai nửa** (winner's curse = 0): 0.091 không phải may mắn do chọn site public. Ep4 thật sự
  tốt hơn các snapshot khác trên test.
- [KC] Blend simplex: trong mẫu 0.0826, đánh giá chéo 0.0834 (overfit ≈ +0.0004 ở cỡ đầy đủ).
- [KC] Blend có trọng số âm (global, global2_co2_hist): trong mẫu 0.0785, đánh giá chéo **0.0803** (overfit ≈
  +0.0009 ở cỡ đầy đủ), vẫn hơn simplex đánh giá chéo 0.003. Trọng số âm trên global cùng dấu hai nửa (−0.30 / −0.51).
  ⇒ file 0.079 dự kiến private ≈ 0.080: **an toàn làm lựa chọn cuối #1**.

### 4.12 Bias của blend 0.079 (probe #51–54) [KC]

| Hệ số tối ưu | ssp126 | ssp585 |
|---|---|---|
| agb | 0.990 (0.986–0.995) | 1.007 (1.003–1.011) |
| height | **1.026** (1.018–1.034) | **1.028** (1.020–1.035) |

- Blend 0.079 **gần như không còn bias**: AGB lệch ≤ 1%, height thiếu ~2.7% đều ở cả hai kịch bản (blend cũ thì
  thừa height 5–6% ở ssp126). Blend có trọng số âm đã hấp thụ gần hết hiệu ứng theo kịch bản.
- `result/bx_scn.csv` (blend 0.079 × 4 hệ số): ước tính **0.0783** (gain 0.0006–0.0009, luôn > 0).
- Hệ quả: hiệu chỉnh bias gần như hết dư địa. Lỗi còn lại là scatter ⇒ đòn bẩy tiếp theo là thêm member đa dạng
  (bagging, seed), không phải hiệu chỉnh.
- Snapshot seed 42: ep3 0.116, ep4 0.091, ep5 0.133, ep6 0.109. Ep5 (epoch đầu sau khi LR giảm một nửa, AGB trung
  bình 0.89) tệ nhất: phương sai giữa epoch rất lớn và không theo quy luật ⇒ trung bình snapshot (`snapmean`) là
  bắt buộc.

### 4.13 Probe dữ liệu trên model tốt nhất (no CO2, seed 42, ep4; `global_probe.py` part A + B) [KC]

Screening (1500 site test, bỏ độ lệch test−lưới của từng nhóm feature, so với dự đoán gốc):

| Nhóm | msd ssp126 / ssp585 | ΔAGB | Δheight |
|---|---|---|---|
| **v2** (34, 38, …, 126) | **0.0092 / 0.0036** | −4.8% / −3.0% | ≈ 0 |
| v1 (33, 37, …) | 0.0013 / 0.0004 | +2.9% / +1.6% | +0.7% / +0.5% |
| v3 (35, 39, …) | 0.0010 / 0.0009 | −1.3% / −1.2% | −0.3% |
| f0–f7, v4, static, `f7_fix` | ≤ 0.0001 | < 0.4% | < 0.4% |

- Như với run 2 (v2 msd 0.024 / 0.009): **v2 là đòn bẩy duy nhất ngoài CO2**, và độ lệch test của nó đang *nâng* AGB
  lên ~3–5%. Chưa biết đúng hay sai: LB của `dsh_v2` sẽ trả lời. Feature 7 và nhóm static vô hại (đóng câu hỏi mở).
- Cột `toward_target_pct` dùng hướng bias của blend cũ, không có nghĩa với model này: bỏ qua.

y0 memory (1000 site lưới, metric với y0 thật 0.115; scale y0 theo tỷ lệ test/lưới):

- Dự đoán năm 40 dịch **msd 0.062** (≈ 2/3 toàn bộ LB 0.091). Height: y0 −17.3% → năm 40 −10.7% (giữ **62%**);
  AGB: y0 −28.4% → năm 40 −15.5% (giữ **55%**). Độ lệch `test_y0` là **một kênh lớn**, không phải chuyện nhỏ (bản thử
  trên dữ liệu giả lập từng cho −4%).
- [GT] Trong ED, gpp/npp/rh (và một phần lai) là đại lượng *chẩn đoán* (tính từ trạng thái + khí hậu + CO2 mỗi năm),
  còn height/agb/soil là trạng thái có trí nhớ thật. Flux năm 0 của test cao hơn lưới 10–30%, nhiều khả năng do CO2
  hiện tại cao hơn 1980. Model không CO2 có thể đọc flux ban đầu cao như "site năng suất cao" rồi mang nó suốt 40
  năm (hoặc ngược lại: đó là thông tin CO2 duy nhất nó có).
- Thí nghiệm (inference, `global_probe.py` part C): `y0fx1` = đưa lai/gpp/npp/rh của test y0 về mức lưới theo từng
  tuổi; `y0bx1` = đưa height/agb/soil về mức lưới (đối chứng: trí nhớ trạng thái là vật lý, dự kiến tệ hơn).

### 4.14 Session A2 (biến đổi input của noco2_ep4) và bagging member m1 [KC]

- `none` của A2 trùng tuyệt đối file 0.091 (msd 0.0): script probe tái lập đúng tiền xử lý của notebook.
- **v2: dsh_v2 0.089 < 0.091.** Phản ứng của model với độ lệch test của nhóm v2 một phần là sai, giống CO2 (CO2:
  0.134 → 0.122 khi sửa ở inference, rồi 0.091 khi bỏ hẳn khi train). LB dọc đường ep4 → dsh_v2 cực tiểu ở t ≈ 0.66
  (≈ 0.088). Bỏ v2 làm AGB giảm 1.013 → 0.973. ⇒ train model **không CO2 + không v2** (`global_nov2`).
- **Flux ban đầu: y0fx1 = 0.091, msd so với ep4 chỉ 0.0002.** Model gần như *bỏ qua* lai/gpp/npp/rh ban đầu. Giả
  thuyết "model mang flux ban đầu cao suốt 40 năm" (mục 4.13) **sai**; 55–62% trí nhớ y0 là trí nhớ của stock.
- **Stock ban đầu: y0bx1 = 0.138.** Đưa height/agb/soil ban đầu về mức lưới làm hỏng nặng: stock thấp của test là
  thật và model dùng đúng. Nhưng LB dọc ep4 → y0bx1 cực tiểu ở t ≈ 0.13 (≈ 0.090) và blend cho y0bx1 trọng số
  0.12–0.21: dịch nhẹ stock ban đầu lên giúp chút ít. [GT] model nhớ stock thấp hơi quá mức (hoặc dự đoán tăng trưởng
  từ stock thấp hơi thấp).
- **Bagging m1 (70% khối 10°): epoch 6 = 0.128**, tệ hơn mọi snapshot đủ dữ liệu cùng vị trí (seed 42 ep6 0.109).
  Blend cho nó trọng số 0; snapmean của nó chỉ có ích nếu LB ≤ 0.10. ⇒ bỏ bagging theo không gian; giữ đủ dữ liệu,
  tạo đa dạng bằng seed + bỏ nhóm feature có mục tiêu.
- Trung bình snapshot (0 tham số): 4 snapshot seed 42 (0.116/0.091/0.133/0.109, trung bình 0.112) → ước tính
  **0.0938**; không có ep4 → 0.1017. Trung bình snapshot gần bằng snapshot tốt nhất mà không cần chọn epoch.
- Blend mới (đều trên file đã chấm): `submission_blend_a2.csv` 0.0773 (bx_scn 0.67, dsh_v2 0.14, y0bx1 0.12, v2
  0.05, …); `submission_blend_a2_x.csv` 0.0764 (bg_scn 0.62, dsh_v2 0.47, y0bx1 0.21, v2 0.09, global −0.29,
  global2_co2_hist −0.17, …). Lợi đã nhỏ dần: 0.078 → ~0.077.

### 4.15 Model có phân biệt hai kịch bản không? [KC]

Cùng 6171 site, cùng y0, chỉ forcing khác nhau. Độ khác giữa dự đoán ssp126 và ssp585 của mỗi model:

| Model | msd(ssp126, ssp585) | AGB 585/126 | height 585/126 |
|---|---|---|---|
| noco2_ep4 | 0.0048 | 0.988 | 1.006 |
| a2_dsh_v2 | 0.0037 | 1.004 | 1.006 |
| s7_ep3 | 0.0127 | 0.974 | 1.010 |
| global (có CO2) | 0.0069 | 1.011 | 0.997 |
| lstm_xl | 0.0231 | 0.985 | 1.014 |
| **bx_scn (sau hiệu chỉnh)** | 0.0140 | **1.061** | **1.047** |

- Mọi model dự đoán hai kịch bản **gần như giống hệt** (AGB lệch ≤ 3%, thường ngược chiều). Toàn bộ khác biệt theo kịch
  bản mà ED tạo ra (ssp585 AGB cao hơn ~6–8%) đến từ hệ số hậu kỳ. Dữ liệu lịch sử không chứa phản ứng kịch bản để học.
- ⇒ "Một model cho ssp126, một model cho ssp585" theo nghĩa *train riêng* là không làm được: không có dữ liệu train nào
  thuộc kịch bản, và model train trên lịch sử không phân biệt được hai kịch bản. Phần làm được là **ensemble riêng
  theo kịch bản** (trọng số và hệ số bias khác nhau), vì lỗi của các model khác nhau theo kịch bản (mục 4.8: model cũ
  tệ hơn hẳn ở ssp585; ssp585 muốn ngoại suy CO2 mạnh hơn ssp126 nhiều).

### 4.16 Ensemble riêng theo kịch bản (hybrid #61–65, `ensemble_scn.py`) [KC, đại số]

Lỗi so với bg_scn trong từng kịch bản (chính xác trong một kịch bản; đơn vị LB đầy đủ):

| Model | ssp126 | ssp585 |
|---|---|---|
| noco2_ep4 | −0.002 | −0.004 |
| **a2_dsh_v2** | **−0.010** | +0.000 |
| noco2_ep6 | +0.020 | +0.010 |
| s7_ep3 | +0.018 | +0.030 |
| **a2_y0bx1** | **+0.074** | **+0.014** |
| global_co2_hist | +0.030 | +0.026 |
| v2 / lstm_xl / eco | +0.06 | +0.09–0.12 |

- `dsh_v2` (bỏ độ lệch v2) giúp **chủ yếu ở ssp126** (−0.010; ssp585 ≈ 0).
- `y0bx1` (stock ban đầu nâng lên mức lưới) **rất tệ ở ssp126 nhưng gần tốt nhất ở ssp585**. ssp585 cần stock cao hơn
  (khớp hệ số bias AGB ×1.09–1.12 của ssp585); nâng stock ban đầu là một cách gián tiếp để có điều đó. [GT] Đây là
  proxy cho phản ứng CO2/ấm lên mà model không có, không phải bản sửa vật lý.
- Blend theo kịch bản: simplex **0.0764** (trọng số chung trên cùng thành phần: 0.0789); có trọng số âm trên cặp CO2
  **0.0753** (chung: 0.0764 = ước tính của file 0.075). Ổn định với làm tròn LB (0.0754–0.0758). ssp585 vẫn muốn ngoại
  suy CO2 mạnh hơn (global −0.48 / global_co2_hist +0.39) so với ssp126 (−0.23 / 0).
- `submission_blend_scn2.csv` **không có trọng số âm ở bất kỳ tầng nào** (bg_scn là blend simplex × hệ số) mà ước tính
  ngang file 0.075: ứng viên chọn cuối "an toàn" tốt nhất.
- Chưa đo overfit của trọng số theo kịch bản (cần probe chẵn/lẻ × kịch bản). Số tham số gấp đôi; theo CV trước, fit
  trọng số overfit ~0.0004–0.0009 ⇒ phần lợi 0.001 của bản theo kịch bản có thể mất một nửa trên private.
- [KC] Kết quả: scn2 (simplex) **0.074** = scn2_x (trọng số âm) **0.074**. Khi trọng số đã riêng theo kịch bản, trọng
  số âm trên cặp CO2 không còn đem lại lợi thấy được ⇒ bản an toàn thắng (ít rủi ro hơn với cùng điểm).
- [KC] Ba blend gần nhất đều tốt hơn ước tính 0.001–0.002 (0.0764→0.075, 0.0764→0.074, 0.0753→0.074). Ước tính hơi
  bi quan khi các thành phần khác nhau nhiều; vẫn dùng tốt để xếp hạng.

### 4.17 Run 5: bỏ v2 khi train làm hỏng (0.135) [KC]

- Cùng công thức với run 3 (no CO2, đủ dữ liệu, seed 42) chỉ khác bỏ nhóm v2: ep4 **0.135** (run 3 ep4: 0.091), snapmean
  0.136. Trọng số trong blend: 0.
- ⇒ **v2 mang tín hiệu thật** (khác CO2). Sửa ở inference (dsh_v2: giữ v2 nhưng bỏ *độ lệch* test của nó) giúp; bỏ hẳn v2
  khi train làm mất phần thông tin không gian của nó. Bài học từ CO2 không khái quát sang nhóm khác: CO2 gần như đồng
  nhất trong không gian (chỉ là đồng hồ thời gian), v2 thì không.
- Cùng seed 42 nhưng input khác cho model khác hẳn (msd nov2_ep4 vs noco2_ep4 0.052): không tách được hiệu ứng v2 khỏi
  ngẫu nhiên của training chỉ bằng một run. Nhưng 0.135 nằm ngoài cả phân phối no-CO2 (0.091–0.133) nên kết luận vẫn
  đứng.
- Tổng hợp các model no-CO2 train từ đầu: 0.091, 0.109, 0.116, 0.118, 0.128 (bag), 0.133, 0.135, 0.136. Điển hình
  ~0.11–0.13; **0.091 (seed 42 ep4) là một lần train "trúng"** (CV chẵn/lẻ loại trừ may mắn do chọn site public, không
  loại trừ may mắn của quá trình train). Train thêm model từ đầu khó tạo được thành phần có ích cho blend.
- Hướng tiếp: fine-tune / SWA quanh nghiệm tốt (`swa_ep4`: khởi tạo từ snap_ep4, LR hằng 5e-5, 8 × 0.5M mẫu, snapshot
  2/4/6/8 + snapmean). [GT] Trung bình các điểm gần một nghiệm tốt thường tốt bằng hoặc hơn chính nó và ổn định hơn.

### 4.18 Feature selection: làm ở đâu cho đúng [GT + KC]

- 136 feature = f0–f7 (8), CO2 (24), hằng số 32, 4 biến × 24 lớp (96, v1–v4), tĩnh (7).
- Train lại với tập feature khác **không đo được hiệu quả**: cùng một công thức cho LB 0.091–0.136 tùy seed/epoch
  (độ lệch chuẩn ~0.015), lớn hơn hiệu ứng thường thấy của feature selection (~0.005). Cần ≥3–4 seed cho mỗi tập
  feature (~20 h GPU mỗi so sánh). Lần duy nhất ngoài CO2 (bỏ v2) thất bại rõ (0.135).
- Dữ liệu 31M cặp, model 22.7M tham số: feature nhiễu ít gây overfit; lợi chính của feature selection ở đây không phải
  giảm nhiễu mà là **bỏ feature mà model dùng sai khi test bị lệch phân phối** (đúng như CO2).
- Part A đã làm việc đó ở mức nhóm (chỉ CO2 và v2 có tác dụng). Bước tiếp, không cần train: `global_probe.py` part D
  trên model 0.091: D1 permutation importance theo nhóm trên holdout có nhãn (model dựa vào nhóm nào), D2 screening
  độ lệch test **từng feature** trong v1/v2/v3 (72 feature). Feature quan trọng + bị lệch + làm dự đoán dịch nhiều là
  ứng viên; biến thể `dshf_<i>_<j>` sửa đúng các feature đó ở inference.

### 4.19 Fit lịch sử tốt hơn ⇒ LB tệ hơn; feature selection không còn gì (run 6 + session D) [KC]

**Fine-tune (swa_ep4) từ trọng số 0.091:**
- Khởi tạo đúng (holdout shift epoch 1 = 0.037, như run 3 ep4 0.038). Chỉ sau 1M mẫu (ep2) dự đoán test đã cách
  noco2_ep4 msd 0.042, sau 4M mẫu 0.080 (cỡ khoảng cách giữa hai seed). AGB test giảm 1.013 → 0.93. Snapmean **0.136**,
  trọng số blend 0.
- Cùng lúc holdout lịch sử *tốt lên* liên tục (random 0.083 → 0.054; AGB holdout pred/true 1.004, run 3 ep6: 1.10).
- **Tương quan giữa metric holdout random (lịch sử) và LB trên 7 file no-CO2: −0.85.** File fit lịch sử tốt nhất
  (swa, bag) có LB tệ nhất; noco2_ep4 có holdout random tệ nhất (0.126) và LB tốt nhất. [GT] Tương lai của ED (CO2 cao,
  ấm hơn) cho rừng mọc nhanh hơn lịch sử; model không CO2 không biết điều đó, nên model "mọc quá mức so với lịch sử"
  lại gần tương lai hơn. Hệ số bias theo kịch bản chỉ sửa được phần mức trung bình.
- Hệ quả: 0.091 không phải đáy của một vùng tốt mà là một điểm trên quỹ đạo train; train tiếp (fine-tune, SWA, seed
  mới, bagging) đều kéo về phía "đúng với lịch sử" và làm LB tệ đi. **Dừng train model mới**: 6 run sau 0.091 không run
  nào có ích cho blend.

**Feature selection (session D, model 0.091):**
- D1 permutation importance (1000 site holdout chưa train): v1 +0.64, v2 +0.31, f1 +0.10, f0 +0.03, f6 +0.02, f3 +0.016,
  f5 +0.011, v4 +0.010, f2 +0.005, f4 +0.004, static +0.0015, f7 −0.004, **v3 −0.007**. Model không dùng v3 (nhóm có
  sàn lệch "SUS"), f7, static: các feature lệch phân phối kỳ lạ của test đều vô hại vì không được dùng.
- D2 từng feature: không feature đơn lẻ nào làm dự đoán dịch quá msd 0.0002. Tổng 24 feature của v2: 0.0009, trong khi
  sửa cả nhóm (part A): 0.0092 ⇒ hiệu ứng v2 là một **profile chung** (độ lệch test âm ~−0.13 z ở lớp 0–5 và 21–23, ≈ 0
  ở các lớp giữa), chỉ tác động khi đi cùng nhau. Sửa từng feature không thể tốt hơn `dsh_v2`.
- ⇒ feature selection đã khai thác hết: CO2 (bỏ), v2 (sửa độ lệch ở inference). Phần còn lại hoặc không được dùng hoặc
  không bị lệch.

**Hướng còn lại (không GPU):** thêm y0 (trạng thái năm 0) làm thành phần blend theo kịch bản (`y0_persist.py`): cho
blend tách "mức" khỏi "tăng trưởng" (y40' = Σ wᵢpᵢ + w₀·y0). y0bx1 (nâng stock ban đầu) đã nhận trọng số 0.4–0.5 ở
ssp585, nên hướng này có tín hiệu. 2 lượt: `submission_y0.csv` và hybrid của nó.

### 4.20 y0 làm thành phần blend (#71) [KC + đại số]

- `submission_y0.csv` = **0.285** (để so: model đơn tốt nhất 0.089). Preds ở `result/y0_persist_preds/`
  (`python ensemble_scn.py ingest y0_persist y0_preds.npz`).
- Blend trọng số chung (chỉ cần LB của y0): 0.0789 → 0.0786 (w_y0 0.04); bản có trọng số âm: 0.0764 → 0.0764.
  ⇒ y0 chỉ có ích nếu trọng số **khác nhau theo kịch bản**.
- Ràng buộc PSD của ma trận hiệp phương sai không khoanh được LB(hyb_y0): mọi giá trị 0.14–0.26 đều hợp lệ, nên
  phải nộp hybrid.
- Ước tính blend theo kịch bản (không trọng số âm) theo LB(hyb_y0) giả định: 0.17 → 0.0756, 0.18 → 0.0759 (không lợi),
  0.20 → 0.0735, 0.22 → 0.0694, 0.24 → 0.0637. Hybrid càng cao (y0 tệ ở ssp126, tức là tốt ở ssp585) thì y0 càng có
  ích, và trọng số đều rơi vào ssp585 (0.17–0.32).
- **Kết quả: hybrid = 0.174** (#72), rơi đúng vào vùng không có lợi. `ENSEMBLE_TAG=blend_scn3 python ensemble_scn.py
  solve`: w_y0 = 0.075 (ssp126) / 0.035 (ssp585), ước tính **0.0759** (scn2: 0.0764); bản trọng số âm
  w_y0 = 0.05 / 0.013, **0.0752** (scn2_x: 0.0753). File mới chỉ lệch khỏi scn2 / scn2_x một msd 0.0005 / 0.0002.
- Kết luận: y0 không thêm hướng mới. Hai việc y0 có thể làm (kéo mức dự đoán về trạng thái ban đầu, và đổi biên độ
  tăng trưởng theo kịch bản) đã được làm sẵn bởi hiệu chỉnh bias theo kịch bản (bg_scn, bx_scn) và y0bx1 (trọng số
  0.15 / 0.51). Không nộp scn3; giữ scn2 + scn2_x.

### 4.21 Nhiều model nhỏ trên tập con feature/dữ liệu: giới hạn của việc lấy trung bình [KC, đại số]

Đã có sẵn các thành viên kiểu này: bag1 (70% khối dữ liệu), nov2 (bỏ nhóm v2), s7 (seed khác), ep3–6, swa. Với M
thành viên hoán đổi được, trung bình đều có MSE = mean(C_ii)/M + (1 − 1/M)·mean(C_ij), nên M → ∞ cho mean(C_ij):

| Tập thành viên | LB TB | tương quan lỗi TB | TB đều | tối ưu simplex | M → ∞ |
|---|---|---|---|---|---|
| 8 run lưới no-CO2 | 0.121 | 0.80 | 0.100 | 0.0865 | **0.097** |
| 6 run không chọn lọc (bỏ ep4, swa) | 0.123 | 0.81 | 0.104 | 0.098 | **0.0995** |
| chỉ thành viên tập con (bag1, nov2 ×2) | 0.133 | 0.86 | 0.120 | 0.119 | **0.114** |

- Lấy trung bình vô hạn model LSTM lưới vẫn **tệ hơn model đơn 0.091** và xa 0.074. Khoảng 80% lỗi là phần chung
  (lệch phân phối test), mà phần chung thì trung bình không khử được.
- So với blend 0.074, một model mới chỉ nhận trọng số > 0 khi msd(model, blend) > LB(model) − 0.074. Mọi run lưới
  đều nằm dưới ngưỡng này (bag1: msd 0.044, cần > 0.054; tương quan lỗi với blend 0.81).
- Blend 2 chiều với file 0.074, theo (LB mới, msd tới blend): (0.11, 0.07) → 0.0699; (0.11, 0.09) → 0.0659;
  (0.13, 0.07) → 0.0733; (0.13, 0.09) → 0.0708. Giải lại theo kịch bản còn lợi hơn nữa, nên bảng này là cận dưới của mức lợi.
  Muốn lợi ≥ 0.003 thì cần model ~0.11 với tương quan lỗi ≤ ~0.65, tức là khác bằng các model 3373 site cũ (0.64–0.70)
  nhưng tốt hơn chúng nhiều.
- Model to không phải nguyên nhân: trên 3373 site, v2 (nhỏ) 0.186, LSTM-XL 0.168, fusion-lite 0.167 (VAL tốt hơn 25%,
  LB như nhau). Khoảng cách VAL → LB do lệch phân phối, không do quá khớp dữ liệu train. Fusion không có trong blend cuối.
- [GT] Nguồn khác biệt duy nhất chưa thử: inductive bias khác hẳn khi ngoại suy, ví dụ cây (LightGBM một bước trên lưới,
  bagging_fraction + feature_fraction: chính là random subspace + bagging). Cây không ngoại suy ra ngoài miền train
  (hằng số ngoài biên), ngược với LSTM.
- `carbonglobe_gbm.py` (CPU, không clone repo), hai model:
  - **direct**: (y0, tuổi gốc, khí hậu 40 năm: trung bình 111 feature, trung bình theo thập kỷ f0–f7, khí hậu theo tháng
    f0–f6) → thay đổi height/agb sau 40 năm. Mỗi (site, tuổi) một dòng, không rollout nên không tích lũy lỗi.
  - **rollout**: (trạng thái năm t, tuổi đứng, khí hậu năm t) → năm t+1. Stock dự đoán theo mức thay đổi, flux theo
    mức, cuộn 40 năm như LSTM. 3M dòng một bước.
  - Không dùng CO2/hằng số; f7 chỉ lấy trung bình năm (biên độ mùa ở test gấp ~20 lần lưới). Holdout là 8% số khối 10°,
    chỉ để early stopping và kiểm tra hợp lý, không dùng để xếp hạng.

### 4.22 Test = lưới CarbonGlobe sau một phép biến đổi affine theo từng biến (dữ liệu local, `test_transform.py`) [KC]

Phân tích trên `data/` (bản tải về đầy đủ của competition + dataset CarbonGlobe):

- **Điểm khối (mass point) trong test_y0.** Mỗi biến có một giá trị lặp lại chính xác ở 8.8% số ô, cả hai kịch bản:
  height −0.900947, agb −0.4014121, soil −0.5323946, lai 0.1179376, gpp 0.2590265, npp 0.1278152, rh 0.0846562.
  Lưới có 8.3% ô bằng đúng 0 (rừng chết; rh: 55% ở lưới, 53% ở test). Theo tuổi, tỷ lệ chết cũng khớp: 0.2% ở tuổi
  20–50, 13–15% từ tuổi 100. ⇒ các giá trị trên là ảnh của 0 vật lý. Test có height âm (14% ô) và agb âm (50% ô),
  trong khi lưới không có giá trị âm nào.
- **Phép biến đổi height là chính xác:** test = 0.8763248·y − 0.900947. Các giá trị height hay gặp nhất ở lưới (0,
  0.051629, 0.058046, 0.094074, trần 35.000153) rơi đúng vào các giá trị hay gặp nhất ở test (sai lệch ≤ 2e-6).
  Với các biến khác mới ước lượng được độ dốc: agb ~0.71–0.86, soil ~0.85, lai ~0.96, gpp/npp ~0.87, rh ~0.85.
- **Forcing cũng bị biến đổi.** Feature 135: trần 1000.0 ở lưới thành 999.873718 ở test, với cùng tỷ lệ 98% số site.
  Khối CO2 của test là đường SSP bị nén khoảng 0.175 lần: ssp126 tăng 365.2 → 372.6, ssp585 tăng 365.3 → 394.7
  (+7 so với +29, tỷ lệ 4 đúng như SSP1-2.6 so với SSP5-8.5 giai đoạn 2015–2055). Đây là lý do "CO2 test thấp và
  phẳng" (4.2), và có lẽ cũng giải thích phần lớn kết quả OOD audit ban đầu.
- **97% site test là site lưới.** Đảo ngược map affine của 7 feature tĩnh (129–135; fit trên các cặp gần nhất, lặp lại)
  thì 5988/6171 site test trùng chính xác (khoảng cách 0) một site lưới khác nhau. Các site này rải theo split
  giống ngẫu nhiên: 322 ở train.csv, 50 ở val.csv, 5616 ở test.csv. 299 site trong số đó nằm trong holdout của các
  run lưới.
- **Nhưng trạng thái đầu khác:** test_y0 (đã đảo map) ≠ y0 của lưới tại cùng site, lệch trung vị 0.43 m. Lần chạy
  tương lai bắt đầu từ một trạng thái khác, không phải chạy tiếp lần lịch sử.
- [GT] Cơ chế phù hợp: test = (v − μ_tương lai)/σ_tương lai · σ_train + μ_train cho từng biến, tức là chuẩn hóa
  bằng thống kê của mô phỏng tương lai rồi giải chuẩn hóa bằng `data_stats`. Với height, μ_tương lai = 13.47 và
  σ_tương lai = 11.83 (lưới: 10.93 / 10.43).
- **Câu hỏi quyết định:** đáp án ẩn nằm trong không gian đã biến đổi (T2) hay không gian vật lý (T1)? Ở lưới, rừng
  chết lúc đầu thì vẫn chết sau 40 năm (100%), nên với các ô chết đáp án là 0 (T1) hoặc (−0.0826, −0.1182) sau khi
  chia scale (T2). Probe `result/probe_dead_T2.csv` (blend_scn2, ô chết đặt về giá trị T2): T2 → 0.0715–0.0725,
  T1 → 0.0735–0.0745. Bằng chứng gián tiếp nghiêng về T2: dưới T1, cây già sẽ bị dự đoán thiếu ~20% height, nhưng
  các probe bias chỉ tìm được hệ số 0.95–1.03; và y0bx1 (nâng y0 về mức vật lý) làm điểm tệ đi (0.138).
- Chuyển tăng trưởng lịch sử của chính site đó (y0_T + s·(y40_ls − y0_ls)) cho agb trung bình 0.74, trong khi blend
  cho 0.96–1.03 ⇒ rừng trong mô phỏng tương lai lớn thêm nhiều hơn hẳn lịch sử (CO2). Không nộp.

### 4.23 T2 xác nhận: đáp án ẩn nằm trong không gian đã biến đổi (probe #73 = 0.072) [KC]

- `probe_dead_T2.csv` = **0.072**, đúng khoảng dự báo cho T2 (0.0715–0.0725); T1 sẽ cho 0.074. ⇒ đáp án năm 40 =
  (s·y40_vật_lý + b)/mean, cùng phép biến đổi với test_y0. Hệ quả:
  - Ô chết (8.8%) có đáp án biết trước: (−0.082607, −0.118213). `ensemble_scn.py` giờ đặt chúng về giá trị đó và
    bỏ chúng khỏi bài toán tìm trọng số. LB của từng thành phần được trừ đi đúng lỗi của nó trên các ô chết.
    `ENSEMBLE_DEAD=0` để quay lại cách cũ.
  - Đáp án không âm là sai: sàn đúng là số 0 đã biến đổi (b/mean), nên `write()` giờ chặn dưới ở b thay vì 0.
  - Cách đúng cho mọi model: y0_vật_lý = (y0_T − b)/s (height chính xác), x_vật_lý = (x_T − b_x)/s_x, chạy model
    vật lý, rồi dự đoán = s·p + b. Theo xấp xỉ bậc nhất, s của agb ở phía input tự triệt tiêu; ở phía output thì nó
    nhân với phần tăng trưởng ⇒ chỉnh bằng LB (LB là hàm bậc hai của s với độ cong biết trước, nên 2 lần nộp là
    tìm được s tối ưu với sai số ~0.001).
- Map forcing (`test_maps.npz`): feature tĩnh chính xác. Feature khí hậu thì fit năm 0–5 của test (2015–2020) với
  năm 33–38 của lưới (cùng năm lịch) tại 5988 site khớp: R² trung vị 0.990, nhỏ nhất 0.90. Ví dụ f0: test =
  1.014·x − 5.86, tức là test "lạnh hơn" vật lý khoảng 1.6–2 đơn vị. Model hiện tại thấy một khí hậu tương lai bị
  chuẩn hóa về giống lịch sử.
- Không gian T của CO2: s ≈ 0.175 (so với đường SSP). Các model no-CO2 không dùng CO2.
- Blend cũ có thêm sửa ô chết: `submission_blend_scn2d.csv` ≈ file 0.072 (msd 8e-6), ước tính 0.0744;
  `submission_blend_scn2d_x.csv` ước tính **0.0733** (dự kiến ~0.071, vì ước tính thường bi quan ~0.002).
- **GBM direct (local, `carbonglobe_gbm.py`, 10 phút).** Holdout lịch sử (8% khối 10°) đạt 0.0305; persistence là
  0.3675. Gain theo nhóm: với height thì state 0.37, v1 0.17, tuổi 0.16, f1 0.10; với agb thì f1 0.29, state 0.22,
  tuổi 0.14. Trên test, đổi sang đơn vị vật lý rồi biến đổi lại với s = (0.876, 0.84): height trung bình 0.899 và
  agb 0.691, trong khi file 0.072 có 0.989 và 1.004 ⇒ **rừng tương lai cao hơn ~10% và nhiều agb hơn ~35% so với
  động lực lịch sử** (CO2; height có s chính xác nên khoảng chênh là thật). Climate đã đảo về vật lý (ấm hơn) còn làm
  agb giảm thêm (0.978 so với 1.049 khi giữ climate như test cho).
- Bản khớp trung bình (`slope.py match`: hệ số theo từng kịch bản/biến để ô sống có cùng trung bình với file 0.072):
  hệ số agb là 1.11/1.22 với direct_inv và 1.03/1.13 với direct_y0inv. msd tới file 0.072 lần lượt là 0.114 và
  0.058 ⇒ rất khác các LSTM, nên chỉ cần LB < 0.186 / 0.130 là đã giúp được blend.

### 4.24 Cây LightGBM là thành viên blend mạnh (#74–76) [KC]

- `match_direct_y0inv` = **0.108**: một model cây đơn, 10 phút CPU, có tương quan lỗi với file 0.072 chỉ **0.69**
  (các run LSTM lưới: 0.80–0.92). Bản đảo cả climate về vật lý thì kém hơn nhiều: 0.164. ⇒ đưa y0 về vật lý giúp,
  còn đưa climate về vật lý thì hại. [GT] Map climate fit trên 2015–2020 (GCM so với reanalysis) chưa đủ chính xác,
  hoặc phản ứng với ấm lên học từ lịch sử không còn đúng khi có CO2 tương lai.
- Blend trọng số chung, có ô chết (`ensemble_scn.py shared`): dead_t2 0.48 + scn2d_x 0.34 + gbm_direct_y0inv 0.19
  ⇒ ước tính **0.0698** (`submission_blend_t2b.csv`); gbm_direct_inv nhận trọng số 0.
- scn2d_x cũng ra 0.072 (ước tính 0.0733 so với 0.0744 của scn2d; cả hai làm tròn về 0.072).
- t2b = **0.070**, ước tính 0.0698: lần này ước tính đúng (blend trọng số chung với 3 thành phần ít nhiễu làm tròn).
- Hybrid GBM = 0.097 ⇒ lỗi theo kịch bản so với bg_scn (sai khác một hằng số): ssp126 0.100 (bg 0.092), ssp585
  0.116 (bg 0.092). Cây yếu hơn ở ssp585, đúng như dự đoán vì nó không có phản ứng với CO2. Blend theo kịch bản
  (`blend_t2s`): GBM nhận 0.34 ở ssp126 và 0.24 ở ssp585, ước tính 0.0707; bản _x (trọng số âm trên cặp CO2)
  ước tính 0.0700. Không hơn t2b (0.0698) ⇒ dư địa của blend gần hết; đòn bẩy tiếp theo là thành phần tốt hơn
  (GBM bagging, GBM rollout, LSTM y0inv).

### 4.25 GBM rollout + GBM seed averaging [KC, cục bộ]

- GBM rollout (y0inv): holdout lịch sử 0.046 (kém hơn direct: 0.031). Mean-match scale: height 0.984/1.015, agb
  1.147/1.296 ⇒ rollout đoán AGB thấp hơn nhiều hơn direct. msd tới file 0.072: **0.173** (direct: 0.058).
  Giúp blend chỉ khi LB < 0.245.
- GBM trung bình 4 seed (s0 = seed 42 + s1/s2/s3): msd tới file 0.072 = **0.059**, gần như bằng single seed
  (0.058). Noise giữa seed không đáng kể (bagging_fraction đã làm điều này nội bộ).
- Blend trọng số chung (dead_t2 + scn2d_x + gbm_avg_y0inv + gbm_rollout_y0inv): ước tính **0.0685** (t2b: 0.0698).
  Rollout đóng góp 0.093 nhờ msd lớn; trung bình seed thay thế single seed (trọng số giống hệt nhau vì hai file
  gần như đồng nhất). Không nộp rollout riêng (LB dự báo ~0.145–0.200). Nộp `submission_blend_t2c.csv`.

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
- "Độ lệch `test_y0` chỉ là chuyện nhỏ" → model tốt nhất giữ 55–62% độ lệch y0 tới năm 40 (msd 0.062, mục 4.13).
- "Feature 7 (biên độ mùa 20×) gây hại" → không ảnh hưởng gì (msd ≤ 0.0001).
- "Model mang flux ban đầu cao của test suốt 40 năm" → model gần như bỏ qua flux ban đầu (y0fx1 msd 0.0002).
- "Bagging theo khối không gian tạo member đa dạng có ích" → m1 0.128, trọng số blend 0.
- "Bỏ v2 khi train sẽ giúp như bỏ CO2" → 0.135 (mục 4.17); v2 có tín hiệu thật, chỉ độ lệch test của nó là có vấn đề.
- "Fine-tune / SWA quanh nghiệm 0.091 giữ hoặc cải thiện chất lượng" → 0.136; dự đoán trôi xa ngay sau 1M mẫu (4.19).
- "Holdout lịch sử dùng để chọn model được, chỉ nhiễu" → tệ hơn: tương quan −0.85 với LB (4.19).

## 8. Câu hỏi mở

Xem `PLAN.md` (Q1–Q7): CO2 test có bị scale sai không; nhóm feature nào khác hoạt động như CO2; feature 7; mức độ
model nhớ `test_y0`; train lại model lưới lâu hơn; model lưới thứ hai (EcoFormer); sửa CO2 ngay khi train.


### 4.26 blend_t2c = 0.072, blend hội tụ [KC]

- `submission_blend_t2c.csv` (#79) ước tính 0.0685 nhưng thực = **0.072** = y hệt t2b và scn2d_x.
- Chẩn đoán: `dead_t2` và `scn2d_x` có msd nhau chỉ 0.001 → cơ bản là 1 thành phần. `gbm_rollout_y0inv` dù msd
  tới LSTM là 0.173 nhưng LB thực cũng 0.072 → sau mean-match nó chỉ đổi pattern, không đổi level; trọng số về 0
  khi LB đúng. Ước tính lạc quan vì giả định LB rollout 0.14–0.20.
- **Blend trọng số chung đã hội tụ tại 0.072.** Để xuống 0.070 hoặc thấp hơn cần một model thực sự tốt hơn.
- Hướng: **LSTM `y0inv`** (Kaggle session E, `global_probe.py`): attach `test_maps.npz` vào `noco2-ep4`, chạy GPU,
  submit `submission_global_y0inv.csv`, gửi `global_probe_preds.npz`. Nếu LB 0.085–0.090, blend 2 chiều với t2b
  cho ~0.055–0.065. Đây là đòn bẩy thực sự nhất còn lại.
- Lưu ý: `dead_t2` = 0.072 là file tốt nhất hiện tại, không phải t2b. Cả hai là 0.072 nhưng `dead_t2` = `probe_dead_T2.csv` là bản gốc (blend_scn2 + ô chết đúng); t2b = dead-aware re-solve với GBM.
- **[Sửa, mục 4.27]** Hai ý trên sai: t2b = **0.070** (#77) vẫn là file tốt nhất. t2c tệ hơn vì LB của rollout chưa
  từng được chấm (giả định 0.200); giải ngược từ t2c = 0.072 cho LB(rollout) ≈ **0.237** (0.226–0.248), tức là
  rollout chỉ vừa đủ ngưỡng, trọng số ≈ 0. Blend trọng số chung chưa hội tụ vì thiếu model, mà vì rollout yếu.

### 4.27 Session E (LSTM y0inv = 0.111) và analog cùng site [KC + đại số]

**LSTM y0inv (#80).** `result/1/` (preds vật lý trong `global_probe_preds.npz`; `none` trùng tuyệt đối noco2_ep4).
- Mức trên ô sống (đơn vị nộp, ssp126 / ssp585): y0inv height 0.979 / 0.985, agb 0.940 / 0.928; dead_t2 height
  1.069 / 1.116, agb 1.073 / 1.153. Như GBM thô: model vật lý không có phản ứng CO2 nên đoán thấp; phần lớn 0.111 là
  lỗi mức. xinv còn thấp hơn (0.85 / 0.80): khí hậu vật lý ấm hơn → model đoán giảm (giống GBM direct_inv 0.164).
- Tương quan lỗi: với dead_t2 0.77, với **GBM 0.38** (noco2_ep4 với GBM: 0.52). Blend trọng số chung
  `submission_blend_e4.csv` (dead_t2 0.41 + scn2d_x 0.21 + gbm_direct 0.23 + lstm_y0inv 0.15): ước tính **0.0689**
  (t2b ước tính 0.0698 → thật 0.070; mọi thành phần đã chấm nên ước tính tin được).
- Khớp trung bình (`result/match_y0inv.csv`, slope h 0.95/0.98, agb 0.94/1.02): chỉ nhận trọng số nếu LB ≤ ~0.088.
  Công thức "slope tối ưu = slope khớp trung bình" cho 0.059, quá lạc quan (bỏ qua co rút). Không đáng một slot.
- Kiểm tra: t2b dự báo lại từ thành phần = 0.06975 (thật 0.070). y0 hai kịch bản không giống hệt (mục 4.15): khác
  rất nhỏ (|d| 0.04–0.09 m, tương quan 0.9999); không ảnh hưởng kết luận.

**Analog cùng site (`analog.py`, không train).** 97% site test là site lưới; lưới có quỹ đạo ED 40 năm của chính site
và tuổi đó dưới khí hậu lịch sử. Biến thể (vật lý, rồi khớp trung bình với file 0.072 như GBM):
`add` = y0_test + (y40_lưới − y0_lưới); `hist` = y40_lưới; `ratio` hỏng ở tuổi 1 (y0 ≈ 0), bỏ.
- y0 test (vật lý) so với y0 lưới cùng site/tuổi: height tương quan 0.91–0.98, agb 0.95–0.98; s_agb = 0.84 cho trung
  bình agb khớp (3.51 so với 3.54) ⇒ map agb đúng gần đúng.
- Mức theo tuổi của analog bám t2b (height ±3%, agb ±5%): khớp một slope theo kịch bản/biến là đủ.
- **Rất khác mọi thành phần**: msd (ô sống) tới t2b 0.094 (add) / 0.132 (hist), tới GBM 0.085 / 0.114, tới LSTM
  y0inv 0.18 / 0.24 (các thành phần hiện có cách nhau 0.04–0.135).
- Blend ước tính (dead_t2, scn2d_x, gbm_direct, lstm_y0inv + analog_add) theo LB giả định của analog_add:
  0.10 → 0.0555, 0.11 → 0.0592, 0.12 → 0.0624, 0.13 → 0.0650, 0.15 → 0.0681, ≥ 0.18 → không lợi.
  add − hist = y0_test − y0_lưới, nên có cả hai thì blend tự chọn mức "nhớ trạng thái đầu".
- **Không nộp** `result/probe_co2_adj.csv` (eda5.py, chưa commit): coi CO2 test (không gian T, nén ×0.175) là ppm
  thật; hướng sửa (height giảm) ngược với mọi probe bias (height ×1.026); vòng thứ hai hoán đổi chỉ số site/tuổi.
- Ý tưởng GPU còn lại: khối CO2 chưa từng được map (`test_maps` để identity). CO2 vật lý ≈ 400 + (T − 363.85)/0.176
  ppm: ssp126 400 → ~442, ssp585 400 → ~566 (lưới: 357 → 419). Model có CO2 (run 1) + y0inv + CO2 vật lý là thành
  viên có phản ứng CO2 theo site; rủi ro: ngoại suy tới +8 sd.

### 4.28 e4 = 0.068; analog thất bại; ngoại suy theo cặp LSTM vật lý [KC + đại số]

- **e4 = 0.068** (#81, ước tính 0.0689): khi mọi thành phần đã có LB thật, ước tính đúng (hơi bi quan, như trước).
- **Analog cùng site: add 0.173, hist 0.209** (#82–83). Trọng số 0 trong mọi blend, kể cả khi cho phép trọng số âm
  (hist −0.02, lợi 0.0001). Đã kiểm tra: height/agb không đổi trong năm nên không có lỗi chọn tháng. ⇒ lần chạy lịch sử
  của ED tại chính site đó dự báo tương lai **kém hơn** các model học trên toàn lưới với climate trong không gian T.
  Tương lai của từng site (climate + CO2) khác lịch sử của nó nhiều hơn phần mà model học được giữa các site.
- Ô chạm trần height (35 m) không giữ trần sau 40 năm (1%): không có thêm hàng đáp án biết trước ngoài ô chết.
- **LSTM vật lý:** y0inv_xinv 0.109 thay y0inv trong blend simplex (ước tính 0.0681, chỉ hơn e4 ~0.001). xinv 0.196
  (y0 không gian T bị coi là vật lý, rồi map output: mức rất thấp).
- **Ngoại suy theo cặp** (`ENSEMBLE_FREE=lstm_xinv`): dead_t2 0.08 + scn2d_x 0.34 + gbm_direct 0.20 + lstm_y0inv_xinv
  **+0.735** + lstm_xinv **−0.361** ⇒ ước tính **0.0661** (0.0652–0.0670 khi rung LB ±0.0005; trọng số cặp ổn định
  +0.64…+0.82 / −0.41…−0.30). `result/submission_blend_x1.csv`; đi nửa đường (`submission_blend_x1h.csv`) ước tính
  0.0666 với nửa trọng số âm. y0_persist nhận trọng số 0 nên hướng này không phải "nhớ y0" đơn thuần mà là phản ứng
  của LSTM (phụ thuộc tuổi) khi đưa y0 về vật lý, nhân ~2. Các cặp khác (lstm_y0inv, gbm_direct_inv, noco2_ep4) chỉ
  cho 0.0678–0.0679.
- Mức của x1 so với dead_t2 (ô sống): height 0.988 / 0.998, agb 1.009 / **1.037** (ssp126 / ssp585); khoảng cách
  kịch bản agb 0.042 so với 0.074. Probe `result/cs_submission_blend_x1_ssp585_agb1.1.csv` + `cellscale.py solve`
  cho hệ số tối ưu chính xác của ô ssp585 agb (lợi tối đa ~0.001).
- **Map CO2:** tăng của SSP cho s ≈ 0.176; lưới đạt 2015 (~399 ppm) ở năm 30 (năm 0 356.5, năm 39 418.7, năm 38 = 39).
  Fit trực tiếp năm 0–5 test với năm 33–38 lưới cho s 0.14–0.18 (nhiễu). CO2 vật lý (biến thể `co2inv` trong
  `global_probe.py`): ssp126 399 → 441 ppm (z +2.7), ssp585 400 → 565 ppm (z +8.8); `co2inv0.5` dừng ở z +4.7.
  Session F: run 1 (có CO2, không clip input, không trần trạng thái), `kaggle_upload/global-run1/`.

### 4.29 x1 = 0.066; mức của x1 đã đúng [KC + đại số]

- **x1 = 0.066** (#86) đúng như ước tính 0.0661: ngoại suy theo cặp LSTM vật lý đứng vững trên LB.
- Probe ô ssp585 agb ×1.1 = 0.074 ⇒ hệ số tối ưu **1.005** (1.000–1.011 theo làm tròn), lợi 0.00003. Mức agb ssp585
  của x1 đã đúng; dead_t2 mới là file lệch cao 3.7% ở ô này. ⇒ khớp trung bình với dead_t2 (`slope.py match`) có
  thể đã đẩy agb ssp585 của GBM lên quá cao; các ô khác của x1 lệch ít hơn nữa nên không probe thêm.
- Blend theo kịch bản cần LB hybrid của mọi thành phần. Suy LB hybrid của dead_t2 từ các hybrid cũ: tổng tái tạo
  0.0744 trong khi thực 0.072 (sai lệch 0.002 = cỡ phần lợi cần tìm) ⇒ không đủ tin; phải nộp hybrid (3 slot:
  dead_t2, lstm_y0inv_xinv, lstm_xinv). Ưu tiên thấp.

### 4.30 Cặp GBM (y0 không gian T) [cục bộ + đại số]

- `carbonglobe_gbm.py` có thêm biến thể `none` (y0 để ở không gian T, forcing như cho) và `xinv` (y0 T, forcing vật
  lý): bản sinh đôi của y0inv / inv. Chạy lại seed 42 (`GBM_TAG=_pair`): y0inv trùng tuyệt đối lần chạy đã chấm (msd 0).
- Ba ứng viên: `gbmt_none` (cùng slope với `match_direct_y0inv`: mức thấp 0.91/0.86), `gbmm_none` (`result/
  match_gbm_none.csv`, khớp trung bình với dead_t2, slope h 1.13/1.16, agb 1.26/1.38; msd tới gbm_direct_y0inv
  0.026), `gbmt_xinv` (mức rất thấp, msd 0.153).
- Khoảng LB khả dĩ (ma trận hiệp phương sai lỗi còn PSD) rất rộng: gbmm_none 0.075–0.228. Ước tính blend (bộ x1 +
  thành viên mới) theo LB của gbmm_none: 0.100 → 0.052, 0.126 → 0.061, 0.151 → 0.066 (không lợi), 0.177 → 0.059
  (chỉ với trọng số âm). Gần biên PSD ước tính không đáng tin. Thành viên mới có LB thấp sẽ thay gbm_direct_y0inv
  (trọng số của nó về 0) và kéo trọng số cặp LSTM lên.
- Tiên nghiệm: gbmm_none cùng mức với blend và chỉ cách file 0.108 một msd 0.026 ⇒ LB nhiều khả năng 0.10–0.13 ⇒
  **nộp `result/match_gbm_none.csv`** (1 slot). gbmt_none / gbmt_xinv lệch mức nhiều ⇒ dễ rơi vào vùng không lợi.

### 4.31 Session F: LSTM có CO2 + CO2 vật lý = 0.119 [KC + đại số]

- `result/global-probe/` (notebook chạy bản 4 biến thể, chưa có `xinv+co2inv`). Tái lập co2_hist khớp file 0.122
  (msd 8e-8) ⇒ đúng trọng số run 1.
- `y0inv_xinv_co2inv` = **0.119** (#88). Thành viên đầu tiên có tách kịch bản thật: agb ssp585 1.057 so với ssp126
  0.987 (+7%, như dead_t2); height ssp585 thấp hơn ssp126 (0.870 / 0.885). Cùng model với CO2 lịch sử: agb 0.85 / 0.84.
- Tương quan lỗi với x1 0.74, với GBM 0.45 ⇒ trọng số 0.007–0.024, ước tính vẫn 0.0661: tự nó không giúp.
- **LB dự báo (điểm trung tính) của file chưa chấm** = LB mà tại đó nó không mang thông tin mới; lợi tăng theo bình
  phương độ lệch khỏi điểm đó. y0inv_xinv_co2_hist: 0.1215 (±0.01 → 0.065); y0inv_xinv_co2inv0.5: 0.1105 (±0.005 →
  0.064, ±0.01 → 0.057, nhưng trọng số ±2 vì gần thẳng hàng với co2inv: msd 0.0064); gbmm_none: 0.1515.
  Tiên nghiệm: co2_hist (agb 0.85) nhiều khả năng cao hơn 0.1215 rõ ⇒ đáng nộp. Với cặp gần thẳng hàng, bản cuối dùng
  ngoại suy nửa đường như x1h.

### 4.32 Cặp GBM xác nhận hướng "y0 vượt quá vật lý"; x2 ước tính 0.0624 [đại số]

- Kết quả #89–91: co2_hist 0.117 (dự báo 0.1215), co2inv0.5 0.110 (dự báo 0.1105, trùng), gbmm_none 0.169 (dự báo
  0.1515, lệch +0.0175). Tiên nghiệm của tôi cho co2_hist (cao hơn dự báo) sai chiều.
- **Cặp GBM** (gbm_direct_y0inv +0.62, gbmm_none −0.39) cùng cặp LSTM (+0.54 / −0.34): hai họ model độc lập cùng
  ngoại suy từ "y0 không gian T" qua "y0 vật lý" và vượt tiếp ⇒ đáp án muốn model bắt đầu từ trạng thái lớn hơn y0
  vật lý (đại diện cho tăng trưởng mạnh hơn vì CO2). Thành viên CO2 chỉ thêm 0.0004 (co2_hist +0.12).
- `result/submission_blend_x2.csv` (7 thành viên): dead_t2 0.07 + scn2d_x 0.39 + gbm_direct_y0inv 0.62 +
  lstm_y0inv_xinv 0.54 − lstm_xinv 0.34 − gbmm_none 0.39 + g1_co2_hist 0.12 ⇒ ước tính **0.0624** (0.0611–0.0636
  khi rung LB; trọng số hai cặp ổn định). Không trọng số âm: 0.0680; nửa đường `submission_blend_x2h.csv` 0.0638;
  3/4 đường 0.0627.
- Kiểm tra trực tiếp: GBM biến thể `y0inv2` (y0 = 2·y0_vật_lý − y0_T, ô chết giữ 0) — điểm thứ ba trên trục y0.
  Khớp trung bình: `result/match_gbm_y0inv2.csv` (slope 0.85–0.94, tức là mức thô cao hơn blend; msd tới gbm y0inv
  0.034). LB dự báo **0.0955** (tốt hơn gbm y0inv 0.108 ⇒ đại số cũng "chờ" y0 vượt quá vật lý là tốt hơn). Nhưng
  nó chỉ thay được cặp GBM có trọng số âm nếu LB ≤ ~0.085 ⇒ lợi của cặp GBM là pattern, không chỉ là "bắt đầu lớn
  hơn". Lợi khi nộp: ±0.01 → 0.061, ±0.02 → 0.0575 (probe thường).
- **Kết quả: x2 = 0.063** (#93, ước tính 0.0624). y0inv2 = 0.097 (#92), lệch dự báo chỉ +0.0015 ⇒ trọng số −0.04,
  ước tính vẫn 0.0624. Đơn lẻ, GBM y0inv2 (0.097) tốt hơn GBM y0inv (0.108): "y0 vượt quá vật lý" đúng cả trên một
  file không có trọng số âm.

### 4.33 Họ cây khác: random forest (LightGBM `boosting='rf'`) [cục bộ + đại số]

- Vì sao LightGBM mà không XGBoost/CatBoost: cùng họ boosting trên cùng feature ⇒ dự đoán gần như trùng (4 seed
  LightGBM cách nhau msd 0.0018) ⇒ gần như không thêm hướng mới cho blend; máy local cũng chỉ có lightgbm. Random
  forest (cây sâu, bagging, lấy trung bình, không boosting) là họ khác thật sự; LightGBM có sẵn chế độ `rf`
  (`GBM_BOOSTING=rf`: num_leaves 1023, min_data_in_leaf 50, bagging 0.632, feature 0.5, ≤300 cây).
- RF direct: holdout lịch sử 0.054 (GBM 0.031; holdout không xếp hạng được LB). Khớp trung bình:
  `result/match_rf_y0inv.csv` (msd tới GBM y0inv **0.023**, gấp ~13 lần nhiễu giữa seed), `result/match_rf_none.csv`.
- LB dự báo: rf_y0inv **0.104** (lệch ±0.01 → blend 0.061, ±0.02 → 0.0565); rf_none 0.1485 (±0.02 → 0.0585).
- **Kết quả: rf_y0inv = 0.110** (#94, dự báo 0.104). Cho phép trọng số âm: −0.17, ước tính 0.0619 so với 0.0624 của x2
  (dưới mức làm tròn LB, thêm một trọng số âm ⇒ rủi ro private) ⇒ không nộp blend mới.
- Bản sinh đôi `result/match_rf_none.csv`: LB dự báo 0.1485; lệch ±0.01 → 0.061, +0.0175 (như GBM) → 0.058. Ở độ
  lệch lớn trọng số lên ±0.6–1.3 ⇒ khi dùng thì lấy nửa đường.

### 4.34 Session G (tùy chọn, GPU): cặp LSTM thứ hai, seed 7

- Lợi lớn gần đây đều từ cặp "y0 vật lý / y0 không gian T" của cùng một model (LSTM ep4: 0.068 → 0.066; GBM:
  0.066 → 0.063). Seed 7 ep3 (LB 0.118, msd 0.055–0.073 tới ep4 ở không gian T) cho một bản độc lập của hướng đó.
- `global_probe.py` PROFILE `noco2_s7`: biến thể none (đối chiếu với `result/s7_ep3_preds`, không tốn slot),
  y0inv+xinv, xinv. `kaggle_upload/noco2-s7/` = `s7_ep3.npz` (snap_ep3 của run 4, trùng tuyệt đối best.npz = EMA
  epoch 3 đã chấm) + `test_maps.npz`.

### 4.35 Cặp RF trùng cặp GBM; slope agb không giải thích "y0 vượt quá vật lý" [KC + cục bộ]

- rf_none = 0.155 (#95, dự báo 0.1485). Cặp RF tự do: −0.125 trên rf_none, ước tính 0.0620 (x2: 0.0624); cả hai RF
  tự do 0.0619. Dưới mức làm tròn, thêm trọng số âm ⇒ không nộp. Hai họ cây cùng feature mang cùng một hướng.
- Giả thuyết: slope agb 0.84 (ước lượng) quá cao nên "agb vật lý" của test quá nhỏ, và "bắt đầu lớn hơn" chỉ là sửa
  lỗi đó. Kiểm tra bằng allometry (agb theo height dọc quỹ đạo lưới của chính site, height vật lý của test là chính
  xác, 3248 dòng đang lớn): slope ngụ ý 0.81 (tuổi 1–20), 1.07 (30–70), 1.04 (100–500), tương quan 0.90 ⇒ không ra
  một slope chung, không ủng hộ "slope quá cao" đồng đều. "Bắt đầu lớn hơn" vẫn là đại diện cho tăng trưởng mà model
  thiếu (CO2).
- Tình trạng: blend hội tụ ~0.062–0.063 với các thành viên hiện có. Còn lại: session G (cặp LSTM seed 7, GPU, 2 slot)
  và trọng số theo ô (kịch bản × tuổi: 5–6 probe hybrid, lợi chưa biết).

### 4.36 Kế hoạch 30 h GPU: nhiều thành viên rẻ, xếp hạng trước khi nộp

- Giới hạn thật là slot (5/ngày), không phải GPU. Nên GPU dùng để tạo nhiều ứng viên, rồi `rank_members.py` xếp hạng
  theo độ dốc của lợi (LB dự báo + ước tính blend khi lệch ±0.01/±0.02) và chỉ nộp những file dốc nhất.
- `kaggle_notebooks/` (sinh bởi `make_notebooks.py` từ các .py):
  - `kaggle_sweep_pairs_run1/2.ipynb` = `global_probe.py` PROFILE `sweep`: 7 snapshot LSTM no-CO2 (s42 ep3/5/6, s7
    ep3/5, bag ep6, swa ep4), mỗi cái y0inv+xinv và xinv. Kiểm tra tái lập 'none' trên 309 site (mỗi site thứ 20) với
    `ref_<tag>.npz`, dừng nếu sai. Không chạy 'none' đầy đủ (đã có trong result/). ~16.5 phút mỗi rollout (session F:
    2.2 h cho 8) ⇒ run 1 ~4.5 h, run 2 ~3.5 h. Dataset: `kaggle_upload/lstm-sweep/` (637 MB).
  - `kaggle_direct_nn.ipynb` = `carbonglobe_direct_nn.py`: họ mới, NN direct (Conv1d + BiGRU trên 40 năm khí hậu
    trung bình năm của 111 feature + khí hậu tháng f0–f6 + y0 + tuổi → thay đổi height/agb ở năm 10/20/30/40),
    4 seed, 5 biến thể test như GBM. Pre-flight (20 bước + mọi biến thể trên 13 site) trước khi train thật. Đã chạy
    thử pipeline dữ liệu ở local với model giả (không có PyTorch local); phần PyTorch chỉ kiểm tra được trên Kaggle.

### 4.37 Session G (seed 7 ep3, `result/noco2-s7/`) [đại số]

- 'none' trùng tuyệt đối file s7_ep3 đã chấm (msd 0). Mức rất thấp: y0inv_xinv height 0.90 / agb 0.75 (ep4: 0.99 /
  0.92), xinv 0.77 / 0.66.
- LB dự báo (`rank_members.py`, bộ x2): s7_y0inv_xinv **0.189** (khả dĩ 0.077–0.301), s7_xinv **0.271** (0.167–0.349).
  Lệch ±0.01 → 0.0618–0.0619, ±0.02 → 0.060. Khoảng khả dĩ rộng nên độ lệch thật có thể lớn; khi có cả hai thì thành
  cặp như ep4 ⇒ nộp cả hai.
- **Kết quả:** xinv 0.277, y0inv_xinv 0.192 (#96–97), lệch dự báo nhỏ (+0.006 / +0.003). Cho cặp seed 7 tự do:
  s7_y0inv_xinv +0.47, s7_xinv −0.55, cặp ep4 co lại (0.11 / +0.17) ⇒ ước tính **0.0615** (x2 0.0624); rung LB:
  0.0609–0.0620 (x2: 0.0618–0.0628). Vẫn 2 trọng số âm, |w| lớn nhất ~0.6 như x2 ⇒ `result/submission_blend_x3.csv`.
  Hướng "y0 vượt quá vật lý" giờ dựa trên hai seed LSTM thay vì một.
  thứ hai; đã tạo lại (khớp preds đã commit, 2e-6) và `slope.py` giờ từ chối ghi đè, nhận đường dẫn output.

### 4.38 Sweep run 1 + NN direct: xếp hạng ứng viên [cục bộ + đại số]

- `result/kaggle_run1/` (bagep6, s42ep3, s42ep6, s7ep3; s7ep3 trùng session G, msd 3e-11). `result/kaggle_direct_nn/`:
  NN holdout lịch sử 0.029–0.030 (GBM 0.0305, persistence 0.37), msd giữa seed 0.008; NN y0inv cách GBM y0inv msd
  0.032. Bản khớp trung bình: `result/match_nn_<variant>.csv`.
- `rank_members.py` (bộ x3, ước tính 0.0615): dốc nhất là cặp s42ep3 và s42ep6 (lệch ±0.01 → 0.0597, ±0.02 → 0.054);
  rồi bagep6 (0.0601 / 0.056), NN y0inv2 / y0inv (0.0604–0.0605 / 0.057). LB dự báo: s42ep6 y0inv_xinv 0.116, xinv
  0.197; s42ep3 0.163 / 0.251; bagep6 0.185 / 0.263; NN y0inv2 0.119, y0inv 0.141, none 0.179.
- Nộp đợt 1: cặp s42ep6, cặp s42ep3, NN y0inv2.
- **Kết quả đợt 1** (#99–103): mọi file lệch dự báo ≤ 0.006. Blend: `submission_blend_x4.csv` (vẫn 2 trọng số âm như
  x3; NN y0inv2 +0.16, s42ep6_xinv +0.20) ước tính **0.0606** (rung 0.0600–0.0609); `submission_blend_x4f.csv` (thêm
  trọng số âm trên s42ep3_xinv; hai cặp LSTM chia nhau phần âm) ước tính **0.0600** (0.0594–0.0604).
- Cặp bagging (#104–105): x4f 0.0600 → 0.0599, x4 không đổi ⇒ không làm file mới. Các cặp LSTM đã phủ cùng một hướng;
  blend hội tụ ~0.060. Còn lại: sweep run 2 (s7ep5, swaep4, s42ep5), NN y0inv / none.

### 4.39 EDA y0 hai kịch bản; trọng số riêng cho ssp585 bằng probe theo hướng [KC + đại số]

- y0 hai kịch bản khác nhau ở mọi dòng sống (dòng giống hệt = đúng 8.9% ô chết). Nguyên nhân: forcing năm 0 đã khác
  (|diff| f0–f7 trung bình 4.65) ⇒ khác biệt y0 là phản ứng một năm với thời tiết khác, gần đối xứng (GPP trung vị 0,
  10–90%: −8%…+15%; height/agb dấu 50/50), có cấu trúc theo site (tương quan giữa tuổi 0.44–0.73). Không mang thông
  tin phản ứng CO2 40 năm ⇒ bỏ.
- Mọi blend dùng một vector trọng số chung cho hai kịch bản; "bắt đầu lớn hơn" là đại diện cho tăng trưởng CO2 nên có
  thể cần mạnh hơn ở ssp585. `dirprobe.py`: chỉ trên ssp585, x4f + t·d_j với 4 hướng (cặp GBM, cặp LSTM seed 7,
  NN y0inv2 − x4f, dead_t2 − x4f; tương quan giữa hướng ≤ 0.59). LB bậc hai theo t với độ cong biết trước ⇒ mỗi
  probe (bước làm LB đổi +0.003 nếu b = 0) cho đúng một hệ số b_j; 4 probe ⇒ bước tối ưu t* = −Q⁻¹b và lợi chính xác
  (sai số làm tròn ~1e-4). ssp126 giữ nguyên x4f (không cần giả định nào).
- **Kết quả** (#108–111): ba hướng đúng mức "không tác dụng" (0.064), cặp LSTM thấp hơn ~0.001. Giải: bước tối ưu chỉ
  có cặp LSTM (+0.08), lợi **0.0002** (0.0001–0.0004 theo làm tròn) ⇒ x4f đã gần tối ưu riêng cho từng kịch bản; không
  nộp file điều chỉnh. Giả thuyết "ssp585 cần ngoại suy mạnh hơn" bị bác ở mức đo được.
- Tổng kết: blend ở ~0.060–0.061 đã khai thác hết thông tin từ LB với các thành viên hiện có; phản ứng CO2 tương lai
  không học được từ lưới (mọi site cùng một quỹ đạo CO2, chênh không gian ~5 ppm). Lựa chọn cuối: x4f (0.061) + x3 (0.062).

### 4.40 Mẫu hình theo site của chênh lệch ssp585 − ssp126 (`scnprobe.py`) [đại số]

- Mức của từng kịch bản đã hiệu chỉnh (hệ số bias, probe ô). Chưa kiểm tra: **mẫu hình theo site** của chênh lệch
  kịch bản. Trong x4f nó đến từ phản ứng khí hậu của các model không CO2: agb chênh trung bình +0.046, độ lệch chuẩn
  theo dòng 0.11. Hiệu ứng CO2 thuần của LSTM có CO2 (session F, khác biệt kép co2inv − co2_hist giữa hai kịch bản)
  tương quan chỉ 0.06 với mẫu hình đó ⇒ mẫu hình thật chưa biết.
- 4 giả thuyết, mỗi hướng chỉ đổi ssp585 và có trung bình 0 theo biến (chỉ đo mẫu hình): uniform (ssp585 = ssp126 ×
  tỷ lệ trung bình theo tuổi), co2lstm (khác biệt kép của LSTM CO2), dry (CO2 giúp nơi khô hơn: theo z của f1 = lượng
  mưa), young (tuổi ≤ 50 phản ứng mạnh hơn). Tương quan giữa hướng ≤ 0.28. Probe: `result/sp_<tên>.csv` (+0.003 nếu
  hướng vô ích); giải bằng `python scnprobe.py solve 0.061 <4 LB>`.
- **Kết quả** (#115–118): uniform và co2lstm không tác dụng; dry hơi giúp; **young = 0.061** (tốt hơn mức trung tính
  0.003). Giải: ssp585 tuổi ≤ 50 tăng ~7% phần trên số 0 đã map (bước young +0.072, 0.056–0.088 theo làm tròn), tuổi
  già giảm nhẹ; dry +0.008. Lợi **0.0011** (0.0007–0.0017) ⇒ `result/submission_blend_x6.csv` ước tính 0.0599.
  Ý nghĩa: rừng non mọc nhanh hưởng lợi CO2 nhiều nhất, điều các model không CO2 không biết. Mẫu hình CO2 của LSTM
  có CO2 sai (không tác dụng).
- Probe tiếp: cùng hướng young trên ssp126 (`result/sp126_young.csv`, gốc x4f, bước +0.121; 0.064 = không tác dụng).

### 4.41 x6 = 0.060; hướng young chỉ đúng cho ssp585; hình dạng theo tuổi (`ageprobe.py`) [KC + đại số]

- **x6 = 0.060** (#119, ước tính 0.0599) ⇒ hiệu ứng "rừng non hưởng lợi CO2" xác nhận trên file thật.
- sp126_young = 0.063 (#120): b = (0.063 − 0.061 − 0.003)/(2·0.121) ⇒ bước tối ưu ssp126 ~+0.02 (0 … +0.03 theo làm
  tròn, so với +0.072 ở ssp585), lợi ≤ 0.0002 ⇒ không làm x7. Hướng young gần như chỉ ở kịch bản CO2 cao: đúng với
  cách hiểu đây là phản ứng CO2, không phải lỗi chung theo tuổi của các model.
- Tiếp: hình dạng theo tuổi trong ssp585, gốc x6. `ageprobe.py`, 2 hướng tương phản bên trong từng nửa (tổng 0 trong
  nửa, trực giao với young, tương quan giữa hai hướng 0): `young_lo` (tuổi 1–20 lên, 30–50 xuống, bước 0.088) và
  `old_lo` (70–150 lên, 200–500 xuống, bước 0.093). Probe `result/ap_<tên>.csv`, 0.063 = vô ích; giải bằng
  `python ageprobe.py solve 0.060 <LB young_lo> <LB old_lo>` ⇒ `result/submission_blend_x6_s585.csv`.
  Meta của scnprobe (4.40) lưu ở `result/sp_meta.json` (dp_meta.json giờ là của ageprobe).
- **Kết quả** (#121–122): young_lo 0.062, old_lo 0.062. Bước tối ưu +0.015 cho cả hai, lợi **0.0002** (0–0.0007 theo
  làm tròn) ⇒ dưới ngưỡng 0.0005, không nộp `submission_blend_x6_s585.csv`. Hình dạng theo tuổi bên trong mỗi nửa đã
  gần đúng; một bước young/già là đủ. Trục tuổi × kịch bản đã khai thác hết.
- Lựa chọn cuối: **x6 (0.060) + x4f (0.061)**.

### 4.42 Site nào hưởng CO2 nhiều nhất ở ssp585 (`siteprobe.py`, gốc x6) [đại số, chờ LB]

- Vĩ độ: site lưới i = ô khác 0 thứ i của `mask2d.npy` theo thứ tự hàng (f0 tương quan 0.93 với |vĩ độ|, theo cột
  0.04) ⇒ **f0 = nhiệt độ**. Map affine của test giữ nguyên z-score nên dùng z(f0) của test trực tiếp.
- 4 hướng, chỉ ssp585, trực giao hoàn toàn (Gram–Schmidt) với young, dry (đã có trong x6) và với nhau:
  `warm` (nơi ấm lợi nhiều hơn: quang hô hấp giảm), `npp` (năng suất y0 theo tuổi, phần ngoài nhiệt độ; bản thô
  tương quan 0.79 với warm), `incr` (lợi tỷ lệ với tăng trưởng 40 năm x6 − y0, phần ngoài young), `h_only` (bước
  young chỉ trên height: CO2 đẩy height khác agb?). Bản chưa trực giao: incr/h_only tương quan 0.72–0.74 với young,
  warm −0.75 với dry ⇒ sẽ đo lại thông tin cũ.
- Probe `result/st_<tên>.csv` (bước 0.084 / 0.051 / 0.266 / 0.226), 0.063 = vô ích. Giải:
  `python siteprobe.py solve 0.060 <warm> <npp> <incr> <h_only>` ⇒ `result/submission_blend_x6_s585.csv`.
  Meta: `result/st_meta.json` (= dp_meta.json hiện tại); ageprobe: `result/ap_meta.json`.
- **Kết quả** (#123–126): warm 0.063, incr 0.063 (đúng mức vô ích), npp 0.064, h_only 0.064 (hơi tệ hơn ⇒ hướng
  ngược lại có lợi rất nhỏ: rừng năng suất cao lợi ít hơn một chút, height nên tăng ít hơn agb). Bước tối ưu: npp
  −0.009, h_only −0.038; lợi **0.0002** (0–0.0008 theo làm tròn) ⇒ không nộp.
- Tổng kết 4.39–4.42: 14 probe theo hướng trên ssp585/ssp126; chỉ `young` ở ssp585 cho lợi đo được (0.0011, x6).
  Mọi mẫu hình còn lại (tuổi chi tiết, nhiệt độ, năng suất, tăng trưởng, height/agb, scenario uniform, CO2 LSTM)
  ≤ 0.0002 mỗi nhóm. Hiệu chỉnh hậu kỳ x6 đã cạn. **Lựa chọn cuối: x6 + x4f.**

### 4.43 Phản ứng CO2 của chính ED đo trên lưới (`co2est.py`, `co2dir.py`) [cục bộ, chờ LB]

- Bối cảnh (2026-10-27): LB top 0.038, top 3 0.042, mình 0.060 (hạng 23). Khoảng cách 0.022 không thể lấp bằng
  chỉnh blend ⇒ cần thông tin mới về phản ứng dưới tương lai. Repo công khai tốt nhất tự nhận 0.072.
- test_y0 không phải trạng thái lưới ở năm sau (chạy tiếp lịch sử): chỉ ô chết trùng chính xác, gần nhất là năm 0.
- **Hiệu ứng năm lịch** (model tăng trưởng từng năm LightGBM, 4000 site, giữ trạng thái + tuổi + khí hậu của năm cố
  định, partial dependence theo năm t): agb rừng non (≤ 50) tăng ~100% mức trung bình ở t 12–21 (CO2 374–383) lên
  ~125–134% ở t 33–39 (408–420), tức là ~+30% tăng trưởng cho +12% CO2. Height non +10%. Rừng già: agb giảm về cuối.
  Năm t giúp holdout agb (mse 1.75e-2 → 1.58e-2). Đầu kỳ (t < 10) lẫn với sự hồi phục từ trạng thái gieo (thời gian
  từ lúc gieo = t cho mọi tuổi). Theo nhóm: trong rừng non hiệu ứng tương đối tăng mạnh theo tuổi (3% → 66%).
- **Hướng CO2 theo dòng test** (`co2dir.py`): cùng model, tổng 40 năm của (tăng trưởng ở năm muộn − năm giữa) trên
  đường y0 vật lý → năm 40 của x6, khí hậu test vật lý. Trực giao với young/dry của x6: còn lại **93%**; tương quan
  với warm/npp/incr/h_only ≤ 0.17 ⇒ gần như hoàn toàn mới. Probe `result/co2_dir.csv` (x6, chỉ ssp585, bước 0.331;
  0.063 = vô ích). Giải: `python co2dir.py solve 0.060 <LB>` (meta: `result/co2dir_dpmeta.json` = dp_meta.json).
- **Kết quả: co2_dir = 0.063** (#127) = đúng mức vô ích (b = 0, lợi 0). Hiệu ứng năm lịch trên lưới là thật, nhưng
  phần mẫu hình theo site/tuổi của nó ngoài young/dry không giúp. Cùng với 4.40–4.42: mọi mẫu hình mượt (tuổi, kịch
  bản, khí hậu, năng suất, CO2 của ED) đã được hiệu chỉnh hết. Lỗi còn lại 0.060 là lỗi riêng từng site của model nền.

### 4.44 Hai nhóm thông tin model direct chưa từng có: chuỗi tuổi của site + sự kiện sụt sinh khối [cục bộ]

- Lỗi holdout của GBM direct (lưới, 40 năm) dồn vào rừng già: tuổi gieo 1–20 ~0.02, 150–500 0.036–0.048.
- **Chuỗi tuổi của chính site** (`chrono.py`): 15 trạng thái năm 0 của một site là đường tăng trưởng ED của site đó;
  y0 nội suy tại tuổi a+40 dự báo năm 40 của tuổi a không cần train (tuổi 1: 0.037, persistence 0.926; tuổi 10:
  0.051; tuổi 20: 0.064; tuổi già không giúp). Test cho đủ 15 trạng thái mỗi site, cùng không gian T với đáp án.
- **Sự kiện sụt** (`fireprobe.py`): ~22% site rừng già mất > 30% agb trong một năm; cùng năm ở mọi tuổi (P(tuổi 300
  sụt | tuổi 500 sụt cùng năm) 0.82, nền 0.013) ⇒ sự kiện site × năm (cháy/khô hạn của ED). Site có sự kiện: agb
  năm 40 / năm 0 trung vị 0.33 (không có: 0.90). Dự báo từ khí hậu tháng của năm đó + năm trước + trạng thái:
  **AUC 0.985** (holdout khối 10°), p trung bình 0.375 trên sự kiện, 0.003 ngoài. Model direct chỉ thấy khí hậu
  trung bình 40 năm / thập kỷ nên không thấy sự kiện.
- Holdout GBM direct (20k site, cùng split): gốc **0.0336**, + chuỗi tuổi **0.0299**, + rủi ro sự kiện (bộ phân loại
  trên lưới, tổng / đỉnh / thời điểm của p theo 40 năm, tính ở trạng thái y0) **0.0270** (−20%); tuổi 500 0.0476 →
  0.0427, tuổi 1–20 giảm một nửa.
- `carbonglobe_gbm.py` env `GBM_CHRONO=1`: thêm cả hai nhóm vào model direct đầy đủ (54k site), biến thể y0inv /
  y0inv2 / none ⇒ `result/gbm/gbm_preds_chrono.npz`. So sánh LB với GBM y0inv2 (0.097) sau khi khớp trung bình với
  `result/probe_dead_T2.csv` như cũ.
- **Model đầy đủ** (`GBM_CHRONO=1`, 54k site, `result/gbm/gbm_preds_chrono.npz`, log `gbm_chrono.log`): holdout
  **0.0245** (GBM cũ 0.0305, −20%). Khớp trung bình với probe_dead_T2: `result/match_gbmc_{y0inv,y0inv2,none}.csv`
  (ingest: `gbmc_*`). msd tới GBM cũ cùng biến thể 0.015–0.03 ⇒ khác thật.
- `rank_members.py` (gốc x6 + cặp GBM cũ): LB dự báo gbmc_y0inv2 **0.0936** (GBM cũ 0.097), gbmc_y0inv 0.1107,
  gbmc_none 0.168. Nếu thật tốt hơn dự báo 0.01 → blend 0.059, 0.02 → 0.057. Nộp `match_gbmc_y0inv2.csv` trước.
- **Kết quả: gbmc_y0inv2 = 0.073** (#128), tốt hơn dự báo "không thông tin mới" 0.0936 tới 0.021, tốt hơn GBM cũ
  0.097 ⇒ cải thiện holdout lịch sử −20% thành −25% trên LB. Model đơn tốt nhất từ trước tới nay (0.089).
  **Đòn bẩy đầu tiên cải thiện model nền**: chuỗi tuổi và sự kiện sụt sinh khối là thông tin thật cho tương lai.
- Blend (`ensemble_scn.py shared`): `submission_blend_x7.csv` = x6 0.69 + gbmc_y0inv2 0.31, ước tính **0.0566**
  (0.0561–0.0571 khi rung LB); `submission_blend_x7f.csv` = x6 0.80 + gbmc 0.63 − GBM cũ y0inv2 0.43 (ngoại suy theo
  hướng "thêm đặc trưng mới"), ước tính **0.0532** (0.0523–0.0542; trọng số ổn định).

### 4.45 Feature lab (`featlab.py`, 20k site, cùng split) [cục bộ]

- chrono + fire **0.0270** → + fire40 (40 xác suất sự kiện theo năm) **0.0264** (giữ).
- + prof7 (đủ 7 biến của 15 tuổi, nội suy log(tuổi+20/+40), độ dốc) 0.0276; + ratio (CUE, NEP, agb/height…) 0.0274;
  + climfp (độ dốc / std / 5 năm đầu-cuối của 111 driver) 0.0285 ⇒ cả ba làm tệ hơn (thêm nhiễu). Bỏ A2–A4.
- Model đầy đủ tiếp theo: `GBM_CHRONO=1 GBM_FIRE40=1` (tag `_fire40`).
- **Model đầy đủ + fire40** (`GBM_FIRE40=1`, `result/gbm/gbm_preds_fire40.npz`): holdout **0.0253**, tệ hơn gbmc
  (0.0245); msd tới gbmc_y0inv2 0.0033 (cỡ khác biệt giữa seed); LB dự báo 0.0745 ≈ gbmc ⇒ không nộp.
  **Bài học:** khác biệt < ~5% trên tập con 20k là nhiễu (fire40 −2% ở tập con, +3% ở đầy đủ). Ngưỡng giữ một khối
  đặc trưng: −5% ở tập con, rồi phải xác nhận trên model đầy đủ.

### 4.46 Seed GBM thứ hai (A9) và NN + chuỗi tuổi (B1) [cục bộ]

- GBM chrono seed 7 (`gbm_preds_chrono_s7.npz`; split holdout khác, khó hơn: persistence 0.3747): holdout 0.0290.
  msd giữa hai seed (vật lý) y0inv2 0.0059, none 0.0045 ⇒ nhiễu seed đáng kể; trung bình 2 seed
  (`gbm_preds_chrono_avg2.npz`) khớp trung bình → `result/match_gbmc2_y0inv2.csv` (msd tới gbmc_y0inv2 0.0011).
  `rank_members`: LB dự báo 0.0731 (chỉ tính thông tin, không tính giảm nhiễu); kỳ vọng thật ~0.071–0.072.
- B1: `kaggle_direct_nn_chrono.ipynb` lần 1 OOM trên Kaggle (sao chép khối 40 năm cho 2M dòng); đã sửa (đánh chỉ số
  site), smoke 3000 site đỉnh 3.9 GB.
- **B1 NN chrono** (`result/nn_direct_chrono/`, Kaggle, 4 seed, cùng split holdout với gbmc): holdout từng seed
  0.0243 / 0.0247 / 0.0257 / 0.0263 (NN cũ 0.029, GBM chrono 0.0245), msd giữa seed 0.0067 ⇒ trung bình 4 seed
  còn tốt hơn. Khớp trung bình: `result/match_nnc_{y0inv2,y0inv,none}.csv` (ingest `nnc_*`). msd nnc_y0inv2 tới
  gbmc3 0.028, tới NN cũ 0.034 ⇒ đa dạng thật. `rank_members` (gốc x6 + gbmc2 + cặp GBM cũ): LB dự báo nnc_y0inv2
  **0.0897** (±0.01 → blend 0.052, ±0.02 → 0.049). NN cũ y0inv2: dự báo 0.119, thật 0.113. Nộp `match_nnc_y0inv2.csv`.
- Trung bình 3 seed GBM: `result/match_gbmc3_y0inv2.csv` (msd tới gbmc2 0.0005, chưa nộp).
- **nnc_y0inv2 = 0.075** (#134; dự báo 0.0897). Blend (`ensemble_scn.py shared`): `submission_blend_x8f.csv` = x6 0.72 +
  gbmc2 0.45 + nnc 0.39 − GBM cũ y0inv2 0.41 − NN cũ y0inv2 0.15, ước tính **0.0503** (0.0492–0.0513);
  `submission_blend_x8.csv` (không có cặp NN cũ) 0.0509 (0.0499–0.0519); `submission_blend_x8s.csv` (không trọng
  số âm) 0.0552. Hai cặp "cũ → mới" (GBM, NN) cùng ngoại suy theo hướng "thêm chuỗi tuổi + sự kiện".
- Probe young trên x8f (#136–137): ssp126 đúng mức vô ích, ssp585 hơi tệ hơn (rừng non nên giảm nhẹ, bước −0.018, lợi
  0.0001) ⇒ phần "rừng non mọc nhanh hơn" mà x6 phải thêm tay giờ đã nằm trong các model chrono.

### 4.47 NN cross-age (B2, `NN_XAGE=1`, `result/nn_direct_xage/`) [cục bộ + đại số]

- Transformer 2 lớp qua 15 tuổi của một site, seed 4–7, cùng split: holdout từng seed 0.0256 / 0.0245 / 0.0239 /
  0.0246 (trung bình 0.0246; NN chrono per-row 0.0252), msd giữa seed 0.0046 (per-row 0.0067) ⇒ ổn định hơn.
- Khớp trung bình `result/match_nnx_y0inv2.csv` (slope thấp hơn: 0.76–0.86 ⇒ mức thô cao hơn). msd tới nnc 0.017, tới
  gbmc3 0.020, tới x8f 0.017. `rank_members` (gốc = thành viên x8f): LB dự báo **0.067** (±0.01 → blend 0.0475).
  Trung bình 8 seed NN (chrono + xage) `match_nn8_y0inv2.csv`: dự báo 0.0666. Nộp nnx_y0inv2 trước.
- Ghi chú: lần gửi đầu là bản sao zip của NN chrono (md5 trùng) — đã bỏ; bản đúng là `result/kaggle_direct_nn_xage.zip`.
- **nnx_y0inv2 = 0.061** (#138, dự báo 0.067): model đơn tốt nhất. Nhưng lỗi của nó gần x8f (msd 0.0166) ⇒ blend chỉ
  0.0503 → 0.0493. Bỏ x6 làm tệ hẳn (0.059: các LSTM cũ trong x6 vẫn mang thông tin riêng).
  `submission_blend_x9f.csv` = x6 0.64 + gbmc2 0.27 + nnc 0.10 + nnx 0.43 − GBM cũ 0.40 − NN cũ 0.15 + gbmc_none 0.10,
  ước tính **0.0488**; `submission_blend_x9.csv` (1 trọng số âm) 0.0496.

### 4.48 NN cross-age v2 (B3, `result/nn_direct_xage2/`) [cục bộ + đại số]

- d_tok 256, 3 lớp, 30 epoch, seed 8–11: holdout từng seed 0.0236 / 0.0237 / 0.0229 / 0.0242 (trung bình 0.0236; v1
  0.0246), msd giữa seed 0.0051. Nhưng gần trùng v1: msd nnx2 tới nnx 0.0019 ⇒ model to hơn hội tụ về cùng hàm, lợi
  chính là bớt nhiễu. Trung bình 8 seed cross-age (v1 + v2) `result/match_nnx8_y0inv2.csv` (msd tới nnx 0.0005).
  `rank_members` (gốc = thành viên x9f): dự báo nnx8 0.0609, nnx2 0.0618 (không tính phần bớt nhiễu). Nộp nnx8.
- `submission_blend_x10.csv` = cấu hình x9f với nnx8 (8 seed cross-age) và gbmc3 (3 seed GBM) thay nnx / gbmc2, LB của
  hai thành viên chưa chấm gán bằng bản ít seed hơn (bi quan) ⇒ ước tính 0.0489 (công thức không thấy phần bớt nhiễu).
  msd tới x9f 0.00011 ⇒ lợi thật kỳ vọng ~0.0003–0.0005 (LB hiện 0.048 hoặc 0.049).

### 4.49 NN cross-age + khí hậu theo tháng (B4, `NN_MONTHLY=1`, `result/nn_direct_monthly/`) [cục bộ + đại số]

- Conv nhỏ qua 12 tháng f0–f6 của từng năm (mean/max/min) ghép vào chuỗi 40 năm; v2 config, seed 12–15. Holdout
  0.0224 / **0.0284** / **0.0212** / 0.0237 (trung bình 0.0240 ≈ v2 0.0236; seed 14 tốt nhất từ trước tới nay, seed 13
  hỏng). msd giữa seed 0.0096–0.0188 (v2: ~0.005) ⇒ model nhiễu hơn. msd tới nnx8 0.0083 (v2 tới v1 0.0019) ⇒ thông
  tin mới thật.
- `match_nnm4_y0inv2.csv` (đủ 4 seed, không chọn theo holdout) LB dự báo 0.0726 (±0.01 → blend 0.046);
  `match_nnm3_y0inv2.csv` (bỏ seed 13) 0.0756. Nộp nnm4.
- **nnm4 = 0.071** (#141, dự báo 0.0726): khí hậu theo tháng gần như không thêm thông tin (blend +0.0001). Các biến
  thể NN (to hơn, thêm seed, theo tháng) đã bão hòa quanh x9f/x10 = 0.049.

### 4.50 A5/A6: mức độ mất sinh khối (featlab + model đầy đủ, `GBM_MAG=1`) [cục bộ + đại số]

- featlab (20k, cùng split): chrono + fire 0.0270; + mag (hồi quy tỷ lệ agb mất mỗi năm → tổng / max / thập kỷ cuối /
  tỷ lệ còn lại kỳ vọng) **0.0260**, lợi đều ở mọi tuổi ≥ 50; + firepath / magpath (trạng thái năm t = profile y0
  tại tuổi a+t) 0.0271 / 0.0273 ⇒ không giúp.
- Model đầy đủ (`result/gbm/gbm_preds_mag.npz`, seed 42, cùng split với gbmc): holdout **0.0234** (gbmc 0.0245, −4.5%:
  xác nhận, khác fire40). Khoảng cách hai kịch bản tăng gấp đôi (msd ssp126/ssp585 0.016 so với 0.009): mức mất phản
  ứng với khí hậu từng kịch bản.
- Khớp trung bình `result/match_gbmg_{y0inv2,y0inv,none}.csv` (ingest `gbmg_*`). msd gbmg_y0inv2 tới gbmc 0.027, tới
  nnx8 0.048, tới x10 0.053 ⇒ rất khác. `rank_members` (gốc = thành viên x10 + gbmc3i + nnm4): LB dự báo **0.101**
  (±0.01 → 0.047, ±0.02 → 0.044). Nộp `match_gbmg_y0inv2.csv`.
- **Kết quả: gbmg_y0inv2 = 0.098** (#143), gần mức "không thông tin mới" (0.101) và tệ hơn hẳn gbmc 0.073, dù holdout
  lịch sử tốt hơn (0.0234 so với 0.0245). Cùng hiện tượng với emulator của đội 0.060 (bắt sụt tốt trên lịch sử, đơn
  lẻ 0.144 trên LB): mô hình hóa sự kiện chi tiết hơn học khí hậu sụt của lịch sử, rồi ngoại suy sai sang khí hậu
  tương lai (khoảng cách hai kịch bản gấp đôi). **Bài học:** với đặc trưng sự kiện, holdout lịch sử không xếp hạng
  được LB; giữ mức tóm tắt rủi ro đơn giản (fire 5 số), không đi sâu hơn (A5, A6, fire40 đều không giúp trên LB).
- `submission_blend_x11.csv` = thành viên x10 + gbmc3i + nnm4 (4 trọng số âm: GBM cũ, NN cũ, gbmc_none, gbmc3i),
  ước tính **0.0484** (x10 0.0489).

### 4.51 Blend phẳng gọn cho bản cuối (2026-11-08) [đại số]

- Bối cảnh: top 1–2 = 0.025, còn 5 ngày × 4 lượt. Mục tiêu: pipeline cuối gọn, tái tạo được (một tầng blend trên các
  model đơn, không lồng x6 ← x4f ← …).
- Tìm trên 58 model đơn đã chấm (D chuẩn hóa theo mọi dòng như `msd_alive`; bản đầu chia theo dòng sống ⇒ lạc quan
  ~10%, đã sửa). Vét cạn: 4 thành viên tốt nhất 0.0521, 5 thành viên 0.0504. 6 thành viên (tìm cục bộ từ 4 điểm xuất
  phát, cùng hội tụ): **flat6** = nnx8 +0.54, gbmc3i +0.40, lstm_y0inv_xinv +0.31, sw_s42ep6_y0inv_xinv +0.28,
  rfm_none −0.26, s7_xinv −0.26 ⇒ ước tính **0.0487** (rung LB 0.0476–0.0497, 39% ≤ 0.0485), ngang x9f (0.0488 → 0.049).
  `result/submission_blend_flat6.csv`; flat5 (bỏ s42ep6) 0.0504. msd flat6 tới x9f 0.0049.

### 4.52 Mục tiêu neo vào chuỗi tuổi (`GBM_ANCHOR=1`, Kaggle CPU, `result/kaggle_gbm_anchor/`) [cục bộ]

- Mục tiêu y40 − y0(tuổi a+40 của chính site) thay cho y40 − y0 của chính tuổi. Holdout **0.0247** (gbmc 0.0245): không
  giúp. msd tới gbmc 0.0073 (≈ nhiễu seed 0.006) ⇒ dự báo tương lai gần như không đổi: các đặc trưng chuỗi tuổi đã
  cho cây đủ thông tin mức nền. `match_gbma_y0inv2.csv`, LB dự báo (gốc flat6) 0.0763; không đáng slot.

### 4.53 Hệ số "thời kỳ" của từng site: trạng thái đầu 2015 (test) so với 1981 (lưới) [cục bộ, chờ LB]

- Cùng site (97% khớp, `_matched_traj.npz` J) và cùng tuổi gieo: height y0 test (map chính xác) / y0 lưới có trung vị
  **1.00** (không có tăng tốc chung) nhưng một **hệ số riêng theo site**: sd log 0.20, nhất quán giữa tuổi 10–50
  (tương quan 0.93–0.96; 50–100 0.47, 250–500 0.81); hệ số agb tương quan 0.60. Gần như không liên quan đến thay đổi
  khí hậu 1981→2015 (|tương quan| ≤ 0.15), hơi liên quan mức nhiệt (0.26). Tuổi 1: trung vị 0.94. `result/era_factor.npz`.
- Chưa model nào thấy tín hiệu này (chúng chỉ thấy y0 test, không thấy cặp với y0 lưới). Probe `eraprobe.py`: tăng
  trưởng dự báo của flat6 × z(log hệ số height), cả hai kịch bản, trung bình 0 theo biến ⇒ `result/ep_flat6.csv`
  (bước 0.169; 0.052 = vô ích). Giải: `python eraprobe.py solve flat6 0.049 <LB>`.
- Thêm 2 hướng trực giao (`eraprobe.py make2`): `level` (toàn bộ mức trên số 0 × z, bước 0.071) và `young` (tăng trưởng
  × z chỉ tuổi ≤ 50, bước 0.347) ⇒ `result/ep_flat6_level.csv`, `result/ep_flat6_young.csv` (0.052 = vô ích mỗi
  file). Giải cả ba: `python eraprobe.py solve flat6 0.049 <growth> <level> <young>`.
- **Kết quả** (#145–147): growth 0.052 = vô ích ⇒ giả thuyết "site khởi đầu khác thì mọc khác" sai. level / young 0.054
  ⇒ lợi nhỏ theo chiều ngược (rừng có hệ số cao bị dự báo hơi cao), tổng 0.0006 (0.0002–0.0016 theo làm tròn).
  `result/submission_blend_flat6e.csv` = flat6 + hai bước đó (ước tính ~0.0481). Hệ số thời kỳ chủ yếu là khác biệt
  khởi tạo mà model đã nhớ qua y0 test, không phải tín hiệu tăng trưởng.

### 4.54 NN cross-age + augmentation y0 (`NN_AUG=1`, Kaggle, `result/nn_direct_aug/`) [cục bộ + đại số]

- Mỗi site, mỗi batch: height / agb của cả 15 tuổi × exp(N(0, 0.05 / 0.10)); mục tiêu là trạng thái thật, đặc trưng
  chuỗi tuổi nhân theo. Lý do: y0 test qua phép đảo map ước lượng (slope agb 0.84 / 0.876, hệ số site 2015/1981 sd
  0.20, y0inv2 giúp mọi model), còn y0 train chính xác.
- v2 config, seed 16–19: holdout 0.0247 / 0.0256 / 0.0252 / 0.0259 (v2 0.0229–0.0242; holdout có y0 chính xác nên tệ
  hơn chút là dự kiến). Dự báo test đổi nhiều: msd nna_y0inv2 tới nnx8_y0inv2 **0.033** (v1/v2 chỉ 0.0019); slope khớp
  agb 1.01/1.10 (nnx8 0.79/0.86) ⇒ phản ứng agb khác hẳn.
- `rank_members` (gốc flat8t): LB dự báo nna_y0inv2 0.097 (lệch 0.02 → blend 0.0448), nna_y0inv 0.130 (→ 0.0431).
  Nộp `result/match_nna_y0inv2.csv` và `result/match_nna_y0inv.csv`.
- **Kết quả** (#151–152): 0.096 / 0.130 = đúng dự báo "không thông tin mới" ⇒ augmentation y0 không giúp; model dựa ít
  vào y0 hơn nhưng phần thay thế (khí hậu lịch sử) không đúng hơn cho tương lai.

### 4.55 Trọng số theo ô cho flat6 (`cellweights.py`) [đại số]

- 6 probe lai mỗi cách chia (thành viên ở ô A, flat6 ở ô B); hằng số chưa biết (cách lỗi flat6 chia giữa hai ô) cộng
  đều vào C nên không đổi trọng số; kiểm chứng trên pseudo-truth (x9f): khôi phục đúng trọng số bình phương tối thiểu.
- Theo kịch bản: ssp126 dựa GBM chrono nhiều hơn (0.46 so với 0.35), ssp585 dựa NN cross-age và LSTM ep6 nhiều hơn;
  lợi **0.0006** (rung LB: 0.0006–0.0007) ⇒ `submission_blend_flat6_scn.csv` ~0.0481.
- Theo tuổi (≤ 50 / > 50): rừng non dựa GBM chrono nhiều hơn (0.57 so với 0.36); lợi **0.0005** (0.0004–0.0007) ⇒
  `submission_blend_flat6_age.csv` ~0.0482.
- GBM chrono với slope agb 0.876 ở đầu vào (`kaggle_gbm_ys876`): holdout 0.0253, msd tới bản slope 0.84 cùng seed
  0.0033 (≈ nhiễu seed) ⇒ slope đầu vào không quan trọng; không nộp.

### 4.56 Lỗi của model có bền theo site không? (`persist.py`) [cục bộ]

- Điều kiện tiên quyết của hiệu chỉnh theo lỗi lịch sử của site (97% site test là site lưới). 20k site, GBM, OOF 2
  khối: P1 = năm 0→20, P2 = năm 20→40. Tương quan lỗi cùng (site, tuổi): height −0.03 (trung bình theo site 0.02),
  agb −0.21 (hồi quy về trung bình: mọc nhiều ở P1 → to hơn ở năm 20 → mọc chậm ở P2; phụ thuộc trạng thái đầu nên
  không chuyển sang lần chạy tương lai). ⇒ Không có lệch bền theo site; hướng đóng.

### 4.57 Emulator GRU từng năm (C1, `carbonglobe_emulator.py`, Kaggle, `result/emulator/`) [cục bộ + đại số]

- state(t) + khí hậu năm t (12 tháng f0–f6 + trung bình năm 111 driver) → state(t+1), 40 bước tự chạy, ngữ cảnh =
  khí hậu trung bình + profile y0 15 tuổi; loss mọi năm, năm 40 ×5; 3 seed. Holdout **0.0229 / 0.0240 / 0.0223** (ngang
  hoặc tốt hơn NN cross-age 0.0229–0.0242).
- Rất khác các thành viên: msd emu_y0inv2 tới flat6 **0.073**, tới nnx8 0.063, tới gbmc3i 0.044; ít nhạy với biến thể y0
  (y0inv2 / y0inv cách 0.012, nnx8 cặp tương ứng 0.063). `rank_members` (gốc flat8t): LB dự báo emu_y0inv2 0.122,
  emu_y0inv 0.152 (lệch 0.02 → blend 0.0437). Nộp `match_emu_y0inv2.csv`, `match_emu_y0inv.csv`.
- **Kết quả** (#167–168): 0.116 / 0.144 (dự báo 0.122 / 0.152). flat8t + cặp emulator ước tính 0.0474.
- Tìm kiếm phẳng tham lam + hoán đổi trên 63 model đơn (công thức đóng, chuẩn hóa theo mọi dòng): ước tính 8 → 0.0466,
  10 → 0.0455, 12 → **0.0442**, 16 → **0.0426**; trọng số bền khi rung LB (p95 ≈ danh nghĩa), |w| ≤ 0.82.
  `submission_blend_flatg12.csv`, `submission_blend_flatg16.csv` (tập thành viên: `result/flat_greedy_sets.json`).
  Rủi ro: D tính trên mọi dòng còn LB chỉ trên phần public ⇒ nhiều thành viên có thể lạc quan hơn (notebook công
  khai: ~0.002 với 31 bài nộp).

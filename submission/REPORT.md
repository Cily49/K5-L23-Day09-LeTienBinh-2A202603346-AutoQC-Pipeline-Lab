# Báo cáo Day 09 — Lê Tiến Bình, 2A202603346

## 1. Kết luận trước
Pipeline kết luận **Release** (mã thoát 0). Đầu vào **224** ảnh. Ở cửa vào loại **22** ảnh: 2 ảnh hỏng, 3 ảnh nhỏ, 8 ảnh trùng từng byte, 4 ảnh trùng gần đúng và 5 ảnh rò rỉ từ test. Dataset `submission/export/clean_v1` xuất ra **202** ảnh.
Sau khi máy sửa và người duyệt 80 ảnh, `07_error_list_v3` không còn **lỗi bắt buộc** nào. Còn **16/202 ảnh (7,9 %)** có cảnh báo chưa được duyệt, thấp hơn ngưỡng Hold 10 % (`08_summary.json`).
Kết luận Release chỉ đạt được khi đặt `second_opinion.conf = 0.8` (xem §3). Với `conf ≤ 0.7`, cùng dữ liệu này ra **Hold**.

## 2. Lỗi và phân việc
Bảng lấy từ `02_error_list_v1` (lô 202 ảnh sau cửa vào; các mã cửa vào tính trên 224 ảnh).

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| CORRUPT_IMAGE / BAD_SIZE | 2 / 3 ảnh | máy (loại ở cửa vào) | không đọc được hoặc cạnh < 250 px: không dùng để train được |
| DUP_IMAGE_EXACT / DUP_IMAGE_NEAR | 8 / 4 ảnh | máy (loại) | md5 hoặc pHash ≤ 12: giữ lại sẽ làm lệch phân bố, val bị "thấy trước" |
| LEAK_TEST | 5 ảnh | máy (loại) | trùng ảnh test/val nên giữ lại sẽ thổi phồng mAP |
| BAD_CLASS_ID | 6 dòng / 6 ảnh | máy (`drop_line`) | id 93, -1, 4, 7 không thuộc lớp nào nên không đoán được lớp đúng; xoá dòng an toàn hơn |
| BAD_FORMAT | 3 dòng / 3 ảnh | máy (`drop_line`) | có `nan` hoặc thiếu cột |
| BOX_ZERO | 4 dòng / 4 ảnh | máy (`drop_line`) | box w hoặc h = 0 |
| DUP_BOX | 7 dòng / 7 ảnh | máy (`drop_line`) | box cùng lớp, IoU > 0,9 với một dòng trước đó |
| BOX_OUT_OF_IMAGE | 8 dòng / 8 ảnh | máy (`clip_box`) | tràn 0,023–0,06; cắt về [0,1] mà vẫn giữ lớp |
| BAD_CLASS_NAME | 2 tên (`Person`, `Bus`) / data.yaml | máy (`rename_classes`) | ghi lại `names` theo `classes` |
| MISSING_LABEL_FILE | 6 ảnh | **người** (`needs_label`) | máy không được bịa nhãn và không tạo file rỗng (rỗng nghĩa là "ảnh nền") |
| POSSIBLE_WRONG_CLASS | 14 ảnh | **người** | vd. img_0020: xe buýt bị gán `truck`; img_0070: người bị gán `car` (đã mở `viz_labels.py` để xác nhận) |
| POSSIBLE_MISSING | 32 ảnh | **người** | model thấy vật (conf ≥ 0,8) nhưng nhãn không có box |
| POSSIBLE_FALSE_POS | 29 dòng / 27 ảnh | **người** | box nhãn không khớp dự đoán nào |
| POSSIBLE_LOOSE_BOX | 12 ảnh | **người** | đúng vật, đúng lớp nhưng IoU < 0,5 |
| EMPTY_LABEL, BOX_TINY, BOX_HUGE, CLASS_RARE | 12 / 4 / 3 / 1 ảnh | **người** (chỉ cảnh báo) | ảnh nền hay quên gán? box quá nhỏ hoặc quá to? cần mắt người |

Theo `03_fixes.csv`, máy đã thực hiện 35 thay đổi: 20 `drop_line`, 8 `clip_box`, 1 `rename_classes` và 6 `needs_label`. Mỗi thay đổi đều ghi `before → after`.

**Hàng đợi** (`05_review_queue.csv`, 102 ảnh) được xếp theo tổng trọng số `review.weights`. Đầu hàng là img_0020 (11,5 điểm), ảnh vừa bị máy xoá dòng (`FIXED_DROP_LINE`) vừa có cờ WRONG_CLASS và MISSING. Tiếp theo là 6 ảnh `NEEDS_LABEL` (10 điểm), vì đây là ảnh duy nhất mà chỉ người mới xử lý được và nếu bỏ sót thì kết quả chắc chắn là Rework. Sau đó đến các ảnh có WRONG_CLASS (3), MISSING/FALSE_POS (2)… và cuối cùng là các cảnh báo nhẹ BOX_TINY/DENSE (0,3).
Với ngân sách 80 ảnh, người duyệt xử lý được 80/102 ảnh trong hàng đợi, trong đó có đủ 6 ảnh thiếu nhãn. Ở v3, cảnh báo ngữ nghĩa còn lại chỉ là 1 WRONG_CLASS, 1 MISSING, 1 LOOSE_BOX và 13 FALSE_POS. Như vậy phần lớn lỗi ngữ nghĩa (14 + 32 + 12 ảnh ở v1) đã được người xem.

## 3. Calibration (2 ngưỡng)
Nguồn: `submission/calibration/calibration_report.md` (39 ảnh calib).

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | NEAR: 1,00/0,50 → 1,00/0,67 (EXACT giữ 1,00/1,00) | Trên calib, vùng 4–14 phẳng (P=1, R=0,67). Giá trị 16 có F1 calib cao nhất (0,91), nhưng trên raw nó loại **31** ảnh thay vì 22 (xem bên dưới). Mình chọn 12 vì nằm giữa vùng an toàn và bắt được mọi cặp trùng thật đã mở xem (khoảng cách 0, 6, 10). |
| `second_opinion.conf` | 0.25 | **0.8** | WRONG_CLASS 0,56/0,83 → **1,00/0,83**; MISSING 0,50/1,00 → 0,67/**0,75**; LOOSE_BOX 0,29/0,40 → 0,50/0,20; FALSE_POS không đổi 0,31/1,00 | Ngân sách duyệt chỉ có 80 ảnh nên precision quan trọng: mỗi cờ sai tốn một lượt duyệt. Ở 0,25 hàng đợi raw có 143 ảnh và 58/202 ảnh (28,7 %) không được duyệt. Ở 0,8 hàng đợi còn 102 ảnh và 16/202 (7,9 %). Cái giá phải trả: recall MISSING giảm từ 1,00 xuống 0,75 (2/8 lỗi thiếu box trên calib bị bỏ sót) và LOOSE_BOX gần như mù (1/5). |

**Chỗ calibration có thể đánh lừa:**
1. `phash_dist = 16` là "tốt nhất" trên calib (F1 NEAR 0,91), nhưng mình đã chạy pipeline trên raw với 4 giá trị: 4 → loại 19 ảnh, 8 → 21, 12 → 22, **16 → 31**. Mở các cặp ở khoảng cách 16 thì thấy img_0011 (tay vợt) bị coi là trùng với img_0008 (2 người trượt tuyết), và img_0177 (lướt sóng) bị coi là trùng với img_0113 (người ngồi bàn). Đây là hai ảnh hoàn toàn khác nhau. Calib chỉ có 6 cặp trùng gần nên không đủ để thấy loại dương tính giả này.
2. `conf`: F1 trên calib chỉ dựa vào 5–8 lỗi mỗi mã. Mức giảm recall MISSING 1,00 → 0,75 chỉ tương ứng 2 ảnh, nên con số này rất nhiễu. Ngoài ra, chọn 0,8 **cũng chính là** ngưỡng làm kết luận chuyển từ Hold sang Release. Mình chọn theo precision trên calib, nhưng thừa nhận đây có thể là "tối ưu cho cổng". Ở 0,7 (P/R gần như tương đương) kết luận vẫn là Hold với 29/202 = 14,4 %.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn thật.** Lệnh `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt) cho kết quả:
`kết luận: REWORK — còn 6 lỗi bắt buộc (mã: MISSING_LABEL_FILE)`, **mã thoát 20**, `Bước 5/5 KHÔNG xuất`, và thư mục `work/exp_nb` **không được tạo**.
Lý do: 6 ảnh thiếu file nhãn mà không có người gán. Máy không bịa nhãn, và `gate.decide` không bao giờ bỏ qua lỗi ERROR, kể cả với ảnh nằm trong `reviewed`.

**(b) Điểm mù (có bằng chứng):**
1. **Ảnh trùng biến đổi mạnh**: img_0084 và img_0074 (cùng cảnh người cầm ô, chỉ cắt khác) có pHash = 16 > 12, nên vẫn **lọt vào** `clean_v1`. Nâng ngưỡng lên 16 thì lại loại nhầm 2 cặp khác nhau (§3), nên một ngưỡng pHash duy nhất không giải quyết được cả hai.
2. **Box lỏng nhẹ**: ở `conf = 0.8`, recall LOOSE_BOX trên calib chỉ là **0,20** (1/5). Box lệch nhẹ (IoU vẫn ≥ 0,5) không bị cờ, và LOOSE_BOX lại là cờ có trọng số thấp (1,5) trong hàng đợi.
3. **Vật thiếu nhãn mà model không tự tin**: dự đoán có conf < 0,8 bị bỏ qua, nên trên calib sót 2/8 lỗi MISSING. Vật nhỏ hơn 600 px² cũng không bao giờ bị cờ.
4. **Nhầm lớp mà model lớn cũng nhầm** (ví dụ truck/bus nhìn từ xa): `second_opinion` chỉ so với một model, nên khi model sai cùng kiểu với người gán thì không có cờ nào (calib: 1/6 WRONG_CLASS bị sót).
5. **EMPTY_LABEL**: 6 ảnh vẫn có file nhãn rỗng ở v3. Máy không phân biệt được "ảnh nền" với "quên gán".

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Chưa chạy.** Bài này tập trung hoàn thành phần lõi (4 module, calibration, pipeline, báo cáo) trước. Phần train trên Colab (yolo26n, 2 seed × 20 epoch) sẽ làm sau nếu còn thời gian. Hiện chưa có số liệu nên không đưa ra kết luận "clean tốt hơn raw".

## 6. Quyết định và ghi chú AI
**(a) Quyết định:** **Release** `clean_v1` (202 ảnh), kèm hai điều kiện ghi rõ:
- 16 ảnh còn cảnh báo chưa duyệt (chủ yếu FALSE_POS) nên được duyệt ở vòng sau.
- Release phụ thuộc vào `conf = 0.8`. Nếu guideline của dự án ưu tiên recall lỗi thiếu box (`conf ≤ 0.7`), kết luận sẽ là **Hold** và cần thêm khoảng 22–40 lượt duyệt.

Quyết định sẽ đổi thành Hold/Rework nếu: duyệt lại phát hiện các cặp trùng còn sót (vd. img_0084/img_0074) nằm ở cả train lẫn val, hoặc nếu người gán để lại file nhãn trống hoặc hỏng.

**(b) AI:** Có dùng **Claude Code** (agent đọc được repo) để viết 4 module và hỗ trợ chọn ngưỡng. Nhật ký theo mẫu `docs/PROMPTING.md`:

- **Prompt chính:**

  "Bạn đóng vai một kỹ sư QC dữ liệu, giúp mình tự động hoá khâu kiểm tra nhãn cho bộ dữ liệu YOLO (4 lớp: car, person, truck, bus) trong repo này. Mình cần viết 4 module còn để trống là label_format, second_opinion, autofix và gate. Mỗi file đã có docstring mô tả đầu vào, đầu ra và các lỗi cần báo, test nằm trong thư mục tests. Bạn đọc docstring và test trước rồi viết code bám theo đó. Lưu ý ngưỡng lấy từ config, không viết cứng, và không sửa dữ liệu gốc trong data/raw. Viết xong thì chạy pytest cho đến khi xanh hết, rồi giải thích lại cho mình cách từng module hoạt động."

  Prompt chỉ nêu ý chính; phần chi tiết để AI đọc thẳng từ docstring và test, đúng như `docs/PROMPTING.md` khuyên. So với 4 ý của slide: định dạng vào và kiểu kết quả nằm trong docstring/`common.py`; quy tắc kiểm nằm trong bảng mã lỗi của docstring; giới hạn an toàn là câu lưu ý về ngưỡng và data/raw.
- **AI trả gì:** 4 module chạy được ngay (79/79 test xanh). AI cũng tự đề xuất chạy pipeline trên lưới 4×4 giá trị thay vì chọn ngưỡng chỉ dựa vào bảng sweep.
- **Tôi kiểm bằng cách nào** (3 việc của `PROMPTING.md`):
  1. *Đọc code:* tìm chỗ xoá/ghi đè file, tìm `except: pass` nuốt lỗi, kiểm mọi ngưỡng đều đọc từ `cfg`.
  2. *Chạy trên lỗi đã biết:* `pytest`, `make calibrate` , `make run` 2 lần cho cùng kết quả, `--budget 0` ra Rework / mã 20 / không xuất.
  3. *Đối chiếu bằng mắt:* `viz_labels.py` trên img_0020, img_0070, mở các cặp ảnh bị coi là trùng.
- **Chỗ AI sai/thiếu và cách sửa:**
  - Nếu chỉ nhìn bảng sweep, lựa chọn "hiển nhiên" là `phash_dist = 16` (F1 calib cao nhất). Kiểm trên raw thì giá trị này loại 31 ảnh, trong đó có 2 cặp trùng **giả** (tay vợt ↔ người trượt tuyết, lướt sóng ↔ người ngồi bàn). Tôi chốt **12**.
  - Bản đầu của `label_format` gọi `ds.image_path` thừa (quét lại thư mục ảnh cho mỗi file nhãn). Khi đọc lại code tôi đã bỏ, vì `ds.image_size` đã trả `None` cho nhãn mồ côi.
  - Mục 6 của bản nháp REPORT đầu tiên viết rằng "AI đã chọn 16 rồi đổi". Câu này không đúng sự thật, nên tôi đã sửa lại thành "lựa chọn dễ sai" như trên.
- **Bài học về prompt:** prompt hiệu quả nhờ có đủ vai trò, bối cảnh, đặc tả vào/ra, ràng buộc an toàn và tiêu chí hoàn thành đo được (test, mã thoát). Phần ràng buộc (không sửa `data/raw`, không bịa nhãn, không viết cứng ngưỡng) quan trọng nhất, vì AI dễ "làm cho test xanh" bằng cách vi phạm chúng.

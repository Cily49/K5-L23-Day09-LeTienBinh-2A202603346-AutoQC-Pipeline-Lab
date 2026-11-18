"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line

CHECK = "label_format"


def run(ds, cfg):
    lf = cfg["label_format"]
    tol, min_px = lf["overflow_tol"], lf["min_box_px"]
    max_frac, dup_iou = lf["max_box_frac"], lf["dup_box_iou"]
    n_cls = len(ds.classes)
    issues = []

    for path in ds.label_files():
        stem = path.stem
        size = ds.image_size(stem)              # None nếu nhãn mồ côi / ảnh hỏng: bỏ qua BOX_TINY
        prev = []                                                      # (lớp, box) của các dòng hợp lệ phía trước
        for line_no, text in ds.read_label_lines(stem) or []:          # số dòng gốc, đã bỏ dòng trống

            def add(sev, code, msg, **extra):
                issues.append(Issue(CHECK, sev, code, stem, msg, line_no, extra))

            # 1) định dạng
            try:
                c, x, y, w, h = parse_yolo_line(text)
            except ValueError as e:
                add(ERROR, "BAD_FORMAT", f"dòng sai định dạng ({e}): {text.strip()!r}")
                continue
            # 2) id lớp
            if not 0 <= c < n_cls:
                add(ERROR, "BAD_CLASS_ID", f"id lớp {c} ngoài 0..{n_cls - 1}")
                continue
            # 3) kích thước
            if w <= 0 or h <= 0:
                add(ERROR, "BOX_ZERO", f"box rỗng w={w} h={h}")
                continue

            box = (x, y, w, h)
            # 4) tràn ảnh: mức vượt lớn nhất trong 4 mép
            overflow = max(0 - (x - w / 2), (x + w / 2) - 1, 0 - (y - h / 2), (y + h / 2) - 1)
            if overflow > tol:
                add(ERROR, "BOX_OUT_OF_IMAGE", f"box tràn ảnh {overflow:.4f} (> {tol})", overflow=round(overflow, 6))
            # 5) box lặp với một dòng trước đó, cùng lớp
            if any(pc == c and iou_xywh(pb, box) > dup_iou for pc, pb in prev):
                add(ERROR, "DUP_BOX", f"box trùng một box lớp {c} ở dòng trước (IoU > {dup_iou})")
            # 6) cảnh báo kích thước
            if size:
                short = min(w * size[0], h * size[1])
                if short < min_px:
                    add(WARN, "BOX_TINY", f"cạnh ngắn {short:.1f}px < {min_px}px", short_px=round(short, 2))
            if w * h > max_frac:
                add(WARN, "BOX_HUGE", f"box chiếm {w * h:.2f} ảnh (> {max_frac})", area_frac=round(w * h, 4))

            prev.append((c, box))
    return issues

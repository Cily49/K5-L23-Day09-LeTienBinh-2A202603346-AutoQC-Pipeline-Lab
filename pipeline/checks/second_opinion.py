"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa. Lỗi thiếu nhãn / nhầm lớp hợp lệ / box lỏng không bắt được bằng quy tắc cú pháp:
nhờ model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai) -> ảnh bị cờ vào hàng đợi người duyệt.

Với từng ảnh có preds và có file nhãn:
  P = dự đoán có conf >= conf VÀ diện tích px >= min_area_px
  Mỗi dự đoán r ∈ P, cands = nhãn (mọi lớp) có IoU(r, nhãn) >= iou_missing:
    (1) cands rỗng                                  -> POSSIBLE_MISSING
    (2) có cands cùng lớp: IoU tốt nhất < iou_loose  -> POSSIBLE_LOOSE_BOX, ngược lại không cờ
    (3) chỉ có cands khác lớp: IoU tốt nhất >= iou_match -> POSSIBLE_WRONG_CLASS, ngược lại không cờ (mơ hồ)
  Mỗi dòng nhãn hợp lệ: IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) < iou_missing -> POSSIBLE_FALSE_POS (line = số dòng)
Tất cả WARN. Thứ tự: theo ảnh, theo thứ tự dự đoán, rồi FALSE_POS theo số dòng.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds

CHECK = "second_opinion"


def _labels_with_lines(ds, stem):
    """[(số dòng gốc, lớp, (cx, cy, w, h))] của các dòng hợp lệ; dòng hỏng / id lạ bị bỏ qua (check khác đã báo)."""
    out = []
    for n, text in ds.read_label_lines(stem) or []:
        try:
            c, x, y, w, h = parse_yolo_line(text)
        except ValueError:
            continue
        if 0 <= c < len(ds.classes):
            out.append((n, c, (x, y, w, h)))
    return out


def run(ds, cfg, preds=None):
    so = cfg["second_opinion"]
    conf_min, min_area, fp_conf = so["conf"], so["min_area_px"], so["fp_conf"]
    iou_missing, iou_match, iou_loose = so["iou_missing"], so["iou_match"], so["iou_loose"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    names = ds.classes
    issues = []

    for img in ds.image_files():
        stem = img.stem
        if stem not in preds or ds.read_label_lines(stem) is None:
            continue
        size = ds.image_size(stem)
        if size is None:
            continue
        W, H = size
        labels = _labels_with_lines(ds, stem)
        rows = [(int(r[0]), float(r[1]), tuple(r[2:6])) for r in preds[stem]]

        # --- dự đoán của model -> MISSING / LOOSE_BOX / WRONG_CLASS
        for pc, conf, pbox in rows:
            if conf < conf_min or pbox[2] * W * pbox[3] * H < min_area:
                continue
            cands = [(iou_xywh(pbox, lbox), lc) for _, lc, lbox in labels]
            cands = [(iou, lc) for iou, lc in cands if iou >= iou_missing]
            pname = names[pc] if 0 <= pc < len(names) else str(pc)
            if not cands:
                issues.append(Issue(CHECK, WARN, "POSSIBLE_MISSING", stem,
                                    f"model thấy {pname} (conf {conf:.2f}) nhưng không có box nhãn nào khớp",
                                    extra={"conf": conf, "pred_class": pc}))
                continue
            same = [iou for iou, lc in cands if lc == pc]
            if same:
                best = max(same)
                if best < iou_loose:
                    issues.append(Issue(CHECK, WARN, "POSSIBLE_LOOSE_BOX", stem,
                                        f"box {pname} lệch/lỏng so với model (IoU {best:.2f} < {iou_loose})",
                                        extra={"conf": conf, "iou": round(best, 4)}))
                continue
            best_iou, best_cls = max(cands)
            if best_iou >= iou_match:
                lname = names[best_cls]
                issues.append(Issue(CHECK, WARN, "POSSIBLE_WRONG_CLASS", stem,
                                    f"nhãn là {lname}, model nói {pname} (conf {conf:.2f}, IoU {best_iou:.2f})",
                                    extra={"conf": conf, "iou": round(best_iou, 4),
                                           "label_class": best_cls, "pred_class": pc}))

        # --- dòng nhãn không khớp vật nào model thấy -> FALSE_POS
        fp_boxes = [pbox for _, conf, pbox in rows if conf >= fp_conf]
        for n, lc, lbox in labels:
            best = max((iou_xywh(lbox, pb) for pb in fp_boxes), default=0.0)
            if best < iou_missing:
                issues.append(Issue(CHECK, WARN, "POSSIBLE_FALSE_POS", stem,
                                    f"box {names[lc]} không khớp dự đoán nào của model (IoU tốt nhất {best:.2f})",
                                    n, {"iou": round(best, 4)}))
    return issues

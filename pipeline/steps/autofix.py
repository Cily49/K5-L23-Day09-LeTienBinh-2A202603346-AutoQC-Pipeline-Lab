"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file: ghi 1 dòng fix "needs_label" | "needs_label"   |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Nguyên tắc: gom lỗi dòng theo file rồi ghi lại MỘT lần (số dòng trong issues là số dòng gốc);
xoá dòng thắng cắt box; box cắt xong không còn diện tích thì xoá; không động tới WARN; idempotent.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
CLIP_CODES = {"BOX_OUT_OF_IMAGE"}
REMOVE_IMAGE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _is_remove_code(code):
    return code in REMOVE_IMAGE_CODES or code.startswith("DUP_IMAGE_")


def _fx(image, line, action, before, after):
    return {"image": image, "line": "" if line is None else line, "action": action, "before": before, "after": after}


def _clip(text):
    """Cắt box về [0,1]. Trả về dòng mới, hoặc None nếu box không còn diện tích."""
    c, x, y, w, h = parse_yolo_line(text)
    x1, x2 = max(0.0, x - w / 2), min(1.0, x + w / 2)
    y1, y2 = max(0.0, y - h / 2), min(1.0, y + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]
    fixes = []

    # 1) Xoá ảnh trước: các lỗi dòng của ảnh đã xoá sẽ được bỏ qua êm.
    removed = set()
    for i in errors:
        if not _is_remove_code(i.code) or i.image in removed:
            continue
        img, lab = ds.image_path(i.image), ds.label_path(i.image)
        if img is None and not lab.exists():
            continue                                   # đã xoá ở lần chạy trước -> idempotent
        if img is not None:
            img.unlink()
        if lab.exists():
            lab.unlink()
        removed.add(i.image)
        fixes.append(_fx(i.image, None, "remove_image", img.name if img else "(không có ảnh)", f"đã xoá ({i.code})"))

    # 2) Tên lớp trong data.yaml.
    if any(i.code == "BAD_CLASS_NAME" for i in errors) and ds.yaml_path.exists():
        data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        before = data.get("names")
        after = {k: name for k, name in enumerate(cfg["classes"])}
        if before != after:
            data["names"] = after
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append(_fx("data.yaml", None, "rename_classes", str(before), str(after)))

    # 3) Ảnh chưa có nhãn: máy không bịa nhãn, chỉ chuyển người gán (không tạo file rỗng).
    for stem in dict.fromkeys(i.image for i in errors if i.code == "MISSING_LABEL_FILE"):
        if stem not in removed and not ds.label_path(stem).exists():
            fixes.append(_fx(stem, None, "needs_label", "(không có file)", "chuyển người gán"))

    # 4) Lỗi dòng: gom theo file, ghi lại một lần.
    drop, clip = defaultdict(set), defaultdict(set)
    for i in errors:
        if i.line is None or i.image in removed:
            continue
        if i.code in DROP_CODES:
            drop[i.image].add(int(i.line))
        elif i.code in CLIP_CODES:
            clip[i.image].add(int(i.line))

    for stem in sorted(set(drop) | set(clip)):
        path = ds.label_path(stem)
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()   # giữ dòng trống -> số dòng gốc
        out, changed = [], False
        for n, text in enumerate(raw, 1):
            if n in drop[stem]:                        # xoá thắng cắt
                fixes.append(_fx(stem, n, "drop_line", text, "(xoá dòng)"))
                changed = True
                continue
            if n in clip[stem]:
                try:
                    new = _clip(text)
                except ValueError:                     # dòng hỏng mà chỉ bị báo tràn: an toàn nhất là xoá
                    new = None
                if new is None:
                    fixes.append(_fx(stem, n, "drop_line", text, "(xoá dòng: box nằm ngoài ảnh)"))
                    changed = True
                    continue
                if new != text:
                    fixes.append(_fx(stem, n, "clip_box", text, new))
                    changed = True
                out.append(new)
                continue
            out.append(text)
        if changed:
            path.write_text("".join(ln + "\n" for ln in out), encoding="utf-8")

    return fixes

"""CHECK THƯỞNG — Box có tỉ lệ khung (rộng/cao) bất thường so với các box CÙNG LỚP.

Test:  pytest tests/test_aspect_ratio.py -q
Chạy:  python -m pipeline.checks.aspect_ratio --data work/run/ds --out submission/reports/bonus_aspect_ratio.csv

Ý tưởng: mỗi lớp có hình dạng điển hình (person thường cao hơn rộng, bus/car thường rộng hơn cao).
Một box `person` bè ngang gấp 4 lần chiều cao, hay một box `car` cao vống, thường là nhầm lớp hoặc vẽ lệch,
mà `label_format` (chỉ kiểm cú pháp) không bắt được.

Cách tính (thống kê robust, không cần đáp án để chọn ngưỡng theo từng lớp):
  r      = (w * W) / (h * H)               tỉ lệ khung tính bằng PIXEL (ảnh không vuông nên không dùng toạ độ chuẩn hoá)
  x      = log(r)                          để "rộng gấp 2" và "cao gấp 2" đối xứng quanh 0
  với từng lớp có >= min_samples box:  med = median(x),  mad = median(|x - med|)
  z      = (x - med) / (1.4826 * max(mad, min_mad))       (1.4826 * MAD ≈ độ lệch chuẩn nếu phân bố chuẩn)
  |z| > z_thresh  ->  BOX_ASPECT_OUTLIER (WARN, Issue.line = số dòng gốc)

- `z_thresh = 3.5` là mốc kinh điển cho "modified z-score" (Iglewicz & Hoaglin).
- `min_mad` chặn chia cho 0 khi mọi box của một lớp gần như cùng hình dạng.
- Box chạm mép ảnh (cách mép < `edge_tol`) KHÔNG bị chấm: vật bị khung ảnh cắt cụt thì hình dạng méo là đương nhiên
  (trên lô raw, 2/3 cờ ban đầu là người bị cắt ở mép trên: img_0148, img_0159). Box này vẫn được dùng để tính median.
- Lớp có ít hơn `min_samples` box thì bỏ qua: quá ít mẫu để biết đâu là "bình thường".
- Chỉ CẢNH BÁO: box dị dạng có thể đúng (người nằm, xe nhìn chéo) -> người duyệt quyết định.
- Ngưỡng đọc từ `cfg["aspect_ratio"]` nếu có; `configs/pipeline.yaml` là file ghim nên mặc định nằm ở DEFAULTS.
"""
from __future__ import annotations

import argparse
import math
from statistics import median

from ..common import WARN, Issue, parse_yolo_line

CHECK = "aspect_ratio"
DEFAULTS = {"z_thresh": 3.5, "min_samples": 8, "min_mad": 0.05, "edge_tol": 0.01}


def _params(cfg):
    p = dict(DEFAULTS)
    p.update((cfg or {}).get("aspect_ratio", {}) or {})
    return p


def _collect(ds, edge_tol):
    """[(stem, số dòng gốc, lớp, log tỉ lệ khung px, chạm mép?)] của mọi dòng hợp lệ có ảnh đọc được."""
    rows = []
    for lf in ds.label_files():
        stem = lf.stem
        size = ds.image_size(stem)
        if size is None:
            continue
        W, H = size
        for n, text in ds.read_label_lines(stem) or []:
            try:
                c, x, y, w, h = parse_yolo_line(text)
            except ValueError:
                continue                          # dòng hỏng: label_format đã báo
            if not 0 <= c < len(ds.classes) or w <= 0 or h <= 0:
                continue
            edge = min(x - w / 2, y - h / 2, 1 - (x + w / 2), 1 - (y + h / 2)) < edge_tol
            rows.append((stem, n, c, math.log((w * W) / (h * H)), edge))
    return rows


def run(ds, cfg):
    p = _params(cfg)
    rows = _collect(ds, p["edge_tol"])
    by_cls = {}
    for _, _, c, x, _ in rows:
        by_cls.setdefault(c, []).append(x)

    stats = {}
    for c, xs in by_cls.items():
        if len(xs) < p["min_samples"]:
            continue
        med = median(xs)
        mad = median(abs(x - med) for x in xs)
        stats[c] = (med, 1.4826 * max(mad, p["min_mad"]))

    issues = []
    for stem, n, c, x, edge in rows:             # đã theo thứ tự file rồi số dòng
        if c not in stats or edge:
            continue
        med, scale = stats[c]
        z = (x - med) / scale
        if abs(z) > p["z_thresh"]:
            name = ds.classes[c]
            shape = "bè ngang" if z > 0 else "cao hẹp"
            issues.append(Issue(CHECK, WARN, "BOX_ASPECT_OUTLIER", stem,
                                f"box {name} {shape} bất thường: rộng/cao = {math.exp(x):.2f} "
                                f"(điển hình {math.exp(med):.2f}, z = {z:+.1f})",
                                n, {"class": name, "aspect": round(math.exp(x), 3),
                                    "typical": round(math.exp(med), 3), "z": round(z, 2)}))
    return issues


def main(argv=None):
    from ..common import ROOT, Dataset, load_config, write_issues_csv
    ap = argparse.ArgumentParser(description="Check thưởng: box có tỉ lệ khung bất thường theo lớp")
    ap.add_argument("--data", default="work/run/ds", help="thư mục dataset YOLO (images/, labels/)")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default="submission/reports/bonus_aspect_ratio.csv")
    a = ap.parse_args(argv)
    cfg = load_config(ROOT / a.config)
    ds = Dataset(ROOT / a.data, cfg["classes"])
    issues = run(ds, cfg)
    write_issues_csv(issues, ROOT / a.out)
    n_img = len(ds.image_files())
    print(f"ảnh đã quét: {n_img}; {len(issues)} box tỉ lệ khung bất thường "
          f"trên {len({i.image for i in issues})} ảnh -> {a.out}")
    for i in issues:
        print(f"  [WARN] {i.image}.txt:{i.line}  {i.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

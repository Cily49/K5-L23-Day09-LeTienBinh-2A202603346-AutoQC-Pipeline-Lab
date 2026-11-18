"""Test cho check thưởng aspect_ratio (do học viên viết thêm)."""
from pipeline.checks import aspect_ratio

# ảnh 640x480: box person "điển hình" rộng 0.1*640=64px, cao 0.3*480=144px -> rộng/cao ≈ 0.44
PERSON = "1 0.5 0.5 0.1 0.3"


def persons(k, extra=()):
    """k box person bình thường (hơi khác nhau cho có độ phân tán) + các dòng thêm."""
    base = [f"1 0.5 0.5 {0.09 + 0.005 * (i % 5):.3f} 0.3" for i in range(k)]
    return base + list(extra)


def codes(issues):
    return [(i.image, i.line) for i in issues if i.code == "BOX_ASPECT_OUTLIER"]


def test_normal_boxes_are_silent(make_ds, cfg):
    ds = make_ds(images={"a": 1}, labels={"a": persons(10)}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_wide_person_is_flagged_with_line_number(make_ds, cfg):
    ds = make_ds(images={"a": 1}, labels={"a": persons(10, ["1 0.5 0.5 0.6 0.1"])}, size=(640, 480))
    iss = aspect_ratio.run(ds, cfg)
    assert codes(iss) == [("a", 11)]
    assert iss[0].severity == "warn" and iss[0].extra["z"] > 0


def test_classes_are_judged_separately(make_ds, cfg):
    # 10 person cao hẹp + 10 car bè ngang: car bè ngang là BÌNH THƯỜNG với lớp car
    cars = [f"0 0.5 0.5 0.4 {0.10 + 0.005 * (i % 5):.3f}" for i in range(10)]
    ds = make_ds(images={"a": 1}, labels={"a": persons(10) + cars}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_too_few_samples_is_skipped(make_ds, cfg):
    ds = make_ds(images={"a": 1}, labels={"a": persons(3, ["1 0.5 0.5 0.6 0.1"])}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_identical_boxes_do_not_divide_by_zero(make_ds, cfg):
    ds = make_ds(images={"a": 1}, labels={"a": [PERSON] * 10 + ["1 0.5 0.5 0.6 0.1"]}, size=(640, 480))
    assert codes(aspect_ratio.run(ds, cfg)) == [("a", 11)]


def test_threshold_comes_from_config(make_ds, cfg):
    cfg["aspect_ratio"] = {"z_thresh": 1000}
    ds = make_ds(images={"a": 1}, labels={"a": persons(10, ["1 0.5 0.5 0.6 0.1"])}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_pixel_ratio_not_normalised_ratio(make_ds, cfg):
    # cùng dòng toạ độ chuẩn hoá, nhưng ảnh "b" rất dẹt (1280x240): box 0.1x0.3 thành 128x72 px -> bè ngang
    from tests.conftest import smooth_image
    ds = make_ds(images={"a": 1}, labels={"a": persons(10)}, size=(640, 480))
    smooth_image(2, (1280, 240)).save(ds.images_dir / "b.jpg", "JPEG")
    (ds.labels_dir / "b.txt").write_text(PERSON + "\n")
    assert codes(aspect_ratio.run(ds, cfg)) == [("b", 1)]


def test_broken_lines_and_blank_lines_keep_original_numbers(make_ds, cfg):
    ds = make_ds(images={"a": 1}, size=(640, 480))
    body = "\n".join(persons(10)) + "\n\ngarbage\n93 0.5 0.5 0.2 0.2\n1 0.5 0.5 0 0.3\n1 0.5 0.5 0.6 0.1\n"
    (ds.labels_dir / "a.txt").write_text(body)
    # dòng 1-10 bình thường, 11 trống, 12-14 hỏng (check khác báo), 15 bè ngang
    assert codes(aspect_ratio.run(ds, cfg)) == [("a", 15)]


def test_box_cut_by_image_border_is_not_flagged(make_ds, cfg):
    # cùng hình dạng bè ngang như ca bị cờ, nhưng chạm mép trên (y - h/2 = 0): vật bị cắt cụt -> bỏ qua
    ds = make_ds(images={"a": 1}, labels={"a": persons(10, ["1 0.5 0.05 0.6 0.1"])}, size=(640, 480))
    assert aspect_ratio.run(ds, cfg) == []


def test_missing_or_corrupt_image_is_skipped(make_ds, cfg):
    ds = make_ds(images={"a": 1, "bad": b"not an image"}, labels={"a": persons(10), "bad": ["1 0.5 0.5 0.6 0.1"]})
    assert aspect_ratio.run(ds, cfg) == []

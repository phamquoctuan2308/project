"""So sánh số đếm của hệ thống với số đếm tay (ground truth) của một video.

Ví dụ:
    py -m tools.eval_counting --pred outputs/test1_counts.csv --gt data/ground_truth/test1.csv
    py -m tools.eval_counting --pred data/ground_truth/test1_nguoi2.csv --gt data/ground_truth/test1.csv
(lệnh thứ hai so sánh 2 người đếm tay với nhau)

Ground truth có 2 dạng:
  - từng xe: time_s,class,direction -> tính thêm số xe bỏ sót / đếm thừa / nhầm loại
  - tổng   : class,count            -> chỉ so tổng theo lớp
Độ chính xác đếm theo công thức của báo cáo: (1 - |máy - tay| / tay) x 100%, tính cho tổng và từng lớp.
Nếu có outputs/<video>_summary.json, các lượt đếm tay ngoài khoảng hệ thống thực sự đếm
(lúc MOG2 học nền, sau --max-frames) được loại ra.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from train.data import ALL_CLASSES


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_events(rows):
    return [{"time_s": float(r["time_s"]), "class": r["class"].strip().lower(),
             "direction": (r.get("direction") or "").strip().lower()} for r in rows]


def match_events(gt, pred, tol=1.5, offset=0.0):
    """Ghép lượt đếm tay với lượt đếm máy theo thời gian (|Δt| <= tol, cùng chiều), ưu tiên cùng lớp.

    Trả về (các cặp (i_gt, j_pred), chỉ số gt bị bỏ sót, chỉ số pred đếm thừa).
    """
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if g["direction"] and p["direction"] and g["direction"] != p["direction"]:
                continue
            dt = abs(p["time_s"] - (g["time_s"] + offset))
            if dt <= tol:
                pairs.append((dt + (0 if g["class"] == p["class"] else tol / 2), i, j))
    used_g, used_p, matches = set(), set(), []
    for _, i, j in sorted(pairs):
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            matches.append((i, j))
    return matches, [i for i in range(len(gt)) if i not in used_g], [j for j in range(len(pred)) if j not in used_p]


def count_accuracy(machine, manual):
    """Công thức báo cáo, chặn dưới ở 0 khi sai lệch lớn hơn số thật."""
    return max(0.0, (1 - abs(machine - manual) / manual) * 100) if manual else None


def count_table(gt_counts, pred_counts):
    classes = [c for c in ALL_CLASSES if c != "other" and (gt_counts.get(c) or pred_counts.get(c))]
    rows = [{"class": c, "machine": pred_counts.get(c, 0), "manual": gt_counts.get(c, 0)} for c in classes]
    for r in rows:
        r["diff"] = r["machine"] - r["manual"]
        r["accuracy"] = count_accuracy(r["machine"], r["manual"])
    M, T = sum(r["machine"] for r in rows), sum(r["manual"] for r in rows)
    total = {"class": "TỔNG", "machine": M, "manual": T, "diff": M - T, "accuracy": count_accuracy(M, T)}
    per_class = [r["accuracy"] for r in rows if r["accuracy"] is not None]
    macro = sum(per_class) / len(per_class) if per_class else None
    return rows, total, macro


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="CSV từng xe do src.main ghi (hoặc CSV đếm tay khác)")
    p.add_argument("--gt", required=True, help="CSV đếm tay")
    p.add_argument("--summary", help="summary.json của lần chạy (mặc định tự tìm cạnh file --pred)")
    p.add_argument("--direction", default="both", help="chỉ đánh giá một chiều (down / up / left / right)")
    p.add_argument("--tol", type=float, default=1.5, help="sai lệch thời gian tối đa khi ghép lượt (giây)")
    p.add_argument("--offset", type=float, default=0.0, help="cộng vào thời gian đếm tay (bù phản xạ)")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    pred = to_events(read_csv(args.pred))
    gt_rows = read_csv(args.gt)
    per_event = bool(gt_rows) and "time_s" in gt_rows[0]
    summary_path = Path(args.summary) if args.summary else Path(args.pred.replace("_counts.csv", "_summary.json"))
    notes = []
    if args.direction != "both":
        pred = [e for e in pred if e["direction"] == args.direction]

    if per_event:
        gt = to_events(gt_rows)
        if args.direction != "both":
            gt = [e for e in gt if e["direction"] == args.direction]
        if summary_path.exists() and summary_path != Path(args.pred):
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            start, end = s["count_start_s"], s["count_end_s"]
            before = len(gt)
            gt = [e for e in gt if start <= e["time_s"] + args.offset <= end]
            if before != len(gt):
                notes.append(f"Bỏ {before - len(gt)} lượt đếm tay ngoài khoảng hệ thống đếm ({start}s - {end}s)")
        gt_counts = {c: sum(e["class"] == c for e in gt) for c in ALL_CLASSES}
    else:
        gt = None
        gt_counts = {r["class"].strip().lower(): int(r["count"]) for r in gt_rows}
        notes.append("Ground truth chỉ có tổng: không tính được bỏ sót / đếm thừa / nhầm loại")
    pred_counts = {c: sum(e["class"] == c for e in pred) for c in ALL_CLASSES}

    rows, total, macro = count_table(gt_counts, pred_counts)
    lines = [f"Máy: {args.pred}", f"Tay: {args.gt}", f"Chiều: {args.direction}", *notes, "",
             f"{'Lớp':<12}{'Máy':>8}{'Tay':>8}{'Sai lệch':>10}{'Chính xác':>12}"]
    for r in [*rows, total]:
        acc = f"{r['accuracy']:.1f}%" if r["accuracy"] is not None else "-"
        lines.append(f"{r['class']:<12}{r['machine']:>8}{r['manual']:>8}{r['diff']:>+10}{acc:>12}")
    lines.append(f"\nĐộ chính xác đếm tổng: {total['accuracy'] or 0:.1f}%   "
                 f"trung bình theo lớp: {macro if macro is not None else 0:.1f}%")
    result = {"pred": args.pred, "gt": args.gt, "direction": args.direction, "rows": rows, "total": total,
              "macro_accuracy": macro, "notes": notes}

    if gt is not None:
        matches, missed, extra = match_events(gt, pred, args.tol, args.offset)
        correct = sum(gt[i]["class"] == pred[j]["class"] for i, j in matches)
        confusion = {}
        for i, j in matches:
            if gt[i]["class"] != pred[j]["class"]:
                key = f"{gt[i]['class']} -> {pred[j]['class']}"
                confusion[key] = confusion.get(key, 0) + 1
        n_gt, n_pred = len(gt), len(pred)
        events = {"manual": n_gt, "machine": n_pred, "matched": len(matches), "correct_class": correct,
                  "wrong_class": len(matches) - correct, "missed": len(missed), "extra": len(extra),
                  "recall": len(matches) / n_gt if n_gt else None,
                  "precision": len(matches) / n_pred if n_pred else None,
                  "fully_correct_rate": correct / n_gt if n_gt else None, "confusion": confusion}
        result["events"] = events
        lines += ["", f"So khớp từng xe (|Δt| <= {args.tol}s):",
                  f"  đếm đúng xe và đúng loại : {correct}/{n_gt}"
                  f" ({(events['fully_correct_rate'] or 0) * 100:.1f}%)",
                  f"  đúng xe nhưng nhầm loại  : {events['wrong_class']}",
                  f"  bỏ sót (tay có, máy không): {len(missed)}",
                  f"  đếm thừa (máy có, tay không): {len(extra)}",
                  f"  recall {(events['recall'] or 0):.3f}   precision {(events['precision'] or 0):.3f}"]
        lines += [f"  nhầm {k}: {v}" for k, v in sorted(confusion.items(), key=lambda kv: -kv[1])]

    text = "\n".join(lines)
    print(text)
    name = Path(args.pred).stem.replace("_counts", "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_eval.txt").write_text(text + "\n", encoding="utf-8")
    (out / f"{name}_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã ghi {out / (name + '_eval.txt')}")


if __name__ == "__main__":
    main()

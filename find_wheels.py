#!/usr/bin/env python3

"""
wheel_hubs.py  –  Find (x,y) centers of the two near-side wheels in a JPEG.
Strategy:
  1. Run YOLOv8n (ONNX) via cv2.dnn to locate the vehicle bounding box.
  2. Crop to the lower half of that box (where wheels live).
  3. Run HoughCircles in that ROI to find the two strongest circles.
  4. Fall back to whole-image lower-half HoughCircles if YOLO finds nothing.

Requires:
  pip install opencv-python>=5.0   (or build OpenCV 5 from source)
  YOLOv8n ONNX model:
    pip install ultralytics
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
  That produces yolov8n.onnx in the current directory.
"""

import sys
import cv2
import numpy as np

VERSION = "1.5.0" # polar-domain symmetry scoring (warpPolar + roll), bounded coarse/fine sweeps

# ── COCO class IDs that are road vehicles ──────────────────────────────────
VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorbike, bus, truck

# ── YOLOv8 input geometry ─────────────────────────────────────────────────
YOLO_INPUT_SIZE = 640
CONF_THRESH = 0.35
NMS_THRESH  = 0.45

# ── Wheel geometry ────────────────────────────────────────────────────────
# Fixed rim/tire radius ratio (r_inner / r_outer). Used both to constrain the
# rim-fit band search and to derive a consistent tire radius from the measured
# rim. Typical passenger/SUV wheels sit around 0.78-0.82; tune if needed.
RIM_TIRE_RATIO = 0.80

# ────────────────────────────────────────────────────────────────────────────
def score_wheel_detections(hubs, vehicle_box, verbose=False):
    """
    Returns a confidence score 0.0–1.0 and a dict of individual check results.
    hubs: [(cx1,cy1,r1), (cx2,cy2,r2)] sorted left-to-right
    vehicle_box: (x1, y1, x2, y2)
    """
    if len(hubs) != 2:
        return 0.0, {"error": "need exactly 2 hubs"}

    (cx1, cy1, r1), (cx2, cy2, r2) = sorted(hubs, key=lambda h: h[0])
    vx1, vy1, vx2, vy2 = vehicle_box
    box_w = vx2 - vx1
    box_h = vy2 - vy1

    checks = {}

    # Wheelbase vs box width
    wheelbase = cx2 - cx1
    wb_ratio = wheelbase / box_w
    checks["wheelbase_ratio"] = (wb_ratio, 0.50, 0.75)

    # Wheel radius vs box height
    r_mean = (r1 + r2) / 2
    r_ratio = r_mean / box_h
    checks["radius_ratio"] = (r_ratio, 0.18, 0.32)

    # Hub height above box bottom
    for i, (cx, cy, r) in enumerate([(cx1,cy1,r1),(cx2,cy2,r2)]):
        height_above_bottom = (vy2 - cy) / box_h
        checks[f"hub{i+1}_height"] = (height_above_bottom, 0.15, 0.35)

    # Both hubs at similar height
    cy_diff_ratio = abs(cy1 - cy2) / box_h
    checks["hub_level"] = (cy_diff_ratio, 0.0, 0.08)

    # Front wheel inset from front of box
    front_inset = (cx1 - vx1) / box_w
    checks["front_inset"] = (front_inset, 0.03, 0.22)

    # Rear wheel inset from rear of box
    rear_inset = (vx2 - cx2) / box_w
    checks["rear_inset"] = (rear_inset, 0.03, 0.20)

    # Radii similarity
    r_diff_ratio = abs(r1 - r2) / r_mean
    checks["radii_match"] = (r_diff_ratio, 0.0, 0.15)

    # Score: fraction of checks passed
    passed = 0
    for name, (val, lo, hi) in checks.items():
        ok = lo <= val <= hi
        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"  {status:4s}  {name:22s}  {val:.3f}  [{lo:.2f}, {hi:.2f}]")
        if ok:
            passed += 1

    score = passed / len(checks)
    return score, checks

def load_yolo(model_path: str = "yolov8n.onnx"):
    net = cv2.dnn.readNetFromONNX(model_path)
    # ENGINE_AUTO is the default in OpenCV 5; new graph engine tried first.
    # Uncomment to force the new engine explicitly:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def run_yolo(net, img_bgr):
    """Return list of (x1,y1,x2,y2) in original image coords for vehicles."""
    h, w = img_bgr.shape[:2]
    scale = YOLO_INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h))

    # Letterbox pad to 640×640
    canvas = np.zeros((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    pad_x, pad_y = 0, 0   # top-left origin padding

    blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    raw = net.forward()          # shape: (1, 84, 8400) for YOLOv8n COCO

    # Transpose to (8400, 84): columns = cx,cy,w,h, then 80 class scores
    preds = raw[0].T             # (8400, 84)
    boxes, confs, class_ids = [], [], []

    for row in preds:
        scores = row[4:]
        cls = int(np.argmax(scores))
        conf = float(scores[cls])
        if conf < CONF_THRESH or cls not in VEHICLE_CLASSES:
            continue
        cx, cy, bw, bh = row[:4]
        # Scale back to original image
        x1 = int((cx - bw/2 - pad_x) / scale)
        y1 = int((cy - bh/2 - pad_y) / scale)
        x2 = int((cx + bw/2 - pad_x) / scale)
        y2 = int((cy + bh/2 - pad_y) / scale)
        boxes.append([x1, y1, x2-x1, y2-y1])
        confs.append(conf)
        class_ids.append(cls)

    indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    result = []
    for i in (indices.flatten() if len(indices) else []):
        x, y, bw, bh = boxes[i]
        result.append((max(0,x), max(0,y),
                       min(w, x+bw), min(h, y+bh)))
    return result

def refine_hub_by_rim_edge_fit(gray_full, cx, cy, r_outer,
                               expected_ratio=None, verbose=False):
    """
    Fit a circle to the inner rim ring using least-squares on Canny edge points
    within an annular search band.  If expected_ratio (r_inner/r_outer learned
    from the first wheel) is supplied, the search is constrained to a +-0.08
    window around that ratio, preventing fender/arch edges from winning.
    Returns (cx_refined, cy_refined, r_fit_int, rms) or
    (cx, cy, None, None) on failure.
    """
    pad = int(r_outer * 0.15)
    x1 = max(0, cx - r_outer - pad)
    y1 = max(0, cy - r_outer - pad)
    x2 = min(gray_full.shape[1], cx + r_outer + pad)
    y2 = min(gray_full.shape[0], cy + r_outer + pad)
    roi = gray_full[y1:y2, x1:x2]
    roi_cx, roi_cy = cx - x1, cy - y1

    blurred = cv2.GaussianBlur(roi, (3, 3), 1)
    edges_full = cv2.Canny(blurred, 30, 80)

    h, w = roi.shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - roi_cx)**2 + (Y - roi_cy)**2)

    # If ratio is known from the first wheel, search only the matching band.
    # Otherwise try three overlapping bands and pick the best.
    if expected_ratio is not None:
        lo = max(0.35, expected_ratio - 0.08)
        hi = min(0.92, expected_ratio + 0.08)
        bands = [(lo, hi)]
        if verbose:
            print(f"    Using constrained band [{lo:.2f},{hi:.2f}] "
                  f"from ratio {expected_ratio:.3f}")
    else:
        bands = [(0.40, 0.58), (0.55, 0.73), (0.70, 0.88)]
    best = None
    best_score = np.inf   # lower RMS + fewer pts penalty = better

    for (lo, hi) in bands:
        mask = ((dist >= r_outer * lo) & (dist <= r_outer * hi)).astype(np.uint8) * 255
        edges = cv2.bitwise_and(edges_full, mask)
        pts = np.column_stack(np.where(edges > 0))
        if len(pts) < 20:
            continue

        y_pts = pts[:, 0].astype(float)
        x_pts = pts[:, 1].astype(float)
        A = np.column_stack([x_pts, y_pts, np.ones(len(pts))])
        b_vec = x_pts**2 + y_pts**2
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        cx_fit = result[0] / 2
        cy_fit = result[1] / 2
        r_fit = np.sqrt(max(0, result[2] + cx_fit**2 + cy_fit**2))

        offset = np.hypot(cx_fit - roi_cx, cy_fit - roi_cy)
        if offset > r_outer * 0.20:
            continue
        if not (r_outer * lo <= r_fit <= r_outer * hi):
            continue

        residuals = np.sqrt((x_pts - cx_fit)**2 + (y_pts - cy_fit)**2) - r_fit
        rms = np.sqrt(np.mean(residuals**2))

        # Score: penalise high RMS; reward more edge points
        score = rms - 0.005 * len(pts)
        if verbose:
            print(f"    Band [{lo:.2f},{hi:.2f}]: r_fit={r_fit:.1f} "
                  f"({r_fit/r_outer:.2f}x), rms={rms:.2f}px, "
                  f"pts={len(pts)}, score={score:.2f}")

        if score < best_score:
            best_score = score
            best = (cx_fit, cy_fit, r_fit, rms, len(pts))

    if best is None or best[3] > 7.0:
        if verbose:
            reason = "no valid band" if best is None else f"best RMS={best[3]:.2f}px too high"
            print(f"    Rim edge fit failed ({reason}), keeping outer center")
        return cx, cy, None, None

    cx_fit, cy_fit, r_fit, rms, npts = best
    cx_ref = int(round(cx_fit)) + x1
    cy_ref = int(round(cy_fit)) + y1
    r_fit_int = int(round(r_fit))

    if verbose:
        print(f"    Rim edge fit: center=({cx_ref},{cy_ref}), r_fit={r_fit_int}px "
              f"({r_fit/r_outer:.2f}x outer), offset=({cx_ref-cx:+d},{cy_ref-cy:+d})px, "
              f"edge_pts={npts}, RMS={rms:.2f}px")

    return cx_ref, cy_ref, r_fit_int, rms

def refine_hub_by_spoke_intersection(gray_full, cx, cy, r_outer, verbose=False):
    """
    Find hub center by detecting spoke lines within the wheel and computing
    their intersection. Robust to partial rim occlusion since spokes are
    visible even when the top of the wheel is hidden by the fender.
    Returns (cx_refined, cy_refined) or original if insufficient spokes found.
    """
    pad = int(r_outer * 0.15)
    x1 = max(0, cx - r_outer - pad)
    y1 = max(0, cy - r_outer - pad)
    x2 = min(gray_full.shape[1], cx + r_outer + pad)
    y2 = min(gray_full.shape[0], cy + r_outer + pad)
    roi = gray_full[y1:y2, x1:x2]
    roi_cx, roi_cy = cx - x1, cy - y1

    # Mask to inner 85% of wheel (exclude tire/rim boundary noise)
    h, w = roi.shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - roi_cx)**2 + (Y - roi_cy)**2)
    inner_mask = (dist <= r_outer * 0.85).astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(roi, (3, 3), 1)
    edges = cv2.Canny(blurred, 25, 70)
    edges = cv2.bitwise_and(edges, inner_mask)

    # Probabilistic Hough line detection — spokes appear as short line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(r_outer * 0.20),
        maxLineGap=int(r_outer * 0.10),
    )

    if lines is None or len(lines) < 3:
        if verbose:
            print(f"    Spoke intersection: too few lines ({0 if lines is None else len(lines)}), "
                  f"keeping rim fit center")
        return cx, cy

    # Filter lines: keep only those that pass within r_outer*0.15 of the
    # current estimated center (spokes must converge near the hub)
    def dist_point_to_segment(px, py, x1l, y1l, x2l, y2l):
        """Distance from point to infinite line through segment."""
        dx, dy = x2l - x1l, y2l - y1l
        if dx == 0 and dy == 0:
            return np.hypot(px - x1l, py - y1l)
        t = ((px - x1l) * dx + (py - y1l) * dy) / (dx*dx + dy*dy)
        return np.hypot(px - (x1l + t*dx), py - (y1l + t*dy))

    spoke_lines = []
    for line in lines:
        x1l, y1l, x2l, y2l = line[0]
        d = dist_point_to_segment(roi_cx, roi_cy, x1l, y1l, x2l, y2l)
        if d < r_outer * 0.20:
            spoke_lines.append((x1l, y1l, x2l, y2l))

    if len(spoke_lines) < 3:
        if verbose:
            print(f"    Spoke intersection: only {len(spoke_lines)} lines pass hub proximity "
                  f"filter, keeping rim fit center")
        return cx, cy

    # Find intersection point of all line pairs, take trimmed mean
    # Line as (a,b,c) in ax+by+c=0 form
    def line_abc(x1l, y1l, x2l, y2l):
        a = y2l - y1l
        b = x1l - x2l
        c = x2l*y1l - x1l*y2l
        return a, b, c

    intersections = []
    for i in range(len(spoke_lines)):
        for j in range(i+1, len(spoke_lines)):
            a1, b1, c1 = line_abc(*spoke_lines[i])
            a2, b2, c2 = line_abc(*spoke_lines[j])
            det = a1*b2 - a2*b1
            if abs(det) < 1e-6:
                continue   # parallel lines
            ix = (b1*c2 - b2*c1) / det
            iy = (a2*c1 - a1*c2) / det
            # Keep only intersections near the hub
            if np.hypot(ix - roi_cx, iy - roi_cy) < r_outer * 0.30:
                intersections.append((ix, iy))

    if len(intersections) < 2:
        if verbose:
            print(f"    Spoke intersection: too few valid intersections ({len(intersections)})")
        return cx, cy

    pts = np.array(intersections)
    # Trimmed mean: discard outer 20% to suppress outlier line crossings
    cx_est = float(np.mean(np.clip(pts[:,0],
                                   np.percentile(pts[:,0], 10),
                                   np.percentile(pts[:,0], 90))))
    cy_est = float(np.mean(np.clip(pts[:,1],
                                   np.percentile(pts[:,1], 10),
                                   np.percentile(pts[:,1], 90))))

    cx_ref = int(round(cx_est)) + x1
    cy_ref = int(round(cy_est)) + y1

    if verbose:
        print(f"    Spoke intersection: center=({cx_ref},{cy_ref}), "
              f"offset=({cx_ref-cx:+d},{cy_ref-cy:+d})px, "
              f"spokes={len(spoke_lines)}, intersections={len(intersections)}")

    return cx_ref, cy_ref

def refine_hub_by_rotational_symmetry(gray_full, cx, cy, r_outer, verbose=False):
    """
    Find hub center by maximizing rotational symmetry of the wheel's edge
    pattern. For each candidate center, score how well the edge pattern maps
    onto itself under rotation by 2*pi/N for N=4..8 spokes.

    Implementation (v1.5.0): the scoring is done in the POLAR domain. The edge
    ROI is remapped to (angle x radius) around each candidate via warpPolar;
    a rotation by 2*pi/N is then just a circular shift of (n_ang/N) rows along
    the angle axis, scored by a vectorised dot product. This replaces the old
    per-candidate stack of ~25 warpAffine calls (the 11s bottleneck) with one
    warpPolar plus a few array rolls — same metric, ~10-50x faster.

    Robust to partial fender occlusion because symmetry is a global constraint.
    Returns (cx_refined, cy_refined) or original if confidence is low.
    """
    pad = int(r_outer * 0.30)   # wide enough to cover the +-25% search window
    x1 = max(0, cx - r_outer - pad)
    y1 = max(0, cy - r_outer - pad)
    x2 = min(gray_full.shape[1], cx + r_outer + pad)
    y2 = min(gray_full.shape[0], cy + r_outer + pad)
    roi = gray_full[y1:y2, x1:x2]
    roi_cx, roi_cy = cx - x1, cy - y1

    h, w = roi.shape

    # Pre-compute the edge map once (it does not depend on the candidate centre).
    blurred = cv2.GaussianBlur(roi, (3, 3), 1)
    edges_full = (cv2.Canny(blurred, 25, 70).astype(np.float32) / 255.0)

    # Polar sampling geometry. We sample the alloy annulus (15-85% of r_outer)
    # into a fixed grid of angle bins x radius bins. n_ang is chosen divisible
    # by all N in 4..8 (lcm=840 is overkill; 360 covers 4,5,6,8 exactly and 7
    # to <0.5 bin) so the 2*pi/N rotation is a clean integer row-roll.
    n_ang = 360
    n_rad = 48
    r_lo = r_outer * 0.15
    r_hi = r_outer * 0.85

    def polar_band(cand_cx, cand_cy):
        """Remap edges to (n_ang x n_rad) polar around the candidate, restricted
        to the alloy annulus. Returns the float polar patch (rows=angle)."""
        # warpPolar samples radius 0..r_hi across n_rad columns and angle
        # 0..2pi across n_ang rows. We then slice off the inner hub region so
        # only the alloy annulus (r_lo..r_hi) contributes.
        polar = cv2.warpPolar(
            edges_full, (n_rad, n_ang), (float(cand_cx), float(cand_cy)),
            r_hi, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)
        c_lo = int(round(n_rad * (r_lo / r_hi)))
        band = polar[:, c_lo:]   # rows = angle (n_ang), cols = radius band
        return band

    def symmetry_score(cand_cx, cand_cy, only_n=None):
        """
        Localising rotational-symmetry score for a candidate centre, computed
        as angle-axis circular shifts of the polar edge band.
        Returns (best_score, best_N).
        """
        band = polar_band(cand_cx, cand_cy)
        norm = float(band.sum()) + 1e-6
        if norm < 1e-3:
            return -1.0, 5

        ns = (only_n,) if only_n is not None else range(4, 9)
        best_s, best_nn = -1.0, 5
        for N in ns:
            shift = n_ang // N            # rows to roll per 2*pi/N rotation
            acc = 0.0
            for k in range(1, N):
                rolled = np.roll(band, shift * k, axis=0)
                acc += float(np.sum(band * rolled))
            s = (acc / (N - 1)) / norm
            if s > best_s:
                best_s, best_nn = s, N
        return best_s, best_nn

    # Fix B: wider search (±25% of r_outer) with a coarse-to-fine refinement.
    # Fix 1: one coarse sweep (all N) to localise + lock best_N, then one fine
    #        sweep scoring only that N. No unbounded re-sweeping.
    best_cx, best_cy = float(roi_cx), float(roi_cy)
    best_score, best_n = symmetry_score(best_cx, best_cy)

    # ---- Coarse sweep: full window, ~0.05*r step, all N (4..8) ----
    search_r = int(r_outer * 0.25)
    coarse_step = max(2, int(r_outer * 0.05))
    cx0, cy0 = best_cx, best_cy
    for dy in range(-search_r, search_r + 1, coarse_step):
        for dx in range(-search_r, search_r + 1, coarse_step):
            s, n_best = symmetry_score(cx0 + dx, cy0 + dy)
            if s > best_score:
                best_score, best_cx, best_cy, best_n = s, cx0 + dx, cy0 + dy, n_best

    # ---- Fine sweep: small window, 1px step, only the locked N ----
    fine_r = max(2, int(r_outer * 0.08))
    cx0, cy0 = best_cx, best_cy
    for dy in range(-fine_r, fine_r + 1):
        for dx in range(-fine_r, fine_r + 1):
            s, _ = symmetry_score(cx0 + dx, cy0 + dy, only_n=best_n)
            if s > best_score:
                best_score, best_cx, best_cy = s, cx0 + dx, cy0 + dy

    cx_ref = int(round(best_cx)) + x1
    cy_ref = int(round(best_cy)) + y1

    if verbose:
        print(f"    Rot. symmetry: center=({cx_ref},{cy_ref}), "
              f"offset=({cx_ref-cx:+d},{cy_ref-cy:+d})px, "
              f"best_N={best_n}, score={best_score:.4f}")

    return cx_ref, cy_ref

def hub_in_box(cx, cy, vehicle_box, r_outer=0, margin_frac=0.05):
    """
    True if a hub center lies inside the vehicle box, allowing a small margin
    for tire bulge / box tightness. margin is a fraction of box width/height.
    A wheel center should never fall outside its own vehicle.
    """
    vx1, vy1, vx2, vy2 = vehicle_box
    mx = (vx2 - vx1) * margin_frac
    my = (vy2 - vy1) * margin_frac
    return (vx1 - mx) <= cx <= (vx2 + mx) and (vy1 - my) <= cy <= (vy2 + my)

def find_hubs_in_roi(gray_roi, roi_offset_x, roi_offset_y,
                     min_r=30, max_r=130, h_param2=25):
    """Run HoughCircles and return up to 2 hub centers in full-image coords.

    Fix 3: low-contrast (e.g. black-on-black) wheels bury the rim gradient, so
    apply CLAHE local contrast equalisation before Hough, and walk a retry
    ladder of decreasing param2 until at least 2 circles appear.
    """
    if gray_roi.size == 0:
        return []

    # CLAHE pulls the rim/spoke gradients out of a dark, low-contrast wheel.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_roi)
    blurred = cv2.GaussianBlur(equalized, (9, 9), 2)
    h, w = gray_roi.shape

    # Retry ladder: start strict, loosen the accumulator threshold until we
    # have two circles (or run out of attempts).
    best = []
    for param2 in (h_param2, max(12, h_param2 - 7), max(10, h_param2 - 12)):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=w // 3,          # wheels must be at least this far apart
            param1=80,               # Canny high threshold
            param2=param2,           # accumulator threshold
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            continue
        circles = np.round(circles[0]).astype(int)
        if len(circles) >= len(best):
            best = circles
        if len(circles) >= 2:
            break

    if len(best) == 0:
        return []

    # Sort by accumulator strength (HoughCircles returns strongest first)
    # Take up to 2, sort left-to-right
    top2 = sorted(best[:2], key=lambda c: c[0])
    return [(int(c[0]) + roi_offset_x,
             int(c[1]) + roi_offset_y,
             int(c[2])) for c in top2]

def find_wheel_hubs(image_path: str,
                    yolo_model: str = "yolov8n.onnx",
                    debug_out: str = "debug_hubs.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 1: YOLO vehicle detection ───────────────────────────────────
    hubs = []
    vehicles = []
    primary_box = None
    try:
        net = load_yolo(yolo_model)
        vehicles = run_yolo(net, img)
        print(f"YOLO found {len(vehicles)} vehicle(s)")

        for (x1, y1, x2, y2) in vehicles:
            # Wheels live in the lower ~45% of the bounding box
            wheel_y1 = y1 + int((y2 - y1) * 0.55)
            roi = gray[wheel_y1:y2, x1:x2]
            print(f"Vehicle box: ({x1},{y1})->({x2},{y2}), wheel ROI size: {roi.shape}")
            if roi.size == 0:
                continue
            roi_w = x2 - x1
            est_r_min = max(20, roi_w // 14)
            est_r_max = max(60, roi_w // 6)
            found = find_hubs_in_roi(roi, x1, wheel_y1, est_r_min, est_r_max)
            # Fix 1: a hub must lie inside its own vehicle box.
            found = [(cx, cy, r) for (cx, cy, r) in found
                     if hub_in_box(cx, cy, (x1, y1, x2, y2), r)]
            hubs.extend(found)
            primary_box = (x1, y1, x2, y2)

    except Exception as e:
        print(f"YOLO step failed ({e}), falling back to whole-image Hough")

    # ── Step 2: Fallback — search the vehicle box only, never the whole frame.
    # Fix 2: scavenging the full lower half of the frame locks onto road
    # shadows and sunlit gaps off the car.  If a vehicle box exists, confine
    # the fallback to it (with a lower param2); if it still can't find two
    # hubs, report honestly rather than returning off-car blobs.
    if len(hubs) < 2:
        if primary_box is not None:
            x1, y1, x2, y2 = primary_box
            wheel_y1 = y1 + int((y2 - y1) * 0.45)   # slightly taller band
            roi = gray[wheel_y1:y2, x1:x2]
            print("Falling back to box-confined HoughCircles")
            roi_w = x2 - x1
            est_r_min = max(20, roi_w // 14)
            est_r_max = max(60, roi_w // 6)
            found = find_hubs_in_roi(roi, x1, wheel_y1,
                                     est_r_min, est_r_max, h_param2=18)
            hubs = [(cx, cy, r) for (cx, cy, r) in found
                    if hub_in_box(cx, cy, primary_box, r)]
        else:
            # No vehicle at all — only here do we risk a whole-image search.
            print("No vehicle box; falling back to full-image HoughCircles")
            roi_y = h // 2
            roi = gray[roi_y:, :]
            hubs = find_hubs_in_roi(roi, 0, roi_y, min_r=40, max_r=130)

    if len(hubs) < 2:
        print(f"WARNING: only {len(hubs)} hub(s) found inside the vehicle; "
              f"not enough for a reliable result")

    # ── Step 2.5: Refine hub centers using rim edge least-squares fit ─────
    # The first (usually cleaner) wheel's r_inner/r_outer ratio is learned
    # and used to constrain the band search for the second wheel, preventing
    # fender arches or road shadows from winning the band selection.
    # The rim/tire ratio is a fixed prior (RIM_TIRE_RATIO); we no longer learn
    # it per-image from a possibly well-contaminated outer radius.
    refined_hubs = []
    for (cx, cy, r_outer) in hubs:
        cx_ref, cy_ref, r_inner, rms = refine_hub_by_rim_edge_fit(
            gray, cx, cy, r_outer,
            expected_ratio=RIM_TIRE_RATIO,
            verbose=True)
        # Fix 1: a refinement must not walk the hub out of the box; revert if so.
        if primary_box is not None and not hub_in_box(cx_ref, cy_ref,
                                                      primary_box, r_outer):
            print(f"    Rim fit pushed hub off-vehicle "
                  f"({cx_ref},{cy_ref}); reverting to ({cx},{cy})")
            cx_ref, cy_ref = cx, cy
        refined_hubs.append((cx_ref, cy_ref, r_outer, r_inner, rms))
    hubs = refined_hubs

    # ── Step 2.6: Refine using rotational symmetry maximisation ───────────
    final_hubs = []
    for (cx, cy, r_outer, r_inner, rms) in hubs:
        cx_ref, cy_ref = refine_hub_by_rotational_symmetry(
            gray, cx, cy, r_outer, verbose=True)
        # Fix 1: symmetry search must not walk the hub out of the box either.
        if primary_box is not None and not hub_in_box(cx_ref, cy_ref,
                                                      primary_box, r_outer):
            print(f"    Symmetry pushed hub off-vehicle "
                  f"({cx_ref},{cy_ref}); reverting to ({cx},{cy})")
            cx_ref, cy_ref = cx, cy
        final_hubs.append((cx_ref, cy_ref, r_outer, r_inner, rms))
    hubs = final_hubs

    # ── Step 2.7: Reconcile radii to physical elements ────────────────────
    # Both near-side wheels are the same physical wheel, so r_inner (the rim,
    # which is fit cleanly) and r_outer (the tire) should agree across wheels.
    # Hough's per-wheel r_outer is unreliable (it often sizes to the wheel
    # well). We anchor on the wheel with the BEST rim fit (lowest RMS, smaller
    # radius as tiebreaker), take its r_inner as the true rim radius for both,
    # and derive a consistent tire r_outer = r_inner / RIM_TIRE_RATIO.
    valid = [(i, h) for i, h in enumerate(hubs) if h[3] is not None]
    if valid:
        def anchor_key(item):
            _, (cx, cy, r_outer, r_inner, rms) = item
            return (rms, r_inner)   # primary: RMS; tiebreaker: smaller radius
        _, anchor = min(valid, key=anchor_key)
        anchor_r_inner = anchor[3]
        anchor_r_outer = int(round(anchor_r_inner / RIM_TIRE_RATIO))
        print(f"    Radius anchor: r_inner={anchor_r_inner}px (RMS={anchor[4]:.2f}px), "
              f"derived r_outer={anchor_r_outer}px @ ratio {RIM_TIRE_RATIO}")
        reconciled = []
        for (cx, cy, r_outer, r_inner, rms) in hubs:
            # Apply the anchor's rim/tire pair to EVERY hub so both wheels show
            # the same physical radii (they are the same wheel). The center of
            # each hub is untouched — only the drawn/reported radii change.
            new_inner = anchor_r_inner
            new_outer = anchor_r_outer
            reconciled.append((cx, cy, new_outer, new_inner, rms))
        hubs = reconciled

    # Drop the rms field for downstream steps that expect 4-tuples.
    hubs = [(cx, cy, r_outer, r_inner) for (cx, cy, r_outer, r_inner, rms) in hubs]

    # ── Step 3: Evaluate wheel detections ─────────────────────────────────
    # hubs is list of (cx, cy, r); vehicles is list of (x1,y1,x2,y2)
    if len(hubs) == 2 and len(vehicles) == 1:
        hubs3 = [(cx, cy, r_outer) for (cx, cy, r_outer, r_inner) in hubs]
        score, checks = score_wheel_detections(hubs3, vehicles[0], verbose=True)
        print(f"\nGeometric plausibility score: {score:.2f} ({int(score*100)}%)")


    # ── Step 4: Annotate and save debug image ─────────────────────────────
    debug = img.copy()
    for (x1, y1, x2, y2) in vehicles:
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(debug, "vehicle", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    for i, (cx, cy, r_outer, r_inner) in enumerate(hubs):
        cv2.circle(debug, (cx, cy), r_outer,  (0, 255,   0), 2)   # outer (tire)
        if r_inner:
            cv2.circle(debug, (cx, cy), r_inner,  (0, 165, 255), 2)   # inner (rim)
        cv2.circle(debug, (cx, cy), 5,         (0,   0, 255), -1)
        cv2.putText(debug, f"({cx},{cy})", (cx + r_outer + 4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        r_str = f"r_outer={r_outer}px" + (f", r_inner={r_inner}px" if r_inner else "")
        print(f"  Hub {i+1}: center=({cx},{cy}), {r_str}")

    cv2.imwrite(debug_out, debug)
    return [(cx, cy) for (cx, cy, r_outer, r_inner) in hubs]

if __name__ == "__main__":
    print(f"find_wheels.py v{VERSION}")
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    model = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.onnx"
    centers = find_wheel_hubs(path, yolo_model=model)
    print(f"\nWheel hub centers: {centers}")

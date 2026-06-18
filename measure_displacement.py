#!/usr/bin/env python3
"""
measure_displacement.py

Estimate the signed horizontal displacement of a moving object (eg car, bike) between two frames captured a known time apart.

method 1: Keypoint tracking
-------------------------------
1. Detect ORB keypoints in frame 1, match into frame 2, and filter matches by Lowe ratio test.
2. RANSAC for a pure-translation model (dx = constant) to reject outliers and get a robust dx estimate.

method 2: Template matching
----------------------------------
1. Absolute difference -> threshold -> morphology to isolate the changed region.
2. Drop the cast shadow on the road by ignoring everything below a vertical
   cutoff (configurable).
3. Keep the largest connected component as the object mask, take its bounding
   box in frame 1 as a template.
4. Slide that template over frame 2 (cv2.matchTemplate, TM_CCOEFF_NORMED) and
   take the peak as the object's new position. dx = peak_x - template_x.
5. Sub-pixel refinement by parabolic fit around the correlation peak.
6. Report signed dx as a fraction of frame width (right = +, left = -).


Version 1.8.1: expand template area relative to EXIF BBox
Version 1.2.0: reads camera-subject distance from EXIF SubjectDistance (frame 1)
when --distance is not supplied.
Version 1.1.0: adds optional 2-pass refinement that shrinks the template onto
cells whose motion agrees with the dominant vector, excluding background.
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ---- Named constants -------------------------------------------------------
VERSION = "1.8.8 2026-06-17"
FRAME_DT_MS = 50.0          # time between the two frames, milliseconds

DIFF_THRESH = 25            # grayscale diff threshold (0-255)
MORPH_KERNEL = 7            # px, closing/opening kernel for mask cleanup
SHADOW_CUTOFF_FRAC = 0.75  # ignore diff below this fraction of frame height
BBOX_PAD = 6               # px, shrink template inward to avoid mask edges

SEARCH_DY = 40             # px, vertical search slack (motion is ~horizontal)
MIN_PEAK = 0.60            # minimum normalized-correlation peak to trust result

MAX_PLAUSIBLE_MPH = 70.0   # cap matched displacement to this speed when
                           # distance is known
MAX_DX_FRAC = 0.50         # and never beyond this fraction of frame width
EDGE_MARGIN_PX = 4         # template/match boxes must stay this far from edges

# ---- Template-size safeguards ----------------------------------------------
BBOX_MIN_FRAC = 0.6        # template width must be >= this * EXIF BBox width;
                           # else expand the template toward the EXIF BBox
REFINE_MIN_KEEP_FRAC = 0.6  # if no EXIF BBox, refinement may not shrink the
                            # template below this fraction of the pass-1 width

# ---- Keypoint cross-check (verification only, never overrides template) ----
KP_FEATURES = 8000         # max ORB features inside the box. At the old 2000
                           # cap, well-textured boxes (e.g. a car with roof
                           # rack, trim, wheels) routinely hit the cap *before*
                           # ranking reached weaker-but-still-usable corners,
                           # discarding real candidates purely by count, not
                           # contrast. Raising this (not lowering fastThreshold
                           # or edgeThreshold, which an empirical sweep showed
                           # have negligible effect once nfeatures isn't the
                           # bottleneck) roughly doubled-to-quadrupled the
                           # number of points surviving to the final inlier
                           # set in testing, at a ~10ms/frame detection cost.
KP_RATIO = 0.75            # Lowe ratio test
KP_MIN_INLIERS = 5         # need at least this many to report a result
KP_TOL_FRAC = 0.05         # agree if |kp_dx - tmpl_dx| within this frac of |dx|
KP_TOL_PX = 5.0            # ...or this many px, whichever is larger
KP_MAX_CV = 0.2            # agree also requires inlier dx cv below this
KP_MIN_X_SPREAD = 30.0     # px; inlier frame-1 x range (max-min) must be at
                           # least this wide, else the translation estimate
                           # is poorly constrained (e.g. all points on one
                           # small, possibly non-rigid object) -> 'low_spread'
KP_SUBPIX_WIN = 5          # cv2.cornerSubPix half-window size (px)
KP_SUBPIX_ZERO_ZONE = 2    # cv2.cornerSubPix half-size of dead zone (px)
KP_DECLUSTER_DIST = 8.0    # px; min separation (in frame-1) between inlier
                           # points kept for statistics. Several ORB keypoints
                           # often land on the same physical corner (e.g. a
                           # mirror, headlight, wheel-arch shadow); their
                           # localization errors are correlated, not
                           # independent, so clustering inflates the apparent
                           # n and understates the true uncertainty.

VIZ_MARGIN_FRAC = 0.25     # --viz crop: extra fraction of bbox height above/below
VIZ_MIN_MARGIN_PX = 20     # --viz crop: margin never shrinks below this many px,
                           # regardless of template height or info-box size
VIZ_INFO_BOX_CLEARANCE_PX = 12  # --viz crop: required vertical gap below the
                                # info box within the top panel, so the box
                                # never touches/clips the template rectangle
                                # or the seam

# ---- Per-keypoint velocity plot (--plot) -----------------------------------
PLOT_DPI = 110

# ---- Two-pass refinement ---------------------------------------------------
REFINE_CELLS_ACROSS = 10   # target number of grid cells across bbox width
REFINE_MIN_CELL = 40       # px, minimum cell size
REFINE_CELL_PEAK = 0.40    # min per-cell correlation peak to count a cell
REFINE_DX_TOL = 0.15       # cell dx must be within +/-15% of dominant dx
REFINE_DY_TOL = 6          # px, cell dy must be within this of dominant dy
REFINE_MIN_CELLS = 4       # min surviving cells; else fall back to pass 1

# ---- Optical constants (Sony IMX296, 12.0 mm lens) -------------------------
FOCAL_LENGTH_MM = 12.0      # lens focal length
SENSOR_WIDTH_MM = 5.023     # 1456 px x 3.45 um = 5.023 mm
SENSOR_WIDTH_PX = 1456      # horizontal pixels covering the sensor width
MPS_TO_MPH = 2.2369362921   # m/s -> mph


def object_bbox(g1, g2, shadow_cutoff_px):
    """Bounding box (x, y, w, h) of the largest moving region, shadow excluded."""
    diff = cv2.absdiff(g1, g2)
    _, m = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    m[shadow_cutoff_px:, :] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    bw = stats[largest, cv2.CC_STAT_WIDTH]
    bh = stats[largest, cv2.CC_STAT_HEIGHT]
    return (x, y, bw, bh)


def subpixel_peak(corr, px, py):
    """Parabolic sub-pixel refinement of the correlation peak in x."""
    h, w = corr.shape
    if 0 < px < w - 1:
        a, b, c = corr[py, px - 1], corr[py, px], corr[py, px + 1]
        denom = (a - 2 * b + c)
        if abs(denom) > 1e-9:
            return 0.5 * (a - c) / denom
    return 0.0


def match_template(g1, g2, tx, ty, tw, th, max_dx=None):
    """Match a template (from g1 at tx,ty,tw,th) within g2.

    If max_dx is given, restrict the search to candidate positions whose
    horizontal displacement |dx| <= max_dx (px), so implausibly large jumps
    are excluded and the best in-band peak is chosen instead.

    Returns (dx, dy, peak, px_ref, py_abs) where px_ref is sub-pixel refined
    x in g2 and py_abs is the matched y in g2.
    """
    h = g2.shape[0]
    template = g1[ty:ty + th, tx:tx + tw]
    sy0 = max(0, ty - SEARCH_DY)
    sy1 = min(h, ty + th + SEARCH_DY)
    search = g2[sy0:sy1, :]
    if search.shape[0] < th or search.shape[1] < tw:
        return None
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)

    if max_dx is not None:
        # res column c corresponds to candidate top-left x=c, i.e. dx=c-tx.
        # Mask out-of-band columns so the peak is taken only within |dx|<=max_dx.
        ncols = res.shape[1]
        lo = max(0, int(np.floor(tx - max_dx)))
        hi = min(ncols - 1, int(np.ceil(tx + max_dx)))
        if hi < lo:
            return None
        masked = np.full_like(res, -1.0)
        masked[:, lo:hi + 1] = res[:, lo:hi + 1]
        res = masked

    _, peak, _, max_loc = cv2.minMaxLoc(res)
    px, py = max_loc
    px_ref = px + subpixel_peak(res, px, py)
    dx = px_ref - tx
    dy = (sy0 + py) - ty
    return dx, dy, peak, px_ref, sy0 + py


def refine_bbox(g1, g2, tx, ty, tw, th, dom_dx, dom_dy):
    """Subdivide the coarse template into cells, match each, and return the
    tight bbox of cells whose motion agrees with the dominant vector.

    Returns (rx, ry, rw, rh) or None if too few cells survive.
    """
    cell = max(REFINE_MIN_CELL, tw // REFINE_CELLS_ACROSS)
    ncx = max(1, tw // cell)
    ncy = max(1, th // cell)
    cw = tw // ncx
    ch = th // ncy

    cells = []  # (cx, cy, cw, ch, dx, dy)
    for j in range(ncy):
        for i in range(ncx):
            cx = tx + i * cw
            cy = ty + j * ch
            r = match_template(g1, g2, cx, cy, cw, ch)
            if r is None:
                continue
            cdx, cdy, cpeak, _, _ = r
            if cpeak < REFINE_CELL_PEAK:
                continue
            cells.append((cx, cy, cw, ch, cdx, cdy))

    if not cells:
        return None

    # Keep cells agreeing with the dominant motion vector.
    tol_dx = abs(dom_dx) * REFINE_DX_TOL + 2.0  # +2px floor for small dx
    keep = [c for c in cells
            if abs(c[4] - dom_dx) <= tol_dx and abs(c[5] - dom_dy) <= REFINE_DY_TOL]
    if len(keep) < REFINE_MIN_CELLS:
        return None

    xs0 = min(c[0] for c in keep)
    ys0 = min(c[1] for c in keep)
    xs1 = max(c[0] + c[2] for c in keep)
    ys1 = max(c[1] + c[3] for c in keep)
    return xs0, ys0, xs1 - xs0, ys1 - ys0


def time_from_name(name):
    """Extract HH:MM:SS.sss from a filename like '..._HHMMSS_sss...'.

    Expects the date_time pattern YYYYMMDD_HHMMSS_sss. Returns '' if not found.
    """
    if not name:
        return ""
    import os, re
    base = os.path.basename(name)
    m = re.search(r"\d{8}_(\d{2})(\d{2})(\d{2})_(\d{3})", base)
    if not m:
        return ""
    hh, mm, ss, ms = m.groups()
    return f"{hh}:{mm}:{ss}.{ms}"


def ransac_translation_dx(dxs, threshold=3.0, iters=300, rng=None):
    """RANSAC for a 1-D pure-translation model (dx = constant).

    Cars move horizontally with no rotation/scale, so a translation-only
    model is the physically correct fit (unlike a full affine/similarity
    model, which can produce wildly wrong translation estimates when fit to
    a small or narrowly-clustered point set -- the rotation/scale terms can
    absorb noise and push the translation term far from the true value).

    Returns (best_dx, inlier_mask) where best_dx is the median dx of the
    largest inlier set found, and inlier_mask is a boolean array over dxs.
    """
    n = len(dxs)
    if rng is None:
        rng = np.random.default_rng(0)
    best_inliers = None
    best_count = -1
    for _ in range(iters):
        idx = rng.integers(0, n)
        cand = dxs[idx]
        mask = np.abs(dxs - cand) <= threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_inliers = mask
    if best_inliers is None:
        return float(np.median(dxs)), np.ones(n, dtype=bool)
    # Refine: recompute median from the best inlier set, then refit inliers
    # around that median (one pass is enough for a 1-D model).
    refined_dx = float(np.median(dxs[best_inliers]))
    refined_mask = np.abs(dxs - refined_dx) <= threshold
    return refined_dx, refined_mask


def decluster_points(pts1, min_dist):
    """Greedily keep points that are >= min_dist apart in frame-1 (x,y).

    Several ORB keypoints frequently land on the same small physical feature
    (e.g. all the corners of one mirror, or speckle on one headlight). Those
    detections share the same source pixels, so their localization errors
    are correlated rather than independent -- averaging over them doesn't
    reduce uncertainty the way averaging over truly independent points
    would. This greedily picks one representative per cluster (highest
    priority kept first; pass points pre-sorted if a particular priority is
    desired).

    Returns a boolean mask over pts1 marking which points were kept.
    """
    n = len(pts1)
    keep = np.zeros(n, dtype=bool)
    kept_xy = []
    for i in range(n):
        p = pts1[i]
        too_close = False
        for q in kept_xy:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_dist ** 2:
                too_close = True
                break
        if not too_close:
            keep[i] = True
            kept_xy.append(p)
    return keep


def refine_subpixel(gray, pts):
    """Refine integer-ish keypoint locations to sub-pixel using cv2.cornerSubPix.

    pts: Nx2 float array of (x, y) locations in `gray`. Returns a new Nx2
    array; points cornerSubPix can't refine (e.g. too close to the image
    border) are left at their original location.
    """
    if len(pts) == 0:
        return pts
    pts32 = pts.astype(np.float32).reshape(-1, 1, 2)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    try:
        refined = cv2.cornerSubPix(
            gray, pts32,
            (KP_SUBPIX_WIN, KP_SUBPIX_WIN),
            (KP_SUBPIX_ZERO_ZONE, KP_SUBPIX_ZERO_ZONE),
            term)
        out = refined.reshape(-1, 2)
        # cornerSubPix can occasionally wander far from the seed if the local
        # neighborhood is flat/ambiguous; clamp to a small radius of the
        # original detection so a bad refinement can't silently inject a
        # large spurious displacement.
        delta = out - pts
        bad = np.hypot(delta[:, 0], delta[:, 1]) > KP_SUBPIX_WIN
        out[bad] = pts[bad]
        return out
    except cv2.error:
        return pts


def keypoint_check(g1, g2, tx, ty, tw, th, tmpl_dx, max_dx):
    """Independent ORB keypoint check inside the template box.

    Detects features in frame-1 box, matches into a frame-2 band around the
    expected shifted position, rejects outliers, and compares the median
    keypoint dx to the template dx.

    Returns dict: status ('agree'/'disagree'/'low_spread'/'n/a'), n_inliers,
    median_dx, diff_px, cv (coefficient of variation of inlier dx, or None),
    x_spread (max-min of inlier frame-1 x, or None).
    """
    h, w = g2.shape[:2]
    mask1 = np.zeros((h, w), np.uint8)
    mask1[ty:ty + th, tx:tx + tw] = 255
    # Frame-2 search band: box shifted by tmpl_dx, padded by max_dx slack.
    pad = int(abs(max_dx)) + 20
    bx0 = max(0, int(tx + tmpl_dx) - pad)
    bx1 = min(w, int(tx + tw + tmpl_dx) + pad)
    by0 = max(0, ty - 20)
    by1 = min(h, ty + th + 20)
    mask2 = np.zeros((h, w), np.uint8)
    mask2[by0:by1, bx0:bx1] = 255

    orb = cv2.ORB_create(KP_FEATURES)
    kp1, des1 = orb.detectAndCompute(g1, mask1)
    kp2, des2 = orb.detectAndCompute(g2, mask2)
    na = {"status": "n/a", "n_inliers": 0, "median_dx": None,
          "diff_px": None, "cv": None, "x_spread": None,
          "pts1": [], "pts2": [], "resp": []}
    if des1 is None or des2 is None or len(kp1) < KP_MIN_INLIERS:
        return na

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    dxs = []
    pts1 = []   # frame-1 coords
    pts2 = []   # frame-2 matched coords
    resp = []   # per-match quality: min(ORB response in frame1, frame2)
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < KP_RATIO * n.distance:
            kp_a = kp1[m.queryIdx]
            kp_b = kp2[m.trainIdx]
            dxs.append(kp_b.pt[0] - kp_a.pt[0])
            pts1.append(kp_a.pt)
            pts2.append(kp_b.pt)
            resp.append(min(kp_a.response, kp_b.response))
    if len(dxs) < KP_MIN_INLIERS:
        return na
    dxs = np.array(dxs, float)
    pts1 = np.array(pts1, float)
    pts2 = np.array(pts2, float)
    resp = np.array(resp, float)

    # Sub-pixel refinement: ORB/FAST keypoints sit on the pixel grid (no
    # built-in sub-pixel step), so refine each matched location in its own
    # frame using local gradient structure. This shrinks per-point
    # localization noise that the median_dx/cv otherwise inherit directly.
    pts1 = refine_subpixel(g1, pts1)
    pts2 = refine_subpixel(g2, pts2)
    dxs = pts2[:, 0] - pts1[:, 0]

    # Step 1: pre-filter to matches whose dx is near the trusted template dx.
    # This removes static background (dx~0) and most wheel points before fitting.
    # Band is 2x the normal agree tolerance so minor parallax isn't cut.
    band = 2.0 * max(KP_TOL_PX, KP_TOL_FRAC * abs(tmpl_dx))
    near = np.abs(dxs - tmpl_dx) <= band
    if near.sum() < KP_MIN_INLIERS:
        return na
    dxs, pts1, pts2, resp = dxs[near], pts1[near], pts2[near], resp[near]

    # Step 2: RANSAC a pure-translation model (no rotation/scale). Vehicles
    # travel horizontally along the road, so translation is the physically
    # correct model; inliers are points whose dx agrees with the consensus.
    kp_dx, inlier_mask = ransac_translation_dx(dxs, threshold=3.0)
    if int(inlier_mask.sum()) >= KP_MIN_INLIERS:
        ip1, ip2 = pts1[inlier_mask], pts2[inlier_mask]
        inl = dxs[inlier_mask]
        iresp = resp[inlier_mask]
    else:
        # Fall back to the pre-filtered set if RANSAC didn't find enough.
        ip1, ip2 = pts1, pts2
        inl = dxs
        iresp = resp
        kp_dx = float(np.median(inl))
    if len(inl) < KP_MIN_INLIERS:
        return na

    # Decluster: keep at most one representative per tight spatial cluster
    # in frame-1, so cv/x_spread reflect independent samples rather than
    # several correlated detections of the same physical feature (e.g.
    # multiple ORB corners on one mirror or headlight). kp_dx itself (the
    # point estimate, from ransac_translation_dx above) is left as computed
    # on the full inlier set -- the median is not meaningfully biased by
    # duplicate detections of the same feature, only the variance estimate is.
    dkeep = decluster_points(ip1, KP_DECLUSTER_DIST)
    if int(dkeep.sum()) >= KP_MIN_INLIERS:
        ip1, ip2, inl, iresp = ip1[dkeep], ip2[dkeep], inl[dkeep], iresp[dkeep]
    # else: declustering would drop below the minimum sample size, so keep
    # the full (clustered) set rather than report nothing.

    diff = kp_dx - tmpl_dx
    tol = max(KP_TOL_PX, KP_TOL_FRAC * abs(tmpl_dx))
    mean = float(np.mean(inl))
    cv = float(np.std(inl) / abs(mean)) if abs(mean) > 1e-6 else None
    x_spread = float(ip1[:, 0].max() - ip1[:, 0].min())

    # Reject results where inlier points are too tightly clustered in x:
    # a translation estimate from a narrow cluster (e.g. all on one small
    # object like a pedestrian's arm) is poorly constrained and can be
    # thrown off by small localization/matching noise, even though the
    # raw per-point dx values look individually plausible.
    if x_spread < KP_MIN_X_SPREAD:
        return {"status": "low_spread", "n_inliers": int(len(inl)),
                "median_dx": kp_dx, "diff_px": float(diff), "cv": cv,
                "x_spread": x_spread,
                "pts1": ip1.tolist(), "pts2": ip2.tolist(),
                "resp": iresp.tolist()}

    # Agree requires displacement within tolerance AND a tight cv.
    agree = abs(diff) <= tol and cv is not None and cv < KP_MAX_CV
    status = "agree" if agree else "disagree"
    return {"status": status, "n_inliers": int(len(inl)),
            "median_dx": kp_dx, "diff_px": float(diff), "cv": cv,
            "x_spread": x_spread,
            "pts1": ip1.tolist(), "pts2": ip2.tolist(),
            "resp": iresp.tolist()}


def plot_keypoint_velocity(kp, mpp, plot_path, tmpl_dx_mph=None, title=None):
    """Scatter plot: per-keypoint speed (mph) vs frame-1 x position (px).

    Dot color encodes ORB corner-response score (min of the two matched
    keypoints' response values) via a continuous colormap -- a proxy for
    per-point detection/localization quality. Useful for checking whether
    dx scatter is driven by weak/low-contrast corners rather than by
    position-dependent effects.

    Uses the RANSAC-inlier points in kp['pts1']/kp['pts2'] (the same points
    drawn as green dots in --viz). Requires a valid mpp (so a distance was
    available); otherwise there is no mph to plot and the caller should skip.
    Plotted regardless of kp['status'] (including 'low_spread'/'disagree')
    since the per-point scatter is exactly what helps diagnose those cases.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts1 = kp.get("pts1", [])
    pts2 = kp.get("pts2", [])
    if not pts1 or not pts2 or mpp is None:
        return False

    xs = np.array([p[0] for p in pts1], float)
    xlabel = "frame-1 x position (px)"

    resp = kp.get("resp", [])
    have_resp = bool(resp) and len(resp) == len(pts1)
    colors = np.array(resp, float) if have_resp else None

    dxs = np.array([p2[0] - p1[0] for p1, p2 in zip(pts1, pts2)], float)
    speeds = np.abs(dxs * mpp) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=PLOT_DPI)
    if have_resp:
        sc = ax.scatter(xs, speeds, s=18, c=colors, cmap="viridis",
                         edgecolors="black", linewidths=0.4, zorder=3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("ORB response (min of matched pair)")
    else:
        ax.scatter(xs, speeds, s=18, c="tab:green", edgecolors="black",
                   linewidths=0.4, zorder=3)

    median_speed = float(np.median(speeds))
    ax.axhline(median_speed, color="gray", linestyle="--", linewidth=1,
               zorder=1, label=f"median = {median_speed:.1f} mph")
    if tmpl_dx_mph is not None:
        ax.axhline(tmpl_dx_mph, color="tab:blue", linestyle="-", linewidth=1,
                   zorder=1, label=f"template = {tmpl_dx_mph:.1f} mph")

    status = kp.get("status")
    x_spread = kp.get("x_spread")
    base_title = title or "Per-keypoint speed vs. quality"
    if status is not None and status != "agree":
        spread_txt = f", x_spread={x_spread:.1f}px" if x_spread is not None else ""
        base_title += f"  [{status}{spread_txt}]"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("keypoint speed (mph)")
    ax.set_title(base_title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return True


def measure_template(img1, img2, viz_path=None, debug_path=None, refine=True,
                     distance=None, name1=None, name2=None, plot_path=None):
    h, w = img1.shape[:2]
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cutoff = int(round(SHADOW_CUTOFF_FRAC * h))

    bbox = object_bbox(g1, g2, cutoff)
    if bbox is None:
        raise RuntimeError("No moving region found; check DIFF_THRESH.")
    x, y, bw, bh = bbox

    # Shrink template inward so we match the object interior, not mask edges.
    tx, ty = x + BBOX_PAD, y + BBOX_PAD
    tw, th = bw - 2 * BBOX_PAD, bh - 2 * BBOX_PAD
    if tw < 20 or th < 20:
        tx, ty, tw, th = x, y, bw, bh  # too small to pad

    # Reject if the moving object's bbox touches a frame edge (entering/leaving,
    # so it's clipped and the displacement would be unreliable).
    if x <= EDGE_MARGIN_PX or (x + bw) >= (w - EDGE_MARGIN_PX):
        raise RuntimeError("object bbox at frame edge; clipped/leaving view.")

    # Maximum plausible horizontal displacement (px): the tighter of the
    # speed cap (needs distance) and the width fraction.
    max_dx = MAX_DX_FRAC * w
    if distance is not None:
        mpp = meters_per_pixel(distance)
        max_dx_speed = (MAX_PLAUSIBLE_MPH / MPS_TO_MPH) * (FRAME_DT_MS / 1000.0) / mpp
        max_dx = min(max_dx, max_dx_speed)

    # ---- Pass 1: coarse match over the full moving bbox --------------------
    r1 = match_template(g1, g2, tx, ty, tw, th, max_dx)
    if r1 is None:
        raise RuntimeError("Template/search size mismatch in pass 1.")
    dx, dy, peak, px_ref, py_abs = r1
    if peak < MIN_PEAK:
        raise RuntimeError(f"Weak correlation peak ({peak:.2f} < {MIN_PEAK}); "
                           "object may be poorly isolated.")
    dx1, peak1 = dx, peak
    refined = False
    # Remember the pass-1 (unrefined, larger) template geometry for fallback.
    p1tx, p1ty, p1tw, p1th = tx, ty, tw, th

    # ---- Pass 2: refine template to cells agreeing with dominant motion ----
    if refine:
        rb = refine_bbox(g1, g2, tx, ty, tw, th, dx, dy)
        if rb is not None:
            rtx, rty, rtw, rth = rb
            r2 = match_template(g1, g2, rtx, rty, rtw, rth, max_dx)
            if r2 is not None and r2[2] >= MIN_PEAK:
                tx, ty, tw, th = rtx, rty, rtw, rth
                dx, dy, peak, px_ref, py_abs = r2
                refined = True

    # ---- Template-size safeguard ------------------------------------------
    # A template much smaller than the object can match spuriously. Use the
    # EXIF BBox (both width and height) as a floor (it is rarely too small);
    # if absent, ensure refinement did not shrink the template below a
    # fraction of pass-1 size.
    exif_bb = exif_bbox_full(name1) if name1 else None
    expanded = False
    if exif_bb is not None:
        bx1, by1, bx2, by2 = exif_bb
        exif_w = bx2 - bx1
        exif_h = by2 - by1
        need_w = tw < BBOX_MIN_FRAC * exif_w
        need_h = th < BBOX_MIN_FRAC * exif_h
        if need_w or need_h:
            # Expand toward the EXIF BBox bounds (clipped to frame) on
            # whichever axis is short, keeping the other axis as-is, then
            # re-match.
            if need_w:
                ex1 = max(EDGE_MARGIN_PX + 1, min(bx1, tx))
                ex2 = min(w - EDGE_MARGIN_PX - 1, max(bx2, tx + tw))
            else:
                ex1, ex2 = tx, tx + tw
            if need_h:
                ey1 = max(0, min(by1, ty))
                ey2 = min(h, max(by2, ty + th))
            else:
                ey1, ey2 = ty, ty + th
            etx, etw = ex1, ex2 - ex1
            ety, eth = ey1, ey2 - ey1
            if etw >= 20 and eth >= 20:
                rexp = match_template(g1, g2, etx, ety, etw, eth, max_dx)
                if rexp is not None and rexp[2] >= MIN_PEAK:
                    # Keep the expanded match (larger template is more
                    # reliable here even if its peak is marginally lower).
                    tx, ty, tw, th = etx, ety, etw, eth
                    dx, dy, peak, px_ref, py_abs = rexp
                    expanded = True
    elif refined and tw < REFINE_MIN_KEEP_FRAC * p1tw:
        # No EXIF BBox: fall back to the larger pass-1 template.
        tx, ty, tw, th = p1tx, p1ty, p1tw, p1th
        dx, dy, peak, px_ref, py_abs = r1
        refined = False

    # Reject if the matched box in frame 2 touches a frame edge.
    nx_chk = int(round(px_ref))
    if nx_chk <= EDGE_MARGIN_PX or (nx_chk + tw) >= (w - EDGE_MARGIN_PX):
        raise RuntimeError("matched box at frame edge; match unreliable.")

    frac = dx / w

    # Independent keypoint cross-check (annotate only; never overrides).
    kp = keypoint_check(g1, g2, tx, ty, tw, th, dx, max_dx)

    if plot_path:
        mpp_plot = meters_per_pixel(distance) if distance is not None else None
        tmpl_dx_mph = (abs(dx * mpp_plot) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH
                       if mpp_plot is not None else None)
        plot_title = os.path.basename(name1) if name1 else None
        plot_keypoint_velocity(kp, mpp_plot, plot_path,
                               tmpl_dx_mph=tmpl_dx_mph, title=plot_title)

    if viz_path:
        # Left-side info column content is computed first (before the crop
        # band) so its required height can set a floor on the vertical
        # margin below -- otherwise a very short/thin template (small th)
        # can produce a crop band shorter than the info box, clipping it.
        info = [f"corr: {peak:.3f}"]
        if distance is not None:
            mpp = meters_per_pixel(distance)
            disp_m = dx * mpp
            info += [f"dist: {distance:.3f} m",
                     f"scale: {mpp * 1000:.3f} mm/px",
                     f"disp: {disp_m:+.3f} m",
                     f"tm speed: {abs(dx * mpp) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH:.1f} mph"]
            if kp.get("status") == "agree" and kp.get("median_dx") is not None:
                kp_speed = abs(kp["median_dx"] * mpp) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH
                info += [f"kp speed: {kp_speed:.1f} mph"]
            if kp.get("n_inliers") and kp.get("cv") is not None:
                info += [f"N_kp: {kp['n_inliers']}  cv: {kp['cv']:.4f}"]
        fs, fth = 0.6, 1
        widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, fth)[0][0]
                  for t in info]
        line_h = 24
        pad = 8
        box_w = max(widths) + 2 * pad
        box_h = line_h * len(info) + 2 * pad - 6

        # Vertical crop band, centered on the template, with VIZ_MARGIN_FRAC of
        # the bbox height added above and below. Same band for both frames so
        # the stack stays vertically registered. The margin is floored so the
        # info box (drawn on the top panel only, i.e. within `margin + th/2`
        # roughly) always fits within one panel's height without clipping --
        # see VIZ_MIN_MARGIN_PX / the box_h-derived floor below.
        nx = int(round(px_ref))
        ny = py_abs
        margin = int(round(VIZ_MARGIN_FRAC * th))
        # The info box is drawn at the top of the *combined* (both-panel)
        # image, so it must fit within one panel's height (by1-by0 = th +
        # 2*margin). Require some clearance below the box too.
        min_margin_for_box = int(np.ceil((box_h + VIZ_INFO_BOX_CLEARANCE_PX - th) / 2.0))
        margin = max(margin, VIZ_MIN_MARGIN_PX, min_margin_for_box)
        band_h = th + 2 * margin
        by0 = ty - margin
        by1 = by0 + band_h
        if by0 < 0:
            by1 -= by0
            by0 = 0
        if by1 > h:
            by0 -= (by1 - h)
            by1 = h
        by0 = max(0, by0)
        # If the frame itself is too short to fit the floored band (rare),
        # by1-by0 will be less than band_h; that's an unavoidable hardware
        # limit, not something more margin math can fix.

        # Compute the speed/px label and its extent up front so timestamps can
        # be shifted clear of it. Label is centered on the seam midpoint.
        # Prefer the keypoint-derived speed for the arrow label (more
        # authoritative than the template match alone); fall back to the
        # template-based speed when no kp_speed is available.
        kp_speed_mph = None
        if (distance is not None and kp.get("status") == "agree"
                and kp.get("median_dx") is not None):
            mpp = meters_per_pixel(distance)
            kp_speed_mph = abs(kp["median_dx"] * mpp) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH
        if kp_speed_mph is not None:
            label = f"{kp_speed_mph:.1f} mph"
        elif distance is not None:
            mpp = meters_per_pixel(distance)
            v_mph = abs((dx * mpp) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH)
            label = f"{v_mph:.1f} mph"
        else:
            label = f"{abs(dx):.0f} px"
        seam_y = by1 - by0                  # crop height = top panel height
        if dx >= 0:                         # moving right -> leading right edges
            x_from, x_to = tx + tw, nx + tw
        else:                               # moving left -> leading left edges
            x_from, x_to = tx, nx
        lx = min(x_from, x_to) + abs(x_to - x_from) // 2 - 40
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        (char_w, _), _ = cv2.getTextSize("m", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        # Shift timestamps so their whole length clears the label by >=1 char.
        # For leftward travel shift right (to label's right end), else left.
        ts_top = time_from_name(name1)
        ts_bot = time_from_name(name2)
        (tsw, _), _ = cv2.getTextSize(ts_top or ts_bot or "00:00:00.000",
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if dx < 0:                          # going left: place to label's right
            ts_x_top = lx + lw + char_w
        else:                               # going right: place to label's left
            ts_x_top = lx - char_w - tsw
        # Same shift relative to each box's leading edge for consistency.
        shift = ts_x_top - tx
        ts_x_bot = nx + shift

        top = img1[by0:by1].copy()      # frame 1 with template (green)
        bot = img2[by0:by1].copy()      # frame 2 with match (red)
        cv2.rectangle(top, (tx, ty - by0), (tx + tw, ty - by0 + th),
                      (0, 255, 0), 2)
        cv2.rectangle(bot, (nx, ny - by0), (nx + tw, ny - by0 + th),
                      (0, 180, 180), 2)

        def stamp(panel, bx, box_bottom_y, text, color):
            if not text:
                return
            bx = max(2, min(bx, panel.shape[1] - tsw - 2))
            ty_txt = box_bottom_y + 18
            ty_txt = min(ty_txt, panel.shape[0] - 6)
            cv2.putText(panel, text, (bx, ty_txt), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, text, (bx, ty_txt), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        stamp(top, ts_x_top, ty - by0 + th, ts_top, (0, 255, 0))
        stamp(bot, ts_x_bot, ny - by0 + th, ts_bot, (0, 180, 180))

        # Green dots at inlier keypoints: frame-1 locations on the top panel,
        # frame-2 matched locations on the bottom panel (cropped coords).
        for px_, py_ in kp.get("pts1", []):
            yy = int(round(py_)) - by0
            if 0 <= yy < top.shape[0]:
                cv2.circle(top, (int(round(px_)), yy), 3, (0, 255, 0), -1,
                           cv2.LINE_AA)
        for px_, py_ in kp.get("pts2", []):
            yy = int(round(py_)) - by0
            if 0 <= yy < bot.shape[0]:
                cv2.circle(bot, (int(round(px_)), yy), 3, (0, 255, 0), -1,
                           cv2.LINE_AA)

        vis = np.vstack([top, bot])

        # Arrow along the seam midline, leading edge frame1 -> frame2.
        # Black underlay first, then green on top, for an outline effect.
        cv2.arrowedLine(vis, (x_from, seam_y), (x_to, seam_y),
                        (0, 0, 0), 6, tipLength=0.08)
        cv2.arrowedLine(vis, (x_from, seam_y), (x_to, seam_y),
                        (0, 255, 0), 3, tipLength=0.08)

        ly = seam_y - 12
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Info box, drawn over a 50% black rectangle (content was computed
        # above, before the crop band, so the margin floor could account
        # for it).
        x0, y0 = 6, 6
        ov = vis.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        vis = cv2.addWeighted(ov, 0.7, vis, 0.3, 0) # darken the box area so white text shows up reasonably against any
        for i, t in enumerate(info):
            yt = y0 + pad + line_h * (i + 1) - 8
            cv2.putText(vis, t, (x0 + pad, yt), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (0, 255, 0), fth, cv2.LINE_AA)

        if viz_path.lower().endswith((".jpg", ".jpeg")):
            cv2.imwrite(viz_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            cv2.imwrite(viz_path, vis)

    if debug_path:
        # Single-frame overlay on frame 2: the template (shaded, from frame 1)
        # at its ORIGINAL location, and its MATCHED location, with a vector.
        nx = int(round(px_ref))
        ny = py_abs
        dbg = img2.copy()

        # Shade the matched region semi-transparently (where it ended up).
        shade = dbg.copy()
        cv2.rectangle(shade, (nx, ny), (nx + tw, ny + th), (0, 0, 255), -1)
        dbg = cv2.addWeighted(shade, 0.30, dbg, 0.70, 0)

        # Original template position (green outline) and matched (red outline).
        cv2.rectangle(dbg, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
        cv2.rectangle(dbg, (nx, ny), (nx + tw, ny + th), (0, 0, 255), 2)

        # Displacement vector between box centers.
        c0 = (tx + tw // 2, ty + th // 2)
        c1 = (nx + tw // 2, ny + th // 2)
        cv2.arrowedLine(dbg, c0, c1, (0, 255, 255), 2, tipLength=0.05)

        label = f"dx={dx:+.1f}px ({frac:+.3f} W)  peak={peak:.3f}"
        cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(debug_path, dbg)

    return {
        "dx_px": float(dx), "dy_px": float(dy),
        "frame_width_px": w, "frame_height_px": h, "fraction_width": float(frac),
        "peak": float(peak), "template_wh": (int(tw), int(th)),
        "shadow_cutoff_px": cutoff,
        "refined": refined, "dx1_px": float(dx1), "peak1": float(peak1),
        "kp": kp,
    }


def exif_bbox_full(path):
    """Parse 'BBox=x1,y1,x2,y2' from EXIF ImageDescription; return
    (x1, y1, x2, y2) or None.

    The box is a rough estimate of the object extent in pixels; it is
    occasionally too large but rarely too small, so it is used as a floor
    for both template width and height.
    """
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(path)
        exif = img._getexif()
    except Exception:
        return None
    if not exif:
        return None
    desc = exif.get(0x010E)  # ImageDescription
    if not desc:
        return None
    import re
    m = re.search(r"BBox=(\d+),(\d+),(\d+),(\d+)", str(desc))
    if not m:
        return None
    x1, y1, x2, y2 = (int(v) for v in m.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def exif_subject_distance(path):
    """Read EXIF SubjectDistance (tag 0x9206) in meters, or None if absent.

    Uses Pillow; the tag is a RATIONAL (numerator/denominator) per the EXIF
    spec, e.g. 19557/1000 = 19.557 m.
    """
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(path)
        exif = img._getexif()
    except Exception:
        return None
    if not exif:
        return None
    val = exif.get(0x9206)  # SubjectDistance
    if val is None:
        return None
    try:
        # Pillow returns a float, an IFDRational, or a (num, den) tuple.
        if isinstance(val, tuple) and len(val) == 2:
            num, den = val
            d = num / den if den else None
        else:
            d = float(val)
    except Exception:
        return None
    if d is None or d <= 0:
        return None
    return d


def meters_per_pixel(object_distance_m):
    """Object-plane meters per pixel via thin-lens formula.

    Magnification m = f / (d - f), so the object width imaged across the full
    sensor width is sensor_width / m = sensor_width * (d - f) / f.
    """
    d_mm = object_distance_m * 1000.0
    object_width_mm = SENSOR_WIDTH_MM * (d_mm - FOCAL_LENGTH_MM) / FOCAL_LENGTH_MM
    object_width_m = object_width_mm / 1000.0
    return object_width_m / SENSOR_WIDTH_PX


def resolve_distance(path, cli_distance):
    """Return (distance, source) using CLI override then EXIF."""
    if cli_distance is not None:
        return cli_distance, "--distance"
    d = exif_subject_distance(path)
    return (d, "EXIF") if d is not None else (None, None)


def velocity_fields(dx_px, distance):
    """Return dict of mpp, disp_m, v_mps, v_mph (or Nones if no distance)."""
    if distance is None:
        return {"mpp": None, "disp_m": None, "v_mps": None, "v_mph": None}
    mpp = meters_per_pixel(distance)
    disp_m = dx_px * mpp
    v_mps = disp_m / (FRAME_DT_MS / 1000.0)
    return {"mpp": mpp, "disp_m": disp_m, "v_mps": v_mps,
            "v_mph": v_mps * MPS_TO_MPH}


def parse_event_files(folder):
    """Group JPEGs in folder by trailing event number N.

    Filenames: YYYYMMDD_HHMMSS_sss_N.jpg. Returns an ordered dict
    {N: [sorted file paths by time]} for events with >= 2 files.
    """
    import os, re
    pat = re.compile(r"^(\d{8})_(\d{6})_(\d{3})_(\d+)\.jpe?g$", re.IGNORECASE)
    groups = {}
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if not m:
            continue
        date, hms, ms, n = m.groups()
        key = int(n)
        groups.setdefault(key, []).append((date + hms + ms, os.path.join(folder, fn)))
    events = {}
    for n in sorted(groups):
        files = [p for _, p in sorted(groups[n])]
        events[n] = files
    return events


def run_folder(folder, save_viz, refine, cli_distance, save_plot):
    """Process every event in a folder; write CSV (and optional viz JPEGs)."""
    import csv
    events = parse_event_files(folder)
    if not events:
        sys.exit(f"No files matching YYYYMMDD_HHMMSS_sss_N.jpg in {folder}")

    motion_dir = os.path.join(folder, "motion")
    os.makedirs(motion_dir, exist_ok=True)
    csv_path = os.path.join(motion_dir, "events.csv")

    rows = []
    fastest = []
    for n, files in events.items():
        if len(files) < 2:
            print(f"event {n}: only {len(files)} file; skipped")
            continue
        f1, f2 = files[0], files[1]
        b1 = os.path.basename(f1)
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        if img1 is None or img2 is None:
            print(f"event {n}: could not read images; skipped")
            rows.append([b1, n, "", "", "", "", "", "", "", "", "", "", "ERROR: unreadable"])
            continue

        distance, _ = resolve_distance(f1, cli_distance)
        viz_path = None
        if save_viz:
            viz_path = os.path.join(motion_dir, os.path.splitext(b1)[0] + ".jpg")
        plot_path = None
        if save_plot:
            plot_path = os.path.join(motion_dir, os.path.splitext(b1)[0] + "_plot.png")
        try:
            r = measure_template(img1, img2, viz_path, None, refine,
                                 distance, f1, f2, plot_path)
        except RuntimeError as e:
            print(f"event {n}: {e}; skipped")
            rows.append([b1, n, "", "", "", "", "", "", "", "", "", "", f"ERROR: {e}"])
            continue

        vf = velocity_fields(r["dx_px"], distance)
        v_mph = f"{vf['v_mph']:+.3f}" if vf["v_mph"] is not None else ""
        spd = abs(vf["v_mph"]) if vf["v_mph"] is not None else None
        spd_s = f"{spd:.3f}" if spd is not None else ""
        dist_s = f"{distance:.3f}" if distance is not None else ""
        scale_s = f"{vf['mpp'] * 1000:.4f}" if vf["mpp"] is not None else ""
        disp_s = f"{vf['disp_m']:+.4f}" if vf["disp_m"] is not None else ""
        kp = r["kp"]
        kp_cv_s = f"{kp['cv']:.4f}" if kp["cv"] is not None else ""
        if kp["status"] == "agree" and kp["median_dx"] is not None and vf["mpp"] is not None:
            kp_speed = abs(kp["median_dx"] * vf["mpp"]) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH
            kp_spd_s = f"{kp_speed:.3f}"
        else:
            kp_spd_s = ""
        rows.append([b1, n, v_mph, spd_s, f"{r['peak']:.3f}", dist_s, scale_s,
                     disp_s, kp["status"], kp["n_inliers"], kp_cv_s, kp_spd_s, ""])
        if spd is not None and kp_spd_s:
            fastest.append({"n": n, "file": b1, "peak": r["peak"],
                            "distance": distance, "disp_m": vf["disp_m"],
                            "speed": spd, "kp_spd": kp_speed})
        vtxt = f"{vf['v_mph']:+.2f} mph" if vf["v_mph"] is not None else "n/a"
        kptxt = kp["status"]
        if kp["cv"] is not None:
            kptxt += f" (cv={kp['cv']:.3f})"
        print(f"event {n}: {b1}  v={vtxt}  peak={r['peak']:.3f}  kp={kptxt}")

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["first_jpeg", "event_number", "velocity_mph", "speed_mph",
                    "corr_peak", "distance_m", "scale_mm_per_px",
                    "displacement_m", "kp_check", "kp_n", "kp_cv",
                    "kp_speed_mph", "note"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")
    if save_viz:
        print(f"Saved viz JPEGs to {motion_dir}")
    if save_plot:
        print(f"Saved keypoint-velocity plot PNGs to {motion_dir}")

    if fastest:
        top = max(fastest, key=lambda e: e["kp_spd"])
        print("\nFastest event (by keypoint speed):")
        print(f"  event    : {top['n']}")
        print(f"  file     : {top['file']}")
        print(f"  corr     : {top['peak']:.3f}")
        print(f"  distance : {top['distance']:.3f} m")
        print(f"  disp     : {top['disp_m']:+.3f} m")
        print(f"  speed    : {top['speed']:.2f} mph")
        if top['kp_spd'] is not None:
            print(f"  kp_speed : {top['kp_spd']:.2f} mph")
        else:
            print(f"  kp_speed : n/a")
    else:
        print("\nNo events had a usable keypoint-based speed; "
              "no fastest event to report.")


def main():
    print(f"measure_displacement.py version {VERSION}")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame1", nargs="?", help="first JPEG (single-pair mode)")
    ap.add_argument("frame2", nargs="?", help="second JPEG (single-pair mode)")
    ap.add_argument("--folder", help="process every event in this folder; "
                                     "writes CSV (and viz JPEGs if --viz) to "
                                     "<folder>/motion/")
    ap.add_argument("--viz", nargs="?", const=True, default=None,
                    help="single-pair mode: output PNG path. "
                         "folder mode: bare flag enables per-event viz JPEGs.")
    ap.add_argument("--plot", nargs="?", const=True, default=None,
                    help="single-pair mode: output PNG path for per-keypoint "
                         "speed plot. folder mode: bare flag enables "
                         "per-event plot PNGs in <folder>/motion/.")
    ap.add_argument("--debug", help="optional single-frame overlay PNG showing "
                                     "template origin and matched position")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    help="disable the 2-pass template refinement (on by default)")
    ap.add_argument("--distance", type=float, default=None,
                    help="camera-to-object distance in meters; overrides EXIF "
                         "SubjectDistance. If omitted, EXIF is used when present.")
    args = ap.parse_args()

    if args.folder:
        run_folder(args.folder, bool(args.viz), args.refine, args.distance,
                  bool(args.plot))
        return

    if not args.frame1 or not args.frame2:
        sys.exit("Provide two image paths, or use --folder PATH.")

    viz_path = args.viz if isinstance(args.viz, str) else None
    img1 = cv2.imread(args.frame1)
    img2 = cv2.imread(args.frame2)
    if img1 is None or img2 is None:
        sys.exit("Could not read one or both images.")

    distance, dist_src = resolve_distance(args.frame1, args.distance)

    if isinstance(args.plot, str):
        plot_path = args.plot
    elif args.plot:
        base = viz_path if viz_path else args.frame1
        plot_path = os.path.splitext(base)[0] + "_plot.png"
    else:
        plot_path = None

    r = measure_template(img1, img2, viz_path, args.debug, args.refine,
                         distance, args.frame1, args.frame2, plot_path)
    sign = "right (+)" if r["dx_px"] >= 0 else "left (-)"
    print(f"frame size        : {r['frame_width_px']} x {r['frame_height_px']} px")
    print(f"correlation peak  : {r['peak']:.3f}")
    print(f"template size     : {r['template_wh'][0]} x {r['template_wh'][1]} px"
          f"{' (refined)' if r['refined'] else ' (pass-1; refine skipped)'}")
    print(f"pass-1 vs pass-2  : dx {r['dx1_px']:+.2f} (peak {r['peak1']:.3f})  ->  "
          f"dx {r['dx_px']:+.2f} (peak {r['peak']:.3f})")
    print(f"shadow cutoff     : y >= {r['shadow_cutoff_px']} px excluded")
    print(f"displacement      : dx={r['dx_px']:+.2f} px  dy={r['dy_px']:+.2f} px  [{sign}]")
    print(f"fraction of width : {r['fraction_width']:+.4f}  over {FRAME_DT_MS:.0f} ms")

    kp = r["kp"]
    if kp["status"] == "n/a":
        print(f"keypoint check    : n/a (insufficient keypoints)")
    else:
        cvtxt = f"{kp['cv']:.4f}" if kp["cv"] is not None else "n/a"
        spreadtxt = f"{kp['x_spread']:.1f} px" if kp.get("x_spread") is not None else "n/a"
        print(f"keypoint check    : {kp['status']}  "
              f"(n={kp['n_inliers']}, kp_dx={kp['median_dx']:+.1f} px, "
              f"diff={kp['diff_px']:+.1f} px, cv={cvtxt}, x_spread={spreadtxt})")
        if kp["status"] == "agree" and distance is not None and kp["median_dx"] is not None:
            mpp = meters_per_pixel(distance)
            kp_speed = abs(kp["median_dx"] * mpp) / (FRAME_DT_MS / 1000.0) * MPS_TO_MPH
            print(f"keypoint speed    : {kp_speed:.2f} mph (body keypoints)")

    vf = velocity_fields(r["dx_px"], distance)
    if distance is not None:
        print(f"object distance   : {distance:.3f} m (from {dist_src})")
        print(f"scale             : {vf['mpp'] * 1000:.3f} mm/px  "
              f"({vf['mpp']:.6f} m/px)")
        print(f"displacement      : {vf['disp_m']:+.3f} m")
        print(f"velocity          : {vf['v_mps']:+.3f} m/s  ({vf['v_mph']:+.2f} mph)")
    else:
        print("object distance   : not provided (--distance) or found in EXIF; "
              "velocity not computed")

    if plot_path:
        if distance is not None:
            print(f"keypoint plot     : {plot_path}")
        else:
            print("keypoint plot     : skipped (no distance available)")


if __name__ == "__main__":
    main()

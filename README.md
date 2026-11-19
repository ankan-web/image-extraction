# Image Extraction / Self-Embedded Watermarking Project

This project implements a **self-embedded dual watermarking** system for image tamper detection and recovery, based on:

> Huseyin Ozkaya, Veysel Aslantas, "A Novel Self-Embedded Dual Watermarking Method for Image Tamper Detection and Recovery", Arabian Journal for Science and Engineering (2025) 50:19367–19389.

The code can embed a hidden watermark into an image, crop/tamper that watermarked image, detect the tampered blocks, and reconstruct the damaged region from information hidden elsewhere in the same image.

On top of the paper's baseline method, this project adds a **third-tier backup** designed specifically for the paper's admitted worst case: when both main recovery copies of a block are destroyed.

---

## What is in this folder

| File | Purpose |
|---|---|
| `self_embedded_recovery.py` | **Improved** method (baseline + new Tier 3 backup). Full embed → attack → detect → recover pipeline. Also supports `--compare`. |
| `baseline_traditional.py` | The original/traditional method from the paper, without the improvement. Kept as a separate file so you can run it independently. |
| `run_comparison.py` | Single entry point that runs **both** methods on the same image and prints a comparison table. |
| `run_demo.sh` | Interactive bash launcher. Asks questions, then runs the right script. |
| `README.md` | This file. |

---

## Requirements

Python 3.9+ and two external packages:

```bash
pip install numpy opencv-python
```

No SciPy, PyWavelets, Pillow, or other imaging libraries are needed. `argparse` and `pathlib` are built into Python.

Verify:

```bash
python3 -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

---

## Easiest way to run: interactive bash script

From anywhere:

```bash
/home/ankan/image-extraction/run_demo.sh
```

It asks:

1. Original image path
2. Crop ratio (e.g. `0.4` = 40%, default `0.4`)
3. Grayscale? `y/N`
4. Compare traditional vs improved? `Y/n`
5. If comparing is off: which method (`improved` or `traditional`)
6. Use your own cropped 16-bit watermarked image? `y/N`
7. Random seed (default `2025`)
8. Output folder (default `comparison_output` or `recovery_output`)

Then it runs the appropriate Python file automatically.

---

## Manual commands

### 1. Improved method only

```bash
python3 /home/ankan/image-extraction/self_embedded_recovery.py \
  --image "/path/to/your/image.jpg" \
  --crop 0.4 \
  --output /home/ankan/image-extraction/improved_output
```

Add `--grayscale` for grayscale images. Omit `--image` to use a built-in synthetic sample image.

### 2. Traditional/baseline method only

```bash
python3 /home/ankan/image-extraction/baseline_traditional.py \
  --image "/path/to/your/image.jpg" \
  --crop 0.4 \
  --output /home/ankan/image-extraction/traditional_output
```

### 3. Run both and get a comparison table

```bash
python3 /home/ankan/image-extraction/run_comparison.py \
  --image "/path/to/your/image.jpg" \
  --crop 0.4 \
  --output /home/ankan/image-extraction/final_compare
```

Example table:

```text
+---------------------------+--------------+--------------+
| Metric                    |     Baseline |     Improved |
+---------------------------+--------------+--------------+
| Watermarked PSNR (dB)     |        39.65 |        38.51 |
| Detection accuracy (%)    |        99.49 |        99.56 |
| Detection sensitivity (%) |        98.77 |        98.93 |
| Recovered PSNR whole (dB) |        27.05 |        27.26 |
| Recovered SSIM whole      |       0.8333 |       0.8259 |
| Tier-3 assisted blocks    |            - |   405 / 6652 |
| Tier-3 region PSNR (dB)   |        22.70 |        26.56 |
+---------------------------+--------------+--------------+
```

The most important row for the improvement is **Tier-3 region PSNR**, because it isolates only the blocks where both primary recovery copies were destroyed. Whole-image averages dilute this effect because most blocks are already recovered by Tier 1 or Tier 2.

A CSV of the same table is written as `comparison.csv`.

---

## Using your own cropped image

By default the script crops the watermarked image itself. To crop it yourself:

**Step 1 — generate the exact watermarked image:**

```bash
python3 /home/ankan/image-extraction/self_embedded_recovery.py \
  --image "/path/to/your/image.jpg" \
  --output /home/ankan/image-extraction/first
```

This writes `first/watermarked_16bit.png`.

**Step 2 — crop that 16-bit PNG in any image editor** and save it as a 16-bit PNG with the same width/height. Do **not** crop `watermarked_preview.png`; that file is only an 8-bit preview and will destroy the hidden bits.

**Step 3 — recover from your cropped image:**

```bash
python3 /home/ankan/image-extraction/self_embedded_recovery.py \
  --image "/path/to/your/image.jpg" \
  --attacked "/path/to/your_cropped_16bit.png" \
  --output /home/ankan/image-extraction/custom_test
```

Or run both methods:

```bash
python3 /home/ankan/image-extraction/run_comparison.py \
  --image "/path/to/your/image.jpg" \
  --attacked "/path/to/your_cropped_16bit.png" \
  --output /home/ankan/image-extraction/custom_compare
```

The baseline method ignores the extra backup bits, so the same custom attacked image works for both methods.

---

## Output files

For a single improved run:

| File | Meaning |
|---|---|
| `watermarked_16bit.png` | Exact watermarked image with hidden bits preserved. Use this for custom attacks. |
| `watermarked_preview.png` | Normal 8-bit preview for viewing only. |
| `attacked.png` | Image with the cropping attack applied. |
| `detected_map.png` | Black-and-white tamper map; white = detected tampered block. |
| `recovered.png` | Final restored image. |
| `original.png` | Saved only for color runs, for side-by-side display. |

For `run_comparison.py`, outputs are separated:

```text
comparison_output/
├── comparison.csv
├── baseline/
│   ├── watermarked_16bit.png
│   ├── watermarked_preview.png
│   ├── attacked.png
│   ├── detected_map.png
│   └── recovered.png
└── improved/
    ├── watermarked_16bit.png
    ├── watermarked_preview.png
    ├── attacked.png
    ├── detected_map.png
    ├── tier3_assisted_map.png
    └── recovered.png
```

`tier3_assisted_map.png` shows exactly which blocks used the new backup tier.

---

## How the method works

### Image preparation

- The image is padded to dimensions divisible by 4.
- It is split into four quadrants: A (top-left), B (top-right), C (bottom-left), D (bottom-right).
- Each quadrant is split into non-overlapping 2×2 pixel blocks.
- A lossless integer lifting-based 2D DWT is applied to every 2×2 block, giving `LL1`, `LH1`, `HL1`, `HH1`.

### Baseline watermark (8 bits per block)

For every block:

- **5 recovery bits** are derived from the seven MSBs of `LL1`.
- **3 authentication bits** are derived from the first five of those MSBs plus the block number.

Embedding:

- Authentication bits go into `LL1` bits 2–4 (LSB2, LSB3, LSB4).
- Recovery bits are written twice, into two partner blocks chosen by a pseudo-random lookup table:
  - Copy 1: `LH1` bits 0–2 and `HH1` bits 2–3 of partner Y1.
  - Copy 2: `HL1` bits 0–2 and `HH1` bits 0–1 of partner Y2.

Detection compares the extracted authentication bits with freshly regenerated ones. The result is cleaned with 3×3 morphological opening/closing and small-component removal.

### Recovery tiers (improved version)

For each tampered block:

1. **Tier 1:** if Y1 is intact, reconstruct from its full 5-bit copy.
2. **Tier 2:** else if Y2 is intact, reconstruct from its full 5-bit copy.
3. **Tier 3 (new):** else if the cross-paired coarse backup is intact:
   - Take the ordinary 8-neighbour average.
   - Compare its brightness bucket with the stored 2-bit bucket.
   - If different, shift the block mean to the stored bucket midpoint (32, 96, 160, or 224) while preserving local texture from the neighbour average.
4. **Tier 4:** if everything is lost, use plain 8-neighbour averaging (the original paper's worst case).

Tiers 1 and 2 are unchanged from the paper.

---

## The new improvement in detail

The paper's own limitation: if both Y1 and Y2 are destroyed, the block is recovered only by averaging its 8 neighbours, which ignores the lost block's actual brightness and can be badly wrong near edges or highlights.

The improvement adds a **2-bit coarse brightness hint** per block:

| 2-bit code | Brightness range | Midpoint |
|---|---|---|
| `00` | 0–63 | 32 |
| `01` | 64–127 | 96 |
| `10` | 128–191 | 160 |
| `11` | 192–255 | 224 |

These are not pixel data; they are only a coarse bucket. The bucket is computed from the block's original average brightness.

The bits are stored in space the baseline never used:

- `LH1` bit 3 (LSB4) — one bit
- `HL1` bit 3 (LSB4) — one bit

No new coefficient is touched, so the extra visible distortion is small. In tests, watermarked PSNR dropped from about 39.65 dB to 38.51 dB on a high-contrast grayscale sample.

### Cross-pairing

Storing the backup in the same blocks as Y1/Y2 would be useless: an attack destroying both primary copies would probably destroy the backup too. The backup is therefore stored in blocks that hold a **different** block's primary recovery data, generated with separate pseudo-random permutations and a fixed opposite-part mapping. This makes the backup physically independent of the block's own two primary copies.

---

## Metrics reported

- **Watermarked PSNR** — imperceptibility of the hidden data. Higher is better.
- **Detection accuracy** — `(TP + TN) / (TP + TN + FP + FN)` at block level.
- **Detection sensitivity** — `TP / (TP + FN)` at block level.
- **Recovered PSNR** — whole-image restoration quality vs the original.
- **Recovered SSIM** — structural similarity, computed with an 11×11 Gaussian window.
- **Tier-3 assisted blocks** — how many detected blocks actually used the new backup.
- **Tier-3 region PSNR** — restoration quality restricted to those blocks; the cleanest way to show the improvement.

---

## Proving it is a real reconstruction

Each run saves `attacked.png`, `detected_map.png`, and `recovered.png`. To verify the recovered image is not simply the original renamed:

```bash
python3 - <<'PY'
import cv2, numpy as np
p = "/home/ankan/image-extraction/final_compare/improved"
o = cv2.imread("/path/to/your/image.jpg")
a = cv2.imread(f"{p}/attacked.png")
r = cv2.imread(f"{p}/recovered.png")
d = cv2.imread(f"{p}/detected_map.png", 0) > 0
m = np.repeat(np.repeat(d, 2, 0), 2, 1)[:o.shape[0], :o.shape[1]]
ps = lambda x, y: 10 * np.log10(255**2 / np.mean((x.astype(float) - y.astype(float))**2))
print("outside crop identical:", np.array_equal(o[~m], r[~m]))
print("inside crop changed:", not np.array_equal(o[m], r[m]))
print("PSNR attacked -> recovered:", round(ps(o, a), 2), "->", round(ps(o, r), 2))
PY
```

Also compare file hashes:

```bash
sha256sum "/path/to/your/image.jpg" /home/ankan/image-extraction/final_compare/improved/recovered.png
```

They must be different.

---

## Important notes and troubleshooting

- Always use the **same `--seed`** for embedding and recovery. Default is `2025`.
- Use `watermarked_16bit.png` for custom attacks, not the 8-bit preview.
- A custom attacked image must have exactly the same dimensions as the original.
- Color images are processed channel by channel; the tamper map is the OR of the channel maps.
- Images whose dimensions are not multiples of 4 are padded internally and cropped back afterwards.
- If you see `use the saved watermarked_16bit.png, not an 8-bit preview`, you passed a normal 8-bit image to `--attacked`.
- If you see a shape mismatch, your cropped image changed the image size.
- Very smooth images may show only a small Tier-3 gain because neighbour averaging is already good there. High-contrast edges and highlights show the benefit most clearly.

---

## Suggested presentation flow

1. Run `run_demo.sh` with compare enabled on a test image.
2. Show the four images side by side: original → watermarked → attacked → recovered.
3. Show `detected_map.png` to prove localization.
4. Show `tier3_assisted_map.png` to prove the new tier actually activated.
5. Show the printed/CSV table, focusing on **Tier-3 region PSNR**.
6. Run the verification snippet and `sha256sum` to prove the recovered file is not the original renamed.

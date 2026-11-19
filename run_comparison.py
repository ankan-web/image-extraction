import argparse
from pathlib import Path

import cv2
import numpy as np

import baseline_traditional as baseline
import self_embedded_recovery as improved


def main():
    parser = argparse.ArgumentParser(description="Run traditional and improved methods together")
    parser.add_argument("--image", type=Path, required=True, help="source original image")
    parser.add_argument("--output", type=Path, default=Path("comparison_output"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--crop", type=float, default=0.4)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--attacked", type=Path, help="optional custom cropped watermarked 16-bit PNG")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.output / "baseline"
    improved_dir = args.output / "improved"
    baseline_dir.mkdir(exist_ok=True)
    improved_dir.mkdir(exist_ok=True)

    flags = cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_COLOR
    source = cv2.imread(str(args.image), flags)
    if source is None:
        raise FileNotFoundError(args.image)
    original = source.astype(np.int32)
    channels = 1 if source.ndim == 2 else 3

    baseline_watermarked = baseline.embed_image(original, args.seed)
    improved_watermarked = improved.embed_image(original, args.seed)

    if args.attacked is None:
        baseline_attacked, attacked_pixels = baseline.apply_crop(baseline_watermarked, args.crop)
        improved_attacked, _ = improved.apply_crop(improved_watermarked, args.crop)
    else:
        baseline_attacked = improved.load_exact(args.attacked)
        improved_attacked = baseline_attacked.copy()
        if baseline_attacked.shape != improved_watermarked.shape:
            raise ValueError(f"attacked shape {baseline_attacked.shape} != {improved_watermarked.shape}")
        if baseline_attacked.min() < -1000 or baseline_attacked.max() > 1000:
            raise ValueError("use a 16-bit PNG cropped from watermarked_16bit.png")
        if channels == 3:
            attacked_pixels = np.any(improved_attacked != improved_watermarked, axis=2)
        else:
            attacked_pixels = improved_attacked != improved_watermarked

    baseline_restored, baseline_detected = baseline.recover_image(baseline_attacked, args.seed)
    improved_restored, improved_detected, assisted = improved.recover_image(
        improved_attacked, args.seed, return_assisted=True
    )

    padded_truth, truth_pad_h, truth_pad_w = improved.pad_to_multiple(attacked_pixels, 4)
    truth = improved.unpad(improved.block_ground_truth(padded_truth), truth_pad_h // 2, truth_pad_w // 2)
    baseline_accuracy, baseline_sensitivity = baseline.detection_scores(baseline_detected, truth)
    improved_accuracy, improved_sensitivity = improved.detection_scores(improved_detected, truth)

    baseline.save_exact(baseline_dir / "watermarked_16bit.png", baseline_watermarked)
    baseline.save_preview(baseline_dir / "watermarked_preview.png", baseline_watermarked)
    baseline.save_preview(baseline_dir / "attacked.png", baseline_attacked)
    baseline.save_preview(baseline_dir / "detected_map.png", baseline_detected.astype(np.uint8) * 255)
    baseline.save_preview(baseline_dir / "recovered.png", baseline_restored)

    improved.save_exact(improved_dir / "watermarked_16bit.png", improved_watermarked)
    improved.save_preview(improved_dir / "watermarked_preview.png", improved_watermarked)
    improved.save_preview(improved_dir / "attacked.png", improved_attacked)
    improved.save_preview(improved_dir / "detected_map.png", improved_detected.astype(np.uint8) * 255)
    improved.save_preview(improved_dir / "recovered.png", improved_restored)
    improved.save_preview(improved_dir / "tier3_assisted_map.png", assisted.astype(np.uint8) * 255)

    assisted_pixels = np.repeat(np.repeat(assisted, 2, axis=0), 2, axis=1)[:original.shape[0], :original.shape[1]]
    rows = [
        ("Watermarked PSNR (dB)", f"{baseline.psnr(original, baseline_watermarked):.2f}", f"{improved.psnr(original, improved_watermarked):.2f}"),
        ("Detection accuracy (%)", f"{baseline_accuracy * 100:.2f}", f"{improved_accuracy * 100:.2f}"),
        ("Detection sensitivity (%)", f"{baseline_sensitivity * 100:.2f}", f"{improved_sensitivity * 100:.2f}"),
        ("Recovered PSNR whole (dB)", f"{baseline.psnr(original, baseline_restored):.2f}", f"{improved.psnr(original, improved_restored):.2f}"),
        ("Recovered SSIM whole", f"{improved.ssim(original, baseline_restored):.4f}", f"{improved.ssim(original, improved_restored):.4f}"),
        ("Tier-3 assisted blocks", "-", f"{int(assisted.sum())} / {int(improved_detected.sum())}"),
    ]
    if np.any(assisted_pixels):
        rows.append((
            "Tier-3 region PSNR (dB)",
            f"{baseline.psnr(original[assisted_pixels], baseline_restored[assisted_pixels]):.2f}",
            f"{improved.psnr(original[assisted_pixels], improved_restored[assisted_pixels]):.2f}",
        ))

    print(f"image: {source.shape[1]}x{source.shape[0]}, channels={channels}")
    print()
    improved.print_comparison_table(rows)
    with open(args.output / "comparison.csv", "w") as handle:
        handle.write("Metric,Baseline,Improved\n")
        for name, baseline_value, improved_value in rows:
            handle.write(f"{name},{baseline_value},{improved_value}\n")
    print(f"table saved to {(args.output / 'comparison.csv').resolve()}")
    print(f"outputs written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

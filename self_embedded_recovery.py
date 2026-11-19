import argparse
from pathlib import Path

import cv2
import numpy as np


PARTNERS = ((1, 2), (0, 3), (3, 0), (2, 1))
BACKUP_HOST_PARTS = (1, 0, 3, 2)


def bits_at(values, positions):
    values = np.asarray(values, dtype=np.int64)
    return np.stack([(values >> p) & 1 for p in positions], axis=-1)


def bits_to_int(bits):
    bits = np.asarray(bits, dtype=np.int64)
    k = bits.shape[-1]
    weights = (1 << np.arange(k - 1, -1, -1, dtype=np.int64))
    return (bits * weights).sum(axis=-1)


def set_bits(values, assignments):
    out = np.asarray(values, dtype=np.int64).copy()
    for position, bits in assignments:
        mask = 1 << position
        out = (out & ~mask) | ((np.asarray(bits, dtype=np.int64) & 1) << position)
    return out


def generate_watermark(ll_values, block_indices):
    ll = np.asarray(ll_values, dtype=np.int64).reshape(-1)
    idx = np.asarray(block_indices, dtype=np.int64).reshape(-1)
    b = bits_at(ll, (8, 7, 6, 5, 4, 3, 2))
    m1, m2, m3, m4, m5, m6, m7 = (b[:, i] for i in range(7))

    r1 = bits_to_int(np.stack((m1, m2, m3), axis=1))
    r2 = bits_to_int(np.stack((m2, m3, m4), axis=1))
    r3 = bits_to_int(np.stack((m5, m6, m7), axis=1))
    r5 = (np.abs(r1 - r3) <= np.abs(r2 - r3)).astype(np.int64)
    wr = np.stack((m1, m2, m3, m4, r5), axis=1)

    a1 = r1
    a2 = bits_to_int(np.stack((m4, m5), axis=1))
    mi1 = idx % 2
    mi2 = idx % 3
    a1_bit = (a1 >> mi1) & 1
    a2_bit = (a2 >> mi2) & 1
    a1_bit2 = (a1 >> mi2) & 1
    a1_auth = (1 - ((a1 + a2) & 1)).astype(np.int64)
    a2_auth = a1_bit ^ a2_bit
    a3_auth = a1_bit ^ a1_bit2
    wa = np.stack((a1_auth, a2_auth, a3_auth), axis=1)
    return wr, wa


def recover_ll(wr):
    r1, r2, r3, r4, r5 = (int(v) for v in wr)
    tail = (r1, r2, r3) if r5 == 1 else (r2, r3, r4)
    return int(bits_to_int(np.array((r1, r2, r3, r4, *tail, 0, 0))))


def psnr(original, compared):
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(compared, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def ssim(original, compared):
    first = np.asarray(original, dtype=np.float32)
    second = np.asarray(compared, dtype=np.float32)
    if first.ndim == 3:
        first = cv2.cvtColor(np.clip(first, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        second = cv2.cvtColor(np.clip(second, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = (11, 11)
    sigma = 1.5
    mu_x = cv2.GaussianBlur(first, kernel, sigma)
    mu_y = cv2.GaussianBlur(second, kernel, sigma)
    mu_xx = cv2.GaussianBlur(first * first, kernel, sigma)
    mu_yy = cv2.GaussianBlur(second * second, kernel, sigma)
    mu_xy = cv2.GaussianBlur(first * second, kernel, sigma)
    sigma_x2 = mu_xx - mu_x * mu_x
    sigma_y2 = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerator / denominator))


def print_comparison_table(rows):
    name_width = max(len(row[0]) for row in rows)
    value_width = 12
    line = "+" + "-" * (name_width + 2) + "+" + "-" * (value_width + 2) + "+" + "-" * (value_width + 2) + "+"
    print(line)
    print(f"| {'Metric':<{name_width}} | {'Baseline':>{value_width}} | {'Improved':>{value_width}} |")
    print(line)
    for name, baseline, improved in rows:
        print(f"| {name:<{name_width}} | {baseline:>{value_width}} | {improved:>{value_width}} |")
    print(line)


def pad_to_multiple(image, multiple=4):
    h, w = image.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, pad_h, pad_w


def unpad(image, pad_h, pad_w):
    h, w = image.shape[:2]
    if pad_h:
        image = image[:-pad_h, :]
    if pad_w:
        image = image[:, :-pad_w]
    return image


def save_exact(path, image):
    shifted = np.asarray(image, dtype=np.int32) + 32768
    cv2.imwrite(str(path), shifted.astype(np.uint16))


def load_exact(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return raw.astype(np.int32) - 32768


def save_preview(path, image):
    out = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


class SelfEmbeddedWatermarker:
    def __init__(self, shape, seed=2025):
        self.height, self.width = shape
        if self.height % 4 or self.width % 4:
            raise ValueError("padded image dimensions must be divisible by 4")
        self.rows = self.height // 4
        self.cols = self.width // 4
        self.blocks_per_part = self.rows * self.cols
        self.permutations = self._build_permutations(seed)
        self.backup_permutations = self._build_backup_permutations(seed)

    def _build_permutations(self, seed):
        rng = np.random.default_rng(seed)
        return tuple(
            (rng.permutation(self.blocks_per_part), rng.permutation(self.blocks_per_part))
            for _ in range(4)
        )

    def _build_backup_permutations(self, seed):
        rng = np.random.default_rng(seed + 977)
        size = self.blocks_per_part
        result = []
        for _ in range(4):
            first = rng.permutation(size)
            if size > 1:
                while np.any(first == np.arange(size)):
                    first = rng.permutation(size)
            result.append((first, rng.permutation(size)))
        return tuple(result)

    def backup_host_parts(self, part):
        opposite = BACKUP_HOST_PARTS[part]
        return PARTNERS[opposite][0], PARTNERS[opposite][1]

    def _block_averages(self, image):
        blocks = np.asarray(image, dtype=np.int64).reshape(self.height // 2, 2, self.width // 2, 2)
        totals = blocks.sum(axis=(1, 3))
        return self._split((totals + 2) // 4)

    def _extract_backup(self, tamper, coefficient_parts):
        _, lh_parts, hl_parts, _ = coefficient_parts
        buckets = {}
        intact = {}
        for part in range(4):
            first_part, second_part = self.backup_host_parts(part)
            first_index, second_index = self.backup_permutations[part]
            bit0 = (lh_parts[first_part].flat[first_index] >> 3) & 1
            bit1 = (hl_parts[second_part].flat[second_index] >> 3) & 1
            buckets[part] = (bit0 | (bit1 << 1)).astype(np.int64)
            intact[part] = (~tamper[first_part][first_index]) & (~tamper[second_part][second_index])
        return buckets, intact

    @staticmethod
    def _correct_to_bucket(value, bucket):
        mean_value = np.mean(value)
        guess = int(np.clip(np.rint(mean_value), 0, 255)) // 64
        if guess == bucket:
            return np.clip(value, 0, 255)
        detail = value - mean_value
        return np.clip((bucket * 64 + 32) + detail, 0, 255)

    @staticmethod
    def _neighbour_average(blocks, row, col):
        values = []
        rows, cols = blocks.shape[:2]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(row) + dr, int(col) + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    value = blocks[nr, nc]
                    if not np.isnan(value).any():
                        values.append(value)
        if not values:
            return None
        return np.mean(values, axis=0)

    def _forward_dwt(self, image):
        h, w = image.shape
        blocks = image.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).astype(np.int64)
        p00 = blocks[..., 0, 0]
        p01 = blocks[..., 0, 1]
        p10 = blocks[..., 1, 0]
        p11 = blocks[..., 1, 1]
        d0 = p00 - p01
        d1 = p10 - p11
        s0 = p01 + (d0 >> 1)
        s1 = p11 + (d1 >> 1)
        dv = s0 - s1
        ll = (s1 + (dv >> 1)) << 1
        return ll, d0, d1, dv

    def _inverse_dwt(self, ll, lh, hl, hh):
        average = ll >> 1
        s1 = average - (hh >> 1)
        s0 = s1 + hh
        p11 = s1 - (hl >> 1)
        p10 = p11 + hl
        p01 = s0 - (lh >> 1)
        p00 = p01 + lh
        blocks = np.stack((p00, p01, p10, p11), axis=-1).reshape(ll.shape[0], ll.shape[1], 2, 2)
        return blocks.transpose(0, 2, 1, 3).reshape(self.height, self.width)

    def _split(self, coefficients):
        r, c = self.rows, self.cols
        return {
            0: coefficients[:r, :c],
            1: coefficients[:r, c:],
            2: coefficients[r:, :c],
            3: coefficients[r:, c:],
        }

    def _merge(self, parts):
        r, c = self.rows, self.cols
        merged = np.empty((2 * r, 2 * c), dtype=np.int64)
        merged[:r, :c] = parts[0]
        merged[:r, c:] = parts[1]
        merged[r:, :c] = parts[2]
        merged[r:, c:] = parts[3]
        return merged

    def _global_indices(self, part):
        return part * self.blocks_per_part + np.arange(self.blocks_per_part)

    def _block_position(self, part, local_index):
        row, col = divmod(int(local_index), self.cols)
        if part == 0:
            return row, col
        if part == 1:
            return row, self.cols + col
        if part == 2:
            return self.rows + row, col
        return self.rows + row, self.cols + col

    def embed_once(self, image):
        ll, lh, hl, hh = self._forward_dwt(image)
        ll_parts = self._split(ll)
        lh_parts = self._split(lh)
        hl_parts = self._split(hl)
        hh_parts = self._split(hh)
        local = np.arange(self.blocks_per_part)

        for part in range(4):
            wr, wa = generate_watermark(ll_parts[part].reshape(-1), self._global_indices(part))
            q1, q2 = PARTNERS[part]
            t1, t2 = self.permutations[part]

            ll_parts[part].flat[local] = set_bits(
                ll_parts[part].flat[local], ((3, wa[:, 0]), (2, wa[:, 1]), (1, wa[:, 2]))
            )
            lh_parts[q1].flat[t1] = set_bits(
                lh_parts[q1].flat[t1], ((0, wr[:, 0]), (1, wr[:, 1]), (2, wr[:, 2]))
            )
            hh_parts[q1].flat[t1] = set_bits(
                hh_parts[q1].flat[t1], ((2, wr[:, 3]), (3, wr[:, 4]))
            )
            hl_parts[q2].flat[t2] = set_bits(
                hl_parts[q2].flat[t2], ((0, wr[:, 0]), (1, wr[:, 1]), (2, wr[:, 2]))
            )
            hh_parts[q2].flat[t2] = set_bits(
                hh_parts[q2].flat[t2], ((0, wr[:, 3]), (1, wr[:, 4]))
            )

        averages = self._block_averages(image)
        for part in range(4):
            bucket = np.clip(averages[part].reshape(-1) >> 6, 0, 3)
            first_part, second_part = self.backup_host_parts(part)
            first_index, second_index = self.backup_permutations[part]
            lh_parts[first_part].flat[first_index] = set_bits(
                lh_parts[first_part].flat[first_index], ((3, bucket & 1),)
            )
            hl_parts[second_part].flat[second_index] = set_bits(
                hl_parts[second_part].flat[second_index], ((3, (bucket >> 1) & 1),)
            )

        return self._inverse_dwt(
            self._merge(ll_parts),
            self._merge(lh_parts),
            self._merge(hl_parts),
            self._merge(hh_parts),
        )

    def embed(self, image):
        current = np.asarray(image, dtype=np.int64).copy()
        for _ in range(8):
            candidate = self.embed_once(current)
            if np.array_equal(candidate, current):
                break
            current = candidate
        return current

    def _extract(self, image):
        ll, lh, hl, hh = self._forward_dwt(image)
        ll_parts = self._split(ll)
        lh_parts = self._split(lh)
        hl_parts = self._split(hl)
        hh_parts = self._split(hh)
        local = np.arange(self.blocks_per_part)
        tamper = {}
        wr1 = {}
        wr2 = {}

        for part in range(4):
            ll_values = ll_parts[part].flat[local]
            embedded = bits_at(ll_values, (3, 2, 1))
            _, regenerated = generate_watermark(ll_values, self._global_indices(part))
            tamper[part] = np.any(embedded != regenerated, axis=1)

            q1, q2 = PARTNERS[part]
            t1, t2 = self.permutations[part]
            first = np.concatenate(
                (bits_at(lh_parts[q1].flat[t1], (0, 1, 2)), bits_at(hh_parts[q1].flat[t1], (2, 3))),
                axis=1,
            )
            second = np.concatenate(
                (bits_at(hl_parts[q2].flat[t2], (0, 1, 2)), bits_at(hh_parts[q2].flat[t2], (0, 1))),
                axis=1,
            )
            wr1[part] = first
            wr2[part] = second

        return tamper, wr1, wr2, (ll_parts, lh_parts, hl_parts, hh_parts)

    def _postprocess(self, tamper):
        full = np.zeros((2 * self.rows, 2 * self.cols), dtype=np.uint8)
        for part in range(4):
            for local_index, flag in enumerate(tamper[part]):
                row, col = self._block_position(part, local_index)
                full[row, col] = 255 if flag else 0

        kernel = np.ones((3, 3), dtype=np.uint8)
        full = cv2.morphologyEx(full, cv2.MORPH_OPEN, kernel)
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((full > 0).astype(np.uint8), 8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 2:
                full[labels == label] = 0

        cleaned = {part: np.zeros(self.blocks_per_part, dtype=bool) for part in range(4)}
        for part in range(4):
            for local_index in range(self.blocks_per_part):
                row, col = self._block_position(part, local_index)
                cleaned[part][local_index] = full[row, col] > 0
        return cleaned, full > 0

    def _single_block_from_coefficients(self, ll, lh, hl, hh):
        average = int(ll) >> 1
        s1 = average - (int(hh) >> 1)
        s0 = s1 + int(hh)
        p11 = s1 - (int(hl) >> 1)
        p10 = p11 + int(hl)
        p01 = s0 - (int(lh) >> 1)
        p00 = p01 + int(lh)
        return np.array((p00, p01, p10, p11), dtype=np.float64).reshape(2, 2)

    def recover(self, image, use_backup=True, return_assisted=False):
        tamper, wr1, wr2, coefficients = self._extract(image)
        tamper, tamper_mask = self._postprocess(tamper)
        ll_parts, lh_parts, hl_parts, hh_parts = coefficients
        buckets, backup_intact = self._extract_backup(tamper, coefficients) if use_backup else ({}, {})

        block_rows = 2 * self.rows
        block_cols = 2 * self.cols
        attacked_blocks = image.reshape(block_rows, 2, block_cols, 2).transpose(0, 2, 1, 3)
        recovered_blocks = attacked_blocks.astype(np.float64).copy()
        assisted_mask = np.zeros((block_rows, block_cols), dtype=bool)
        unresolved = []

        for part in range(4):
            q1, q2 = PARTNERS[part]
            t1, t2 = self.permutations[part]
            for local_index in np.flatnonzero(tamper[part]):
                candidate_first = self._single_block_from_coefficients(
                    recover_ll(wr1[part][local_index]),
                    lh_parts[part].flat[local_index],
                    hl_parts[part].flat[local_index],
                    hh_parts[part].flat[local_index],
                )
                candidate_second = self._single_block_from_coefficients(
                    recover_ll(wr2[part][local_index]),
                    lh_parts[part].flat[local_index],
                    hl_parts[part].flat[local_index],
                    hh_parts[part].flat[local_index],
                )
                first_ok = not tamper[q1][t1[local_index]]
                second_ok = not tamper[q2][t2[local_index]]
                block_row, block_col = self._block_position(part, local_index)

                if first_ok and second_ok:
                    recovered_blocks[block_row, block_col] = (candidate_first + candidate_second) / 2
                elif first_ok:
                    recovered_blocks[block_row, block_col] = candidate_first
                elif second_ok:
                    recovered_blocks[block_row, block_col] = candidate_second
                else:
                    recovered_blocks[block_row, block_col] = np.nan
                    unresolved.append((part, int(local_index), block_row, block_col))

        if use_backup:
            for part, local_index, row, col in unresolved:
                if backup_intact[part][local_index]:
                    estimate = self._neighbour_average(recovered_blocks, row, col)
                    if estimate is not None:
                        recovered_blocks[row, col] = self._correct_to_bucket(
                            estimate, int(buckets[part][local_index])
                        )
                        assisted_mask[row, col] = True

        pending = np.argwhere(np.isnan(recovered_blocks[..., 0, 0]))
        while pending.size:
            progressed = False
            for row, col in pending:
                neighbours = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = int(row) + dr, int(col) + dc
                        if 0 <= nr < block_rows and 0 <= nc < block_cols:
                            value = recovered_blocks[nr, nc]
                            if not np.isnan(value).any():
                                neighbours.append(value)
                if neighbours:
                    recovered_blocks[row, col] = np.mean(neighbours, axis=0)
                    progressed = True
            if not progressed:
                fallback = np.nanmean(recovered_blocks)
                if np.isnan(fallback):
                    fallback = float(np.mean(attacked_blocks))
                recovered_blocks[np.isnan(recovered_blocks)] = fallback
                break
            pending = np.argwhere(np.isnan(recovered_blocks[..., 0, 0]))

        recovered = np.rint(recovered_blocks).astype(np.int32)
        restored = recovered.transpose(0, 2, 1, 3).reshape(self.height, self.width)
        attacked = attacked_blocks.astype(np.int32).transpose(0, 2, 1, 3).reshape(self.height, self.width)
        pixel_mask = np.repeat(np.repeat(tamper_mask, 2, axis=0), 2, axis=1)
        if restored.ndim == 3:
            final = np.where(pixel_mask[..., None] > 0, restored, attacked)
        else:
            final = np.where(pixel_mask > 0, restored, attacked)
        if return_assisted:
            return final, tamper_mask, assisted_mask
        return final, tamper_mask


def embed_image(image, seed=2025):
    padded, pad_h, pad_w = pad_to_multiple(image)
    watermarker = SelfEmbeddedWatermarker(padded.shape[:2], seed)
    if padded.ndim == 3:
        result = np.empty_like(padded, dtype=np.int32)
        for channel in range(padded.shape[2]):
            result[..., channel] = watermarker.embed(padded[..., channel])
    else:
        result = watermarker.embed(padded)
    return unpad(result, pad_h, pad_w).astype(np.int32)


def recover_image(image, seed=2025, use_backup=True, return_assisted=False):
    padded, pad_h, pad_w = pad_to_multiple(image)
    watermarker = SelfEmbeddedWatermarker(padded.shape[:2], seed)
    if padded.ndim == 3:
        restored = np.empty_like(padded, dtype=np.int32)
        maps = []
        assisted_maps = []
        for channel in range(padded.shape[2]):
            result = watermarker.recover(padded[..., channel], use_backup, return_assisted)
            channel_restored, channel_map = result[:2]
            restored[..., channel] = channel_restored
            maps.append(channel_map)
            if return_assisted:
                assisted_maps.append(result[2])
        tamper_mask = np.logical_or.reduce(maps)
        assisted_mask = np.logical_or.reduce(assisted_maps) if assisted_maps else None
    else:
        result = watermarker.recover(padded, use_backup, return_assisted)
        restored, tamper_mask = result[:2]
        assisted_mask = result[2] if return_assisted else None
    restored = unpad(restored, pad_h, pad_w).astype(np.int32)
    tamper_mask = unpad(tamper_mask, pad_h // 2, pad_w // 2)
    if return_assisted:
        return restored, tamper_mask, unpad(assisted_mask, pad_h // 2, pad_w // 2)
    return restored, tamper_mask


def apply_crop(image, ratio=0.4, center=True, seed=0):
    h, w = image.shape[:2]
    side = max(2, int(round(np.sqrt(ratio * h * w))))
    side = min(side, h, w)
    if center:
        y0 = (h - side) // 2
        x0 = (w - side) // 2
    else:
        rng = np.random.default_rng(seed)
        y0 = int(rng.integers(0, h - side + 1))
        x0 = int(rng.integers(0, w - side + 1))
    attacked = image.copy()
    attacked[y0:y0 + side, x0:x0 + side] = 0
    ground_truth = np.zeros((h, w), dtype=bool)
    ground_truth[y0:y0 + side, x0:x0 + side] = True
    return attacked, ground_truth


def block_ground_truth(pixel_mask):
    h, w = pixel_mask.shape
    blocks = pixel_mask.reshape(h // 2, 2, w // 2, 2)
    return np.any(blocks, axis=(1, 3))


def detection_scores(detected, truth):
    detected = np.asarray(detected, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = np.count_nonzero(detected & truth)
    tn = np.count_nonzero(~detected & ~truth)
    fp = np.count_nonzero(detected & ~truth)
    fn = np.count_nonzero(~detected & truth)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    sensitivity = tp / max(1, tp + fn)
    return accuracy, sensitivity


def make_sample_image(size=256, grayscale=False):
    yy, xx = np.mgrid[0:size, 0:size]
    base = ((xx + yy) * 255 / (2 * size)).astype(np.uint8)
    if grayscale:
        image = np.ascontiguousarray(base)
        cv2.circle(image, (size // 3, size // 3), size // 8, 230, -1)
        cv2.rectangle(image, (size // 2, size // 2), (size - 10, size - 10), 120, -1)
        return image
    g = (yy * 255 // size).astype(np.uint8)
    b = (xx * 255 // size).astype(np.uint8)
    image = np.ascontiguousarray(np.dstack((base, g, b)))
    cv2.circle(image, (size // 3, size // 3), size // 8, (230, 230, 230), -1)
    cv2.rectangle(image, (size // 2, size // 2), (size - 10, size - 10), (40, 200, 120), -1)
    return image


def main():
    parser = argparse.ArgumentParser(description="Self-embedded dual watermarking recovery demo")
    parser.add_argument("--image", type=Path, help="source image; omitted means synthetic sample")
    parser.add_argument("--output", type=Path, default=Path("recovery_output"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--crop", type=float, default=0.4)
    parser.add_argument("--attacked", type=Path, help="your own cropped watermarked image (16-bit PNG)")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--compare", action="store_true", help="also run baseline without tier-3 backup")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.image is None:
        source = make_sample_image(grayscale=args.grayscale)
        grayscale = args.grayscale
    else:
        flags = cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_COLOR
        source = cv2.imread(str(args.image), flags)
        if source is None:
            raise FileNotFoundError(args.image)
        grayscale = source.ndim == 2

    original = source.astype(np.int32)
    watermarked = embed_image(original, args.seed)
    if args.attacked is None:
        attacked, attacked_pixels = apply_crop(watermarked, args.crop, center=True)
    else:
        attacked = load_exact(args.attacked)
        if attacked.shape != watermarked.shape:
            raise ValueError(f"attacked image shape {attacked.shape} != watermarked shape {watermarked.shape}")
        if attacked.min() < -1000 or attacked.max() > 1000:
            raise ValueError("use the saved watermarked_16bit.png, not an 8-bit preview, for --attacked")
        if watermarked.ndim == 3:
            attacked_pixels = np.any(attacked != watermarked, axis=2)
        else:
            attacked_pixels = attacked != watermarked
    restored, detected, assisted = recover_image(attacked, args.seed, return_assisted=True)
    baseline = None
    if args.compare:
        baseline, _ = recover_image(attacked, args.seed, use_backup=False)
    padded_truth, truth_pad_h, truth_pad_w = pad_to_multiple(attacked_pixels, 4)
    truth = unpad(block_ground_truth(padded_truth), truth_pad_h // 2, truth_pad_w // 2)
    accuracy, sensitivity = detection_scores(detected, truth)

    save_exact(args.output / "watermarked_16bit.png", watermarked)
    save_preview(args.output / "watermarked_preview.png", watermarked)
    save_preview(args.output / "attacked.png", attacked)
    save_preview(args.output / "detected_map.png", detected.astype(np.uint8) * 255)
    save_preview(args.output / "recovered.png", restored)
    if baseline is not None:
        save_preview(args.output / "recovered_baseline.png", baseline)
    if not grayscale:
        save_preview(args.output / "original.png", original)

    print(f"image: {source.shape[1]}x{source.shape[0]}, channels={1 if grayscale else 3}")
    print(f"watermarked PSNR: {psnr(original, watermarked):.2f} dB")
    print(f"detection accuracy: {accuracy * 100:.2f}%")
    print(f"detection sensitivity: {sensitivity * 100:.2f}%")
    if baseline is not None:
        assisted_pixels = np.repeat(np.repeat(assisted, 2, axis=0), 2, axis=1)[:original.shape[0], :original.shape[1]]
        rows = [
            ("Watermarked PSNR (dB)", f"{psnr(original, watermarked):.2f}", f"{psnr(original, watermarked):.2f}"),
            ("Detection accuracy (%)", f"{accuracy * 100:.2f}", f"{accuracy * 100:.2f}"),
            ("Detection sensitivity (%)", f"{sensitivity * 100:.2f}", f"{sensitivity * 100:.2f}"),
            ("Recovered PSNR whole (dB)", f"{psnr(original, baseline):.2f}", f"{psnr(original, restored):.2f}"),
            ("Recovered SSIM whole", f"{ssim(original, baseline):.4f}", f"{ssim(original, restored):.4f}"),
            ("Tier-3 assisted blocks", "-", f"{int(assisted.sum())} / {int(detected.sum())}"),
        ]
        if np.any(assisted_pixels):
            rows.append((
                "Tier-3 region PSNR (dB)",
                f"{psnr(original[assisted_pixels], baseline[assisted_pixels]):.2f}",
                f"{psnr(original[assisted_pixels], restored[assisted_pixels]):.2f}",
            ))
        print()
        print_comparison_table(rows)
        with open(args.output / "comparison.csv", "w") as handle:
            handle.write("Metric,Baseline,Improved\n")
            for name, baseline_value, improved_value in rows:
                handle.write(f"{name},{baseline_value},{improved_value}\n")
        print(f"table saved to {(args.output / 'comparison.csv').resolve()}")
    else:
        print(f"recovered PSNR: {psnr(original, restored):.2f} dB")
    print(f"outputs written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

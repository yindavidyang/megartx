import copy
import math
import unittest

import prefill_cache_reference as reference


def payload(p):
    # Distinct, nonconstant K/V detect mask, missing-row and V-only errors.
    return ((float((p % 5) - 2) / 4, float((p % 3) - 1) / 2),
            (float(p % 11), float((p * 3) % 7)))


def dense_mask_oracle(position, sliding):
    # Independently traverse the complete prefix and apply a dense causal mask.
    weighted = []
    for key_position in range(position + 1):
        if sliding and position - key_position >= 1024:
            continue
        k, v = payload(key_position)
        score = .5 * k[0] - .25 * k[1]  # score scale 1, no sqrt(D)
        weighted.append((math.exp(score), v))
    denominator = math.fsum(w for w, _ in weighted)
    return tuple(math.fsum(w * v[i] for w, v in weighted) / denominator for i in (0, 1))


class PrefillCacheTests(unittest.TestCase):
    def test_causal_sliding_inclusive_boundaries(self):
        for p, first, count in [(1022, 0, 1023), (1023, 0, 1024),
                                (1024, 1, 1024), (1025, 2, 1024), (3073, 2050, 1024)]:
            keys = reference.visible_positions(p, True)
            self.assertEqual((keys.start, keys.stop, len(keys)), (first, p + 1, count))
            self.assertEqual(list(reference.visible_positions(p, False)), list(range(p + 1)))

    def test_multirow_chunk_requires_more_than_its_final_window(self):
        required = reference.chunk_required_positions(1024, 1280, True)
        self.assertEqual((required.start, required.stop, len(required)), (1, 1280, 1279))
        final_window_only = {p: payload(p) for p in range(256, 1280)}
        with self.assertRaisesRegex(ValueError, "recycled or absent"):
            reference.ideal_attention((.5, -.25), 1024, final_window_only, True)
        staged = {p: payload(p) for p in required}
        self.assertEqual(len(reference.visible_positions(1279, True)), 1024)
        actual = reference.ideal_attention((.5, -.25), 1024, staged, True)
        expected = dense_mask_oracle(1024, True)
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=12)

    def test_chunked_attention_matches_continuous_mask_at_tails_and_wraps(self):
        for sliding in (False, True):
            for chunk in (255, 256, 1024, 2051):
                cache = {}
                checked = {0, 254, 255, 256, 1022, 1023, 1024, 1025, 2049, 2050}
                for start in range(0, 2051, chunk):
                    end = min(start + chunk, 2051)
                    cache.update((p, payload(p)) for p in range(start, end))
                    for p in sorted(checked & set(range(start, end))):
                        actual = reference.ideal_attention((.5, -.25), p, cache, sliding)
                        expected = dense_mask_oracle(p, sliding)
                        for a, b in zip(actual, expected):
                            self.assertAlmostEqual(a, b, places=12)
                    # Recycling happens only after all queries in this chunk finish.
                    if sliding:
                        cache = {p: value for p, value in cache.items() if p >= end - 1024}
                self.assertEqual(set(cache), set(range(1027, 2051) if sliding else range(2051)))

    def test_page_mapping_preserves_logical_positions_and_capacity(self):
        table = [7, 2, 99]
        self.assertEqual(reference.page_address(15, table, 16), (7, 15))
        self.assertEqual(reference.page_address(16, table, 16), (2, 0))
        self.assertEqual(reference.page_address(47, table, 16), (99, 15))
        for p, pages, size in [(48, table, 16), (-1, table, 16), (0, [7, 7], 16), (0, table, 0)]:
            with self.assertRaises(ValueError):
                reference.page_address(p, pages, size)

    def test_global_proportional_rope_uses_full_width_pairs_and_absolute_positions(self):
        local, global_ = reference.rotary_pairs(True), reference.rotary_pairs(False)
        self.assertEqual((len(local), local[0], local[-1]), (128, (0, 128), (127, 255)))
        self.assertEqual((len(global_), global_[0], global_[-1]), (64, (0, 256), (63, 319)))
        angle = reference.rotary_angles(1025, False)
        self.assertAlmostEqual(angle[63], 1025 * 1000000 ** (-126 / 512), places=12)
        # A wrapped storage slot cannot be substituted as the rotary position.
        self.assertNotEqual(angle, reference.rotary_angles(1025 % 1024, False))
        self.assertNotEqual(angle[63], 1025 * 1000000 ** (-126 / 128))

    def test_all_layer_handoff_has_distinct_processed_kv_and_output_reserve(self):
        for n in (257, 1023, 1024, 1025, 2051):
            cache = reference.tagged_handoff(n)
            self.assertTrue(reference.require_handoff(cache, n))
            self.assertEqual(cache["next_position"], n)
            self.assertEqual(cache["capacity"], n + 256)
            self.assertEqual(len(cache["layers"][0]["entries"]), min(n, 1024))
            self.assertEqual(len(cache["layers"][5]["entries"]), n)

    def test_partial_failure_stale_length_capacity_and_kv_alias_rejected(self):
        cache = reference.tagged_handoff(1025)
        mutations = [lambda c: c.update(complete=False), lambda c: c.update(poisoned=True),
                     lambda c: c.update(committed_length=1024), lambda c: c.update(next_position=1),
                     lambda c: c.update(capacity=1025 + 255), lambda c: c.update(kv_dtype="fp8"),
                     lambda c: c["layers"].pop(),
                     lambda c: c["layers"][0]["entries"].pop(1),
                     lambda c: c["layers"][5]["entries"].pop(0),
                     lambda c: c["layers"][5]["entries"].update({1024: ((5, 1024, "processed_k"), (5, 1024, "processed_k"))}),
                     lambda c: c["layers"][0]["entries"].update({1024: ((0, 0, "processed_k"), (0, 0, "processed_v"))})]
        for mutate in mutations:
            changed = copy.deepcopy(cache)
            mutate(changed)
            with self.assertRaises(ValueError):
                reference.require_handoff(changed, 1025)

    def test_wrong_window_and_value_permutation_change_the_toy_result(self):
        position = 1024
        good = {p: payload(p) for p in range(position + 1)}
        correct = reference.ideal_attention((.5, -.25), position, good, True)
        global_result = reference.ideal_attention((.5, -.25), position, good, False)
        wrong_v = {p: (payload(p)[0], payload((p + 1) % 1025)[1]) for p in good}
        wrong_result = reference.ideal_attention((.5, -.25), position, wrong_v, True)
        self.assertNotEqual(correct, global_result)
        self.assertNotEqual(correct, wrong_result)


if __name__ == "__main__":
    unittest.main()

# Layer-0 QKV provenance guard revalidation

CPU review of PR #3 at `704298b00df8a114046dcee86600be00a99c2f07`
found a missing cached Q-weight hash comparison. The checker could report
original Q provenance verified after checking only the full run. This was a
checker defect; it does not show that the retained live weights were wrong.

The repair requires Q provenance on both schema-2 paths, compares their hashes
before projection reads, and checks every supplied Q/K/V hash against the
original checkpoint. A shared verification flag now requires both paths' hash
records. Historical schema-1 Q comparisons retain their weaker diagnostic scope.

Six added CPU regression tests cover a changed cached Q hash, missing Q records
on either or both paths, shared incorrect Q hashes, matched original hashes,
historical schema-1 captures and partially recorded historical Q provenance.
The reference suite passes **341 tests** locally with NumPy 2.4.4 and on the
pinned target with NumPy 2.3.5. All **32 scaffold tests**, structural config
validation, format self-checks and Python compilation pass. Exact-head CI is
linked from the draft PR.

The repaired reader revalidated the immutable `native-full-v2` /
`native-cached-v2-a` pair and original checkpoint revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6`. It reports:

| Check | Result |
| --- | --- |
| Both paths' original Q/K/V hashes verified | All three true |
| Position 31 first discrepancy | None |
| Position 32 first discrepancy | Raw QKV output |
| Position 32 raw QKV differences | 18 / 8192 |
| Writer, stored row and final capture associations | Exact |
| Altered cached Q record, or missing full/cached/both Q records | All four rejected before original projection reads |
| Native accumulation / whole-model full-cached acceptance | Both false |

The revalidation reader SHA256 is
`d5b96dd487753e151aa7bbf5e35091b1a02c8f1d566bc1a00d30240b7589f946`.
The new private report SHA256 is
`907ddf578dfe33365c1ba9c01a4d0c568fa7b3fcfe1bbec60f2c84132b2b06bd`.
Original full/cached binding SHA256 values are respectively
`91a450ed926650fabaa1bb74fe70a704de31ae496dd67f92021183170d9504dd` and
`ca11f923dccf51a4da2a7ab409cbe6d16dca832e29231dcff379a2d5de97809d`.

Revalidation used CPU reads and in-memory corruption controls. The original
checkpoint, runtime and evidence bundles remain unchanged. New QKV arithmetic
qualification artifacts are retained separately from this review repair.

"""CPU tests of fixture/capture rejection; synthetic captures are not GPU evidence."""
from pathlib import Path
import struct
import tempfile
import unittest

import m1_kernel_fixture as fixture


class M1KernelFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.fixtures, self.captures = root / 'fixtures', root / 'captures'
        fixture.create(self.fixtures)
        self.captures.mkdir()
        for i in range(3):
            expected = (self.fixtures / f'case{i}.expected').read_bytes()
            raw = (self.fixtures / f'case{i}.input').read_bytes()
            for backend in ('fused', 'control'):
                (self.captures / f'case{i}.{backend}').write_bytes(expected + raw)

    def mutate(self, path, offset):
        data = bytearray(path.read_bytes())
        data[offset] ^= 1
        path.write_bytes(data)

    def test_cpu_replay_is_byte_exact_but_does_not_attest_producer(self):
        report = fixture.verify(self.fixtures, self.captures)
        self.assertTrue(report['byte_equal'])
        self.assertFalse(report['producer_identity_verified'])
        self.assertFalse(report['candidate_selectable'])
        self.assertFalse(report['full_native_descriptor_abi_verified'])
        a, b = [(self.fixtures / f'case{i}.input').read_bytes() for i in (0, 1)]
        self.assertEqual(a[:32], b[:32])  # Repeated routes with changed byte payloads.
        self.assertNotEqual(a[32:], b[32:])

    def test_corrupt_maps_aq_weight_padding_guard_or_input_is_rejected(self):
        path = self.captures / 'case2.fused'
        original = path.read_bytes()
        # Includes an inactive SF byte, both outer SF guards and unchanged inputs.
        for offset in (0, 32, 64, 1096, 12360, 12392, 12428,
                       fixture.OUTPUT_BYTES - 1, fixture.OUTPUT_BYTES + 1500):
            with self.subTest(offset=offset):
                self.mutate(path, offset)
                with self.assertRaises(ValueError):
                    fixture.verify(self.fixtures, self.captures)
                path.write_bytes(original)

    def test_expected_file_and_control_are_independently_checked(self):
        path = self.fixtures / 'case0.expected'
        original = path.read_bytes()
        self.mutate(path, 1096)
        with self.assertRaisesRegex(ValueError, 'expected fixture changed'):
            fixture.verify(self.fixtures, self.captures)
        path.write_bytes(original)
        self.mutate(self.captures / 'case1.control', 12360)
        with self.assertRaisesRegex(ValueError, 'native bytes/input/guard mismatch'):
            fixture.verify(self.fixtures, self.captures)

    def test_route_change_and_nonfinite_weight_are_rejected(self):
        path = self.fixtures / 'case0.input'
        original = path.read_bytes()
        raw = bytearray(original)
        raw[:4] = raw[4:8]  # Duplicate route is outside the declared fixture.
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'fixture route/extent changed'):
            fixture.verify(self.fixtures, self.captures)
        raw = bytearray(original)
        raw[32:36] = struct.pack('<I', 0x7fc00000)
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'finite FP32'):
            fixture.verify(self.fixtures, self.captures)

    def test_truncated_or_symlinked_files_are_rejected(self):
        path = self.captures / 'case0.control'
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, 'exact extent'):
            fixture.verify(self.fixtures, self.captures)
        path.unlink()
        path.symlink_to(self.captures / 'case0.fused')
        with self.assertRaisesRegex(ValueError, 'regular file'):
            fixture.verify(self.fixtures, self.captures)


if __name__ == '__main__':
    unittest.main()

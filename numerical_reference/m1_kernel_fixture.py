"""Three bounded, synthetic inputs and byte expectations for the actual M1 CUDA experiment."""
import argparse
import json
from pathlib import Path
import stat
import struct

import m1_preparation_reference as oracle

ROUTES = ((127, 0, 82, 42, 126, 7, 89, 12), (127, 0, 82, 42, 126, 7, 89, 12),
          (26, 20, 27, 23, 21, 25, 22, 24))
OUTPUT_BYTES = 2896040

def contexts():
    tables = struct.pack('<128f', *([1.0] * 128))
    return tuple(oracle.StageContext(stage, False, 'none', tables, struct.pack('<f', 1.0),
                                    (tables, tables) if stage == 'fc1' else (tables,),
                                    'synthetic standalone byte fixture; no model execution')
                 for stage in ('fc1', 'fc2'))

def input_row(ids, variant=0):
    sf = bytearray([0xCD]) * 22528
    for block in range(176):
        sf[oracle.sf_coordinate(0, block, 176)] = 128 if block == 127 else (block + variant * 13) % 127
    words = (0, 0x80000000, 1, 0x007fffff, 0x3f800001, 0xbf000000, 0x7f7fffff, 0x00800000)
    weights = struct.pack('<8I', *(words[variant:] + words[:variant]))
    aq = bytes((i + variant * 17) % 256 for i in range(1408))
    return oracle.InputRow(tuple(ids), weights, aq, bytes(sf))

def expected(row, before):
    prepared = oracle.prepare(row, *contexts(), reference_abi=oracle.REFERENCE_ABI)
    after = oracle.materialize_sf(prepared, before, origin=32)
    blob = b''.join(getattr(prepared, name) for name in
                    ('slot_to_sorted', 'sorted_to_slot', 'expert_offsets', 'expanded_aq', 'permuted_weight_bits')) + after
    assert len(blob) == OUTPUT_BYTES
    return blob, after

def create(directory):
    directory = Path(directory)
    directory.mkdir(mode=0o700)
    before = b'\xA7' * 32 + b'\xD3' * 2883584 + b'\xB6' * 32
    for i, ids in enumerate(ROUTES):
        row = input_row(ids, i)
        raw = struct.pack('<8i', *ids) + row.route_weight_bits + row.packed_fp4 + row.swizzled_sf
        blob, before = expected(row, before)
        (directory / f'case{i}.input').write_bytes(raw)
        (directory / f'case{i}.expected').write_bytes(blob)
    return {'scope': 'synthetic M1 maps/AQ/SF only', 'cases': 3, 'candidate_selectable': False}

def read_exact(path, size):
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
        raise ValueError('fixture/capture must be a regular file of the exact extent')
    return path.read_bytes()

def verify(fixtures, captures):
    fixtures, captures = Path(fixtures), Path(captures)
    before = b'\xA7' * 32 + b'\xD3' * 2883584 + b'\xB6' * 32
    for i, ids in enumerate(ROUTES):
        raw = read_exact(fixtures / f'case{i}.input', 24000)
        if struct.unpack('<8i', raw[:32]) != ids:
            raise ValueError('fixture route/extent changed')
        row = oracle.InputRow(ids, raw[32:64], raw[64:1472], raw[1472:])
        blob, before = expected(row, before)  # Recompute; do not trust the native expected file.
        if read_exact(fixtures / f'case{i}.expected', OUTPUT_BYTES) != blob:
            raise ValueError('expected fixture changed')
        for backend in ('fused', 'control'):
            path = captures / f'case{i}.{backend}'
            if read_exact(path, OUTPUT_BYTES + 24000) != blob + raw:
                raise ValueError('native bytes/input/guard mismatch')
    return {'scope': 'independent CPU verification of captured bytes', 'cases': 3,
            'backends': ['fused', 'two_launch_development_control'], 'byte_equal': True,
            'producer_identity_verified': False,
            'full_native_descriptor_abi_verified': False, 'candidate_selectable': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('create'); make.add_argument('directory')
    check = commands.add_parser('verify'); check.add_argument('fixtures'); check.add_argument('captures')
    args = parser.parse_args()
    result = create(args.directory) if args.command == 'create' else verify(args.fixtures, args.captures)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

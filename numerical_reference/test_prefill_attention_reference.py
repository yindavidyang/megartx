"""Independent CPU arithmetic/domain controls; synthetic values only."""
from decimal import (Decimal, localcontext, Context, ROUND_UP, ROUND_DOWN,
                     Inexact, Rounded, Overflow)
from fractions import Fraction
from pathlib import Path
import random
import hashlib
import struct
import subprocess
import sys
import unittest

import prefill_attention_reference as ref


def bits(x):
    word=struct.unpack('<I',struct.pack('<f',x))[0]
    assert word&65535==0
    return word>>16


class SparseReferenceTests(unittest.TestCase):
    def test_imports_are_independent_stdlib_only(self):
        result=subprocess.run([sys.executable,'-S','-c',
            "import prefill_attention_reference, sys; "
            "assert not {'torch','vllm','numpy','megartx'} & set(sys.modules)"],
            cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_every_finite_bf16_word_matches_fraction_decode(self):
        for w in range(65536):
            if w&0x7f80==0x7f80:continue
            decoded=struct.unpack('<f',struct.pack('<I',w<<16))[0]
            self.assertEqual(Fraction(ref.units(w),2**133),Fraction(decoded))

    def test_exact_dot_random_fraction_and_cancellation(self):
        rng=random.Random(1234)
        for dim in (1,2,8,256,512):
            q=[bits(rng.randint(-16,16)/8) for _ in range(dim)]
            k=[bits(rng.randint(-16,16)/8) for _ in range(dim)]
            expected=sum(Fraction(ref.units(a),2**133)*Fraction(ref.units(b),2**133) for a,b in zip(q,k))
            self.assertEqual(Fraction(ref.exact_scores(q,[k])[0],2**266),expected)
        self.assertEqual(ref.exact_scores([bits(2**60),bits(1),bits(2**60)],
                                         [[bits(1),bits(1),bits(-1)]])[0],2**266)
        self.assertEqual(ref.exact_scores([1],[[1]])[0],1)

    def test_fixed_context_ignores_ambient_rounding_traps_exponents(self):
        args=([bits(1)],[[bits(0)],[bits(4)]],[[bits(0)],[bits(8)]])
        expected=ref.ideal_sparse(*args)
        for rounding in (ROUND_UP,ROUND_DOWN):
            with localcontext() as ctx:
                ctx.prec=2;ctx.rounding=rounding;ctx.Emin=-1;ctx.Emax=1
                ctx.traps[Inexact]=True;ctx.traps[Rounded]=True;ctx.traps[Overflow]=True
                self.assertEqual(ref.ideal_sparse(*args),expected)
                self.assertEqual(ref.dyadic(1,-266),ref.dyadic(1,-266))
        frozen=ref.context(96)
        self.assertEqual((frozen.prec,frozen.Emin,frozen.Emax,frozen.clamp),(96,-999999,999999,0))
        self.assertFalse(frozen.traps[Inexact]);self.assertFalse(frozen.traps[Rounded])

    def test_single_key_zero_constant_and_sparse_columns(self):
        self.assertEqual(ref.ideal_sparse([bits(3)],[[bits(4)]],[[bits(-2),bits(1)]]),(-2,1))
        self.assertEqual(ref.ideal_sparse([bits(0)],[[bits(0)]]*17,[[bits(2)]]*17),(2,))
        q=[bits(1)];k=[[bits(0)],[bits(1)]]
        vv=[[bits(x) for x in row] for row in ((1,2,3,4),(-1,4,8,2))]
        full=ref.ideal_sparse(q,k,vv)
        self.assertEqual(ref.ideal_sparse(q,k,[[row[1],row[3]] for row in vv]),(full[1],full[3]))

    def test_closed_form_wrong_scale_and_permutations(self):
        q=[bits(1)];k=[[bits(0)],[bits(4)]];v=[[bits(0)],[bits(8)]]
        actual=ref.ideal_sparse(q,k,v)[0]
        with localcontext(ref.context(128)):
            expected=8*Decimal(4).exp()/(1+Decimal(4).exp())
            self.assertLess(abs(actual-expected),Decimal('1e-120'))
            self.assertGreater(abs(actual-ref.ideal_sparse([bits(.5)],k,v)[0]),Decimal('.1'))
            self.assertGreater(abs(actual-ref.ideal_sparse(q,k,v[::-1])[0]),7)
            self.assertEqual(actual,ref.ideal_sparse(q,k[::-1],v[::-1])[0])

    def test_domain_failures_are_not_native_numerical_failures(self):
        for w in (True,-1,65536,0x7f80,0x7fc0,0xff80):
            with self.assertRaises(ValueError):ref.units(w)
        for args in (([],[[0]],[[0]]),([0],[],[]),([0],[[0,0]],[[0]]),
                     ([0],[[0]],[[0]*9]),([bits(256)],[[bits(256)],[bits(-256)]],[[0],[0]])):
            with self.assertRaises(ValueError):ref.ideal_sparse(*args)
        with self.assertRaises(ValueError):ref.ideal_sparse([0],[[0]],[[0]],deadline=0)

    def test_metrics_keep_tiny_errors_huge_relative_errors_and_zero_norm(self):
        result=ref.metrics([bits(1)],[Decimal('1e-5000')])
        self.assertGreater(Decimal(result['relative_l2_error']),Decimal('1e4000'))
        tiny=ref.metrics([0],[Decimal('1e-5000')])
        self.assertEqual(Decimal(tiny['maximum_absolute_error']),Decimal('1e-5000'))
        zero=ref.metrics([0x8000],[Decimal(0)])
        self.assertIsNone(zero['relative_l2_error'])
        self.assertEqual(zero['observed_negative_zero_count'],1)
        self.assertFalse(zero['reference_enclosure_certified'])
        self.assertIsNone(zero['rounded_ideal_bf16_diagnostics'])

    def test_reference_exact_profile_is_independent_but_matches_plan(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'src'))
        try:
            from megartx import prefill_attention_plan as p
            self.assertEqual(ref.POSITIONS,p.POSITIONS)
            for layer,(d,h,kh,qh) in ref.SPECS.items():
                self.assertEqual((d,h,kh,qh),(p.LAYERS[layer]['dim'],p.LAYERS[layer]['kv_heads'],
                                             p.LAYERS[layer]['selected_kv_heads'],p.LAYERS[layer]['q_heads']))
        finally:sys.path.pop(0)

    def test_full_480_coordinate_zero_fixture_never_grants_native_acceptance(self):
        arrays={}
        for layer,(dim,hkv,kh,qh) in ref.SPECS.items():
            for role,count in (('k',2049*len(kh)*dim),('v',2049*len(kh)*8),
                               ('q',10*len(qh)*dim),('o',10*len(qh)*8)):
                arrays[f'attention-layer-{layer:02d}-{role}.bf16']=bytes(count*2)
        report=ref.analyze_arrays(arrays)
        self.assertEqual(len(report['cases']),60)
        self.assertEqual(report['aggregate']['coordinates'],480)
        self.assertEqual(report['input_manifest'],{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
                                                   for n,b in arrays.items()})
        self.assertEqual(report['aggregate']['maximum_absolute_error'],'0')
        self.assertFalse(report['numerical_qualified']);self.assertFalse(report['native_execution_attested'])
        self.assertIsNone(report['native_arithmetic_acceptance'])
        self.assertTrue(report['external_native_storage_frontier_validation_required'])
        arrays['attention-layer-05-k.bf16']=bytes(2049*2*256*2)
        with self.assertRaises(ValueError):ref.analyze_arrays(arrays)


if __name__=='__main__':unittest.main()

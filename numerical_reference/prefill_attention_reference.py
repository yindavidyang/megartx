"""Independent stdlib CPU attention errors, not native arithmetic acceptance.

No observer, capture-plan, Torch, NumPy, BLAS, vLLM, or CUDA imports. BF16 dots
are exact integers; softmax uses fixed independent Decimal contexts. This
reference has no authority to validate native execution, sources or ownership.
"""
from decimal import (Decimal, Context, localcontext, ROUND_HALF_EVEN, Clamped,
                     InvalidOperation, DivisionByZero, Overflow, Underflow,
                     Subnormal, FloatOperation, Inexact, Rounded)
import struct
import time

POSITIONS = (0,15,16,255,256,1023,1024,1792,2047,2048)
SPECS = {0: (256,8,(0,),(0,1)), 5: (512,2,(0,1),(0,7,8,15))}


def context(precision):
    ctx = Context(prec=precision, rounding=ROUND_HALF_EVEN,
                  Emin=-999999, Emax=999999, capitals=1, clamp=0)
    for signal in ctx.traps:
        ctx.traps[signal] = signal not in (Inexact, Rounded)
    return ctx


def units(word):
    if type(word) is not int or not 0 <= word <= 65535 or word & 0x7f80 == 0x7f80:
        raise ValueError('Finite raw BF16 uint16 word required')
    e, m = (word >> 7)&255, word&127
    return (-1 if word&32768 else 1) * ((m+(128 if e else 0)) << max(0,e-1))


def dyadic(integer, power):
    with localcontext(context(600)):
        return Decimal(integer) * Decimal(2)**power


def check_deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise ValueError('CPU reference deadline exceeded; numerical result unresolved')


def exact_scores(query, keys, deadline=None):
    if not 1 <= len(query) <= 512 or not 1 <= len(keys) <= 2049:
        raise ValueError('Bounded full Q/K geometry required')
    q = [units(x) for x in query]
    out = []
    for row, key in enumerate(keys):
        if row%32 == 0:
            check_deadline(deadline)
        if len(key) != len(q):
            raise ValueError('K dimension is incomplete')
        out.append(sum(a*units(b) for a,b in zip(q,key)))
    return out


def ideal_sparse(query, keys, values, deadline=None):
    if not 1 <= len(values) == len(keys) <= 2049:
        raise ValueError('Complete nonempty aligned K/V rows required')
    width = len(values[0])
    if not 1 <= width <= 8 or any(len(row) != width for row in values):
        raise ValueError('Fixed sparse V columns required')
    scores = exact_scores(query, keys, deadline)
    maximum = max(scores)
    shifted = [dyadic(s-maximum,-266) for s in scores]
    if min(shifted) < -10000:
        raise ValueError('Reference exponent domain exceeded; no silent clipping')
    decoded = [[dyadic(units(v),-133) for v in row] for row in values]
    answers = []
    for precision in (96,128):
        with localcontext(context(precision)):
            check_deadline(deadline)
            weights = [x.exp() for x in shifted]
            denominator = sum(weights,Decimal(0))
            answers.append(tuple(sum((w*row[c] for w,row in zip(weights,decoded)),Decimal(0))/denominator
                                 for c in range(width)))
    with localcontext(context(600)):
        scale = max(abs(v) for row in decoded for v in row)
        tolerance = Decimal(2)**-180 * scale
        gap = max(abs(a-b) for a,b in zip(*answers))
        if gap > tolerance:
            raise ValueError('CPU reference precision disagreement; native result unresolved')
    check_deadline(deadline)
    return answers[1]


def metrics(observed_words, ideal):
    if len(observed_words) != len(ideal) or not ideal:
        raise ValueError('Exact nonempty output correspondence required')
    if any(type(x) is not Decimal or not x.is_finite() for x in ideal):
        raise ValueError('Finite high-precision ideal outputs required')
    observed = [dyadic(units(word),-133) for word in observed_words]
    with localcontext(context(600)):
        errors = [a-b for a,b in zip(observed,ideal)]
        square = sum((e*e for e in errors),Decimal(0))
        ideal_square = sum((x*x for x in ideal),Decimal(0))
        number=lambda x:'0' if not x else format(x,'.17E')
        result = {'coordinates':len(ideal), 'maximum_absolute_error':number(max(map(abs,errors))),
                  'rmse':number((square/len(errors)).sqrt()), 'l2_error':number(square.sqrt()),
                  'relative_l2_error':number((square/ideal_square).sqrt()) if ideal_square else None,
                  'minimum_signed_error':number(min(errors)), 'maximum_signed_error':number(max(errors)),
                  'ideal_minimum':number(min(ideal)), 'ideal_maximum':number(max(ideal)),
                  'metric_encoding':'decimal_scientific_strings_18_significant_digits',
                  'observed_negative_zero_count':sum(word==0x8000 for word in observed_words),
                  'reference_enclosure_certified':False,
                  'rounded_ideal_bf16_diagnostics':None,
                  'rounded_ideal_bf16_reason':'not_computed_without_certified_reference_enclosure'}
    return result


def analyze_arrays(arrays, max_seconds=300):
    """Analyze only the fixed eight arrays after external packet validation.

    Independent geometry/position/head checks are repeated here. Neither this
    function nor successful error calculation supplies native/storage proof.
    """
    if type(max_seconds) is not int or not 0 < max_seconds <= 300:
        raise ValueError('CPU reference wall allowance may only be reduced')
    deadline=time.monotonic()+max_seconds
    expected = {f'attention-layer-{layer:02d}-{role}.bf16' for layer in SPECS for role in 'kvqo'}
    if type(arrays) is not dict or set(arrays) != expected:
        raise ValueError('Exact eight raw arrays required')
    words = {}
    for layer,(dim,hkv,kh,qh) in SPECS.items():
        counts = {'k':2049*len(kh)*dim, 'v':2049*len(kh)*8,
                  'q':10*len(qh)*dim, 'o':10*len(qh)*8}
        for role,count in counts.items():
            check_deadline(deadline)
            name=f'attention-layer-{layer:02d}-{role}.bf16'; raw=arrays[name]
            if type(raw) is not bytes or len(raw)!=count*2:
                raise ValueError('Fixed full-dimension BF16 extent changed')
            # Decode one role at a time; at most 2,098,176 Python uint16s here.
            value=struct.unpack('<'+'H'*count,raw)
            if any(w&0x7f80==0x7f80 for w in value):
                raise ValueError('Nonfinite raw array')
            words[layer,role]=value
    cases=[]; all_ideal=[]; all_observed=[]
    for layer,(dim,hkv,kh,qh) in SPECS.items():
        for pi,p in enumerate(POSITIONS):
            begin=max(0,p-1023) if layer==0 else 0
            for hi,h in enumerate(qh):
                group=h//(16//hkv)
                if group not in kh:
                    raise ValueError('Missing selected GQA key head')
                gi=kh.index(group)
                query=words[layer,'q'][(pi*len(qh)+hi)*dim:(pi*len(qh)+hi+1)*dim]
                keys=[words[layer,'k'][(j*len(kh)+gi)*dim:(j*len(kh)+gi+1)*dim] for j in range(begin,p+1)]
                values=[words[layer,'v'][(j*len(kh)+gi)*8:(j*len(kh)+gi+1)*8] for j in range(begin,p+1)]
                ideal=ideal_sparse(query,keys,values,deadline)
                observed=words[layer,'o'][(pi*len(qh)+hi)*8:(pi*len(qh)+hi+1)*8]
                cases.append({'layer':layer,'position':p,'q_head':h,'kv_head':group,
                              'valid_keys':p-begin+1,'ideal_decimal':[str(x) for x in ideal],
                              **metrics(observed,ideal)})
                all_ideal.extend(ideal);all_observed.extend(observed)
    check_deadline(deadline)
    return {'schema':'megartx-prefill-attention-errors-v1','status':'independent_attention_errors_computed',
            'cases':cases,'aggregate':metrics(all_observed,all_ideal),
            'reference_precision_digits':[96,128], 'reference_context':'explicit_half_even_emin_-999999_emax_999999',
            'external_native_storage_frontier_validation_required':True,
            'native_execution_attested':False, 'native_arithmetic_acceptance':None,
            'numerical_qualified':False,'quality_qualified':False,'performance_qualified':False,
            'sampled_repeatability_qualified':False,'natural_positive_correction_coverage':None}

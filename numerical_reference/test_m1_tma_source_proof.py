"""Independent CPU proof of the audited pinned producer's bounded equations.

This models source semantics; it does not execute or attest the native producer.
The exact CuTE layout proof remains in test_m1_sf_layout_contract.py.
"""
import unittest

from m1_preparation_reference import BufferRef, grouped_sf_base, sf_coordinate


class TmaSourceProofTests(unittest.TestCase):
    def test_every_feasible_active_expert_rank_preserves_payload_owner_bounds(self):
        cases=0
        for n,k in ((1408,2816),(2816,704)):
            for expert in range(128):
                for rank in range(8):
                    if rank>expert or 7-rank>127-expert:
                        continue
                    # Construct a complete distinct route giving this expert
                    # precisely this prefix, then derive offsets independently.
                    ids=list(range(rank))+[expert]+list(range(expert+1,expert+8-rank))
                    self.assertEqual(len(set(ids)),8)
                    offsets=[sum(e<i for e in ids) for i in range(129)]
                    self.assertEqual((offsets[expert],offsets[expert+1]),(rank,rank+1))
                    # Audited source: safe_inc_ptr adjusts FP4 element offsets
                    # by two; output is BF16. Shape is (token_count,n,k).
                    shape=(offsets[expert+1]-offsets[expert],n,k)
                    aq=rank*k//2;output=rank*n*2
                    sf=((rank+127*expert+127)//128)*128*(k//16)
                    self.assertEqual(shape,(1,n,k))
                    self.assertEqual(sf,grouped_sf_base(expert,rank,k))
                    for owner,start,extent,capacity in (
                            ("AQ",aq,k//2,45056),("output",output,n*2,45056),
                            ("SF",sf,128*k//16,2883584),
                            ("weight",expert*n*k//2,n*k//2,128*n*k//2),
                            ("weight_SF",expert*n*k//16,n*k//16,128*n*k//16)):
                        BufferRef(owner,start,extent,capacity)
                        with self.assertRaises(ValueError):
                            BufferRef(owner,start,extent,start+extent-1)
                    # Row-zero writes are contained within the physical carrier;
                    # padding bytes are preserved, not assumed zero.
                    self.assertTrue(all(sf_coordinate(0,b,k//16)<128*k//16 for b in range(k//16)))
                    cases+=1
        self.assertEqual(cases,1936)

    def test_all_inactive_prefixes_have_zero_rows_without_payload_requirements(self):
        cases=0
        sentinel=object()
        for expert in range(128):
            for prefix in range(9):
                if prefix>expert or 8-prefix>127-expert:
                    continue
                ids=list(range(prefix))+list(range(expert+1,expert+9-prefix))
                self.assertEqual(len(set(ids)),8)
                self.assertNotIn(expert,ids)
                before=sum(e<expert for e in ids);after=sum(e<expert+1 for e in ids)
                self.assertEqual((before,after),(prefix,prefix))
                for n,k in ((1408,2816),(2816,704)):
                    # Pinned kernel writes every shape then returns for M=0.
                    entry={"shape":(after-before,n,k),"payload":sentinel}
                    self.assertEqual(entry["shape"][0],0)
                    self.assertIs(entry["payload"],sentinel)
                cases+=1
        self.assertEqual(cases,1080)

    def test_changed_routes_recompute_prefixes_and_never_reuse_entries(self):
        routes=((18,11,16,12,15,13,17,14),(72,65,70,66,69,67,71,68),(127,0,126,1,125,2,124,3))
        previous=None
        for ids in routes:
            offsets=[sum(e<i for e in ids) for i in range(129)]
            active={e:(offsets[e],offsets[e+1]-offsets[e]) for e in range(128) if offsets[e+1]!=offsets[e]}
            self.assertEqual(set(active),set(ids))
            self.assertEqual(sorted(rank for rank,count in active.values()),list(range(8)))
            self.assertTrue(all(count==1 for rank,count in active.values()))
            if previous is not None:self.assertNotEqual(active,previous)
            previous=active


if __name__ == "__main__":unittest.main()

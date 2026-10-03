import copy
import unittest

from megartx.m1_normal_plan import ORIGIN, CASES, REVISION, digest, validate_plan, frames, request


def plan():
    value = {**ORIGIN, "schema": "megartx-m1-normal-plan-v1", "checkpoint_revision": REVISION,
        "outputs": 4, "prefill_chunk": 256, "continuation_token_id": 7, "eos_token_ids": [1,2],
        "expected_live_calls": 210, "expected_fallback_calls": 150,"expected_model_forwards": 12,
        "cases": [{"id": name,"prompt_token_ids": [3]*length,"prompt_sha256": digest([3]*length)} for name,length in CASES]}
    value["plan_sha256"] = digest(value)
    return value


class TestNormalPlan(unittest.TestCase):
    def test_exact_plan_classifies_prefill_tail_and_cached_boundary_inputs(self):
        value = validate_plan(plan())
        actual = [frames(c) for c in value["cases"]]
        self.assertEqual([len(f) for f in actual[0]],[256,1,1,1,1])
        self.assertEqual([len(f) for f in actual[1]],[256,256,256,255,1,1,1])
        self.assertEqual(actual[0][1:],[ [256],[257],[258],[259] ])
        self.assertEqual(actual[1][-3:],[ [1023],[1024],[1025] ])
        self.assertEqual(sum(len(f)==1 for c in actual for f in c)*30,210)
        self.assertEqual(sum(len(f)>1 for c in actual for f in c)*30,150)
        for c in value["cases"]:
            marker = request(value,c)
            self.assertTrue(marker["routing_unchanged"])
            self.assertFalse(marker["routing_intervention"])
            self.assertEqual(marker["prompt_token_ids"][-3:],[7,7,7])

    def test_rehashed_boundary_expansions_are_rejected(self):
        for field,new in (("outputs",5),("prefill_chunk",128),("expected_live_calls",240),
                          ("expected_fallback_calls",120),("expected_model_forwards",13),
                          ("continuation_token_id",1),("checkpoint_revision","changed"),
                          ("routing_intervention",True),("routing_unchanged",False),("route_origin","controlled")):
            with self.subTest(field=field):
                value = plan();value[field] = new
                value["plan_sha256"] = digest({k:v for k,v in value.items() if k!="plan_sha256"})
                with self.assertRaises(RuntimeError): validate_plan(value)

    def test_extra_duplicate_changed_prompt_and_token_rejected(self):
        for change in ("extra","duplicate","length","token","hash"):
            value = plan()
            if change == "extra": value["cases"].append(copy.deepcopy(value["cases"][0]))
            elif change == "duplicate": value["cases"][1]["id"] = "split257"
            elif change == "length": value["cases"][0]["prompt_token_ids"].append(3)
            elif change == "token": value["cases"][0]["prompt_token_ids"][0] = True
            else: value["cases"][0]["prompt_sha256"] = "wrong"
            value["plan_sha256"] = digest({k:v for k,v in value.items() if k!="plan_sha256"})
            with self.assertRaises(RuntimeError): validate_plan(value)

    def test_changed_plan_digest_rejected(self):
        value = plan();value["plan_sha256"] = "0"*64
        with self.assertRaisesRegex(RuntimeError,"digest"): validate_plan(value)

"""Synthetic corruption checks for the independent bounded capture reader."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import layer0_reference as ref
import compare_controlled_capture as evidence


class EvidenceGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "controlled/cached"
        self.directory.mkdir(parents=True)
        (self.root / "run.exit").write_text("0\n")
        (self.root / "status.json").write_text('{"phase":"cleanup_complete"}')
        self.plan = {"tokens": np.arange(33, dtype=np.int64), "token_sha256": "test_token", "schedule_sha256": "test_schedule"}
        constants = {"k_weight_bits": np.zeros((2048, 2816), dtype=np.uint16),
                     "input_norm_weight_bits": np.zeros(2816, dtype=np.uint16),
                     "q_norm_weight_bits": np.zeros(256, dtype=np.uint16), "k_norm_weight_bits": np.zeros(256, dtype=np.uint16),
                     "post_attention_norm_weight_bits": np.zeros(2816, dtype=np.uint16), "pre_ff2_norm_weight_bits": np.zeros(2816, dtype=np.uint16)}
        self.binding = {**evidence.reference.CONTROLLED_ORIGIN, **{k:self.plan[k] for k in ("token_sha256", "schedule_sha256")},
                        "mode":"native", "path":"cached", "schema":2, "sources":ref.PINNED_SOURCES, "batch_invariant":False,
                        "loaded_k_sha256":hashlib.sha256(constants["k_weight_bits"].tobytes()).hexdigest(), "frames":[],
                        "constants_file":"constants.npz", "constants_sha256":self.save("constants.npz", constants)}
        self.frames = []
        for ordinal, (count, positions, prefix) in enumerate(((32,[31],32),(1,[32],33))):
            arrays = {name:np.zeros((len(positions), width), dtype=np.uint16) for name,width in {**ref.WIDTHS,**ref.EXTRA_WIDTHS}.items()}
            arrays.update(positions=np.array(positions,dtype=np.int64), tokens=self.plan["tokens"][positions],
                          rope_cache_bits=np.zeros((1,256),dtype=np.uint16),
                          cache_positions=np.arange(prefix,dtype=np.int64),cache_slots=np.arange(prefix,dtype=np.int64),
                          writer_all_k=np.zeros((count,2048),dtype=np.uint16),writer_all_v=np.zeros((count,2048),dtype=np.uint16),
                          stored_prefix_k=np.zeros((prefix,2048),dtype=np.uint16),stored_prefix_v=np.zeros((prefix,2048),dtype=np.uint16))
            frame = {"forward":ordinal,"source_rows":count,"positions":positions,"file":f"frame-{ordinal}.npz","writer_slots":positions,"cache_shape":[3,8,16,512]}
            self.frames.append(arrays);self.binding["frames"].append(frame)
            self.rewrite_frame(ordinal)
        kv={"logical_positions":np.array([31,32],dtype=np.int64),"key_bits":np.zeros((2,8,256),dtype=np.uint16),"value_bits":np.zeros((2,8,256),dtype=np.uint16)}
        (self.directory/"kv-records.json").write_text(json.dumps([{"layer":0,"file":"kv.npz","sha256":self.save("kv.npz",kv)}]))
        self.stages=[];records=[]
        for position,expert in ((31,42),(32,82)):
            stage={field:np.zeros((1,),dtype=np.uint16) for field in evidence.STAGE_FIELDS}
            stage.update(input_bits=np.zeros((1,2816),dtype=np.uint16),positions=np.array([position],dtype=np.int64),token_ids=self.plan["tokens"][[position]])
            self.stages.append(stage);name=f"stage-{expert}.npz"
            records.append({"layer":0,"file":name,"stage_capture_sha256":self.save(name,stage)})
        self.controlled={"executed_interventions":records}
        self.write_metadata()

    def save(self, name, arrays):
        path=self.directory/name
        with path.open("wb") as stream:np.savez_compressed(stream,**arrays)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def rewrite_frame(self, ordinal):
        frame=self.binding["frames"][ordinal];arrays=self.frames[ordinal]
        frame["sha256"]=self.save(frame["file"],arrays)
        frame["prefix_payload_sha256"]={field:hashlib.sha256(arrays["stored_prefix_"+field].tobytes()).hexdigest() for field in ("k","v")}

    def write_metadata(self):
        (self.directory/"layer0-boundaries.json").write_text(json.dumps(self.binding))
        (self.directory/"controlled-manifest.json").write_text(json.dumps(self.controlled))

    def test_consistent_synthetic_prefix_and_residual_links(self):
        case=ref.load_case(self.root,"cached",self.plan)
        self.assertEqual(len(case["prefixes"][1]["cache_slots"]),33)
        self.assertEqual(set(case["rows"]),{31,32})

    def test_rehashed_prefix_corruption_is_rejected(self):
        self.frames[1]["stored_prefix_k"][4,19]=0x3f80
        self.rewrite_frame(1);self.write_metadata()
        with self.assertRaisesRegex(ValueError,"decode changed"):
            ref.load_case(self.root,"cached",self.plan)

    def test_rehashed_slot_alias_is_rejected(self):
        self.frames[1]["cache_slots"][4]=3
        self.rewrite_frame(1);self.write_metadata()
        with self.assertRaisesRegex(ValueError,"address association"):
            ref.load_case(self.root,"cached",self.plan)

    def test_rehashed_residual_corruption_is_rejected(self):
        self.frames[1]["pre_ff2_norm_in"][0,20]=0x3f80
        self.rewrite_frame(1);self.write_metadata()
        with self.assertRaisesRegex(ValueError,"residual addition"):
            ref.load_case(self.root,"cached",self.plan)

    def test_rehashed_stale_expert_input_is_rejected(self):
        self.stages[1]["input_bits"][0,20]=0x3f80
        record=self.controlled["executed_interventions"][1]
        record["stage_capture_sha256"]=self.save(record["file"],self.stages[1]);self.write_metadata()
        with self.assertRaisesRegex(ValueError,"incoming corrected expert row"):
            ref.load_case(self.root,"cached",self.plan)

    def test_unbounded_embedding_request_rejected_before_read(self):
        for tokens in (np.arange(34,dtype=np.int64),np.arange(33,dtype=np.int32),np.full(33,262144,dtype=np.int64),np.full(33,-1,dtype=np.int64)):
            with self.subTest(shape=tokens.shape),self.assertRaisesRegex(ValueError,"canonical 33"):
                ref.original_layer0_inputs(self.root,tokens)


if __name__ == "__main__":
    unittest.main()

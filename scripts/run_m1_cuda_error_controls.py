"""Build and run a fixed synthetic CUDA-error matrix in disposable children."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_m1_installed_probe import available, gpu, installed_flags, rss, sha

# Do not pass destroyed CUDA handles to the runtime. The retained exploratory
# destroyed-stream child crashed inside the driver's query; it is not a checked
# API-error control. These cases use live owned objects or a cleared context.
CONTROLS = ("supported", "illegal_address", "driver_no_context")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    pinned=json.loads((args.build/"build.json").read_text())
    if (pinned.get("returncode")!=0 or pinned.get("reason") or any(sha(Path(p))!=h for p,h in pinned["installed_pins"].items())):
        raise RuntimeError("failure controls require unchanged successful installed source pins")
    before=gpu()
    if before["processes"] or before["free_bytes"]<2<<30 or available()<8<<30:
        raise RuntimeError("failure controls require complete model cleanup and retained headroom")
    work=args.output.resolve();work.mkdir(mode=0o700)
    (work/"temporary").mkdir()
    sources=("probes/m1_cuda_error_controls.cu","probes/m1_installed_bridge.cuh",
             "kernels/m1_installed_preparation.cuh","kernels/m1_maps_expand.cuh",
             "scripts/run_m1_cuda_error_controls.py")
    for name in sources:
        target=work/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
    ninja=next(Path(p)for p in pinned["installed_pins"] if Path(p).name=="build.ninja")
    runtime=next(Path(p)for p in pinned["installed_pins"] if Path(p).name=="libcudart.so.13")
    module=next(Path(p)for p in pinned["installed_pins"] if Path(p).name=="fused_moe_120.so")
    ffi=next(Path(p)for p in pinned["installed_pins"] if Path(p).name=="libtvm_ffi.so")
    compiler,flags=installed_flags(ninja)
    command=[compiler,*flags,"--cudart=shared","-I"+str(work/"probes"),str(work/sources[0]),str(module),str(ffi),
             "-Xlinker","-rpath","-Xlinker",str(module.parent),
             "-Xlinker","-rpath","-Xlinker",str(ffi.parent),
             "-Xlinker","-rpath","-Xlinker",str(runtime.parent),"-lcuda","-ldl","-o",str(work/"cuda-error-controls")]
    report={"scope":"disposable_synthetic_preparation_and_driver_error_controls_without_model",
            "before":before,"source_hashes":{p:sha(work/p)for p in sources},
            "installed_pins":pinned["installed_pins"],"command":command,"controls":[],
            "limits":{"compiler_rss_bytes":2<<30,"compile_seconds":300,"host_available_bytes":8<<30,
                      "gpu_free_bytes":2<<30,"device_scratch_bytes":8<<20}}
    peak=0;start=time.monotonic();reason=None
    with(work/"compile.stdout").open("xb")as out,(work/"compile.stderr").open("xb")as err:
        proc=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=os.environ|{"TMPDIR":str(work/"temporary")})
        try:
            while proc.poll()is None:
                peak=max(peak,rss(proc.pid))
                if peak>2<<30 or available()<8<<30 or time.monotonic()-start>300:
                    reason="compiler resource bound";break
                time.sleep(.05)
        finally:
            if proc.poll()is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    report.update(compile_returncode=proc.returncode,compile_reason=reason,
                  compile_seconds=time.monotonic()-start,peak_aggregate_rss_bytes=peak)
    try:
        if proc.returncode or reason:
            raise RuntimeError("CUDA error control build failed: "+(work/"compile.stderr").read_text()[-4000:])
        report["binary_sha256"]=sha(work/"cuda-error-controls")
        for control in CONTROLS:
            state=gpu()
            if state["processes"] or state["free_bytes"]<2<<30 or available()<8<<30:
                raise RuntimeError("GPU ownership/headroom changed before disposable control")
            with(work/(control+".stdout")).open("xb")as out,(work/(control+".stderr")).open("xb")as err:
                child=subprocess.Popen([str(work/"cuda-error-controls"),control],stdout=out,stderr=err,start_new_session=True)
                try:code=child.wait(timeout=30)
                finally:
                    if child.poll()is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            entry={"control":control,"pid":child.pid,"returncode":code,"after":gpu()}
            if code!=0:
                report["controls"].append(entry)
                raise RuntimeError("disposable CUDA control failed: "+control)
            value=json.loads((work/(control+".stdout")).read_text())
            if (value.get("control")!=control or value.get("incumbent_calls")!=0
                    or value.get("propagated_errors")!=(0 if control=="supported" else 1)
                    or value.get("scratch_bytes",8<<20)>8<<20
                    or value.get("scratch_bytes")!=(0 if control=="driver_no_context" else 2919976)
                    or value.get("preparation_calls")!=(0 if control=="driver_no_context" else 1)
                    or value.get("trigger_status")!=(700 if control=="illegal_address" else 201 if control=="driver_no_context" else 0)
                    or value.get("device_buffers_reused_after_error")is not False
                    or entry["after"]["processes"]):
                raise RuntimeError("disposable CUDA error/no-retry/cleanup evidence differs")
            entry["result"]=value;report["controls"].append(entry)
        report["passed"]=True
    finally:
        report["installed_pins_unchanged"]=all(sha(Path(p))==h for p,h in pinned["installed_pins"].items())
        report["after"]=gpu()
        (work/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items()if k not in {"installed_pins","command"}},indent=2))


if __name__=="__main__":
    main()

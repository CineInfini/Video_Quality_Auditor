import json
from pathlib import Path
from cineinfini.io.renderers.benchmark_renderer import BenchmarkRenderer
def _audit(name, comps, verdicts=None):
    gates = {}
    verdicts = verdicts or ["ACCEPT"]*len(comps)
    for i,(c,v) in enumerate(zip(comps,verdicts),1): gates[i]={"composite":c,"verdict":v,"motion_peak_div":12.0}
    return {"video_name":name,"version":"0.4.8.1","video_path":f"/tmp/{name}.mp4","n_shots":len(comps),"gates":gates,"duration_s":10.0}
def test_renderer_writes_all_outputs(tmp_path):
    BenchmarkRenderer().render_many([_audit("a",[0.8,0.7]),_audit("b",[0.4,0.5],["REVIEW"]*2)], tmp_path)
    out = tmp_path / "benchmark"
    assert all((out/f).exists() for f in ["benchmark.md","benchmark.html","benchmark.csv","benchmark.json"])
def test_ranking_sorts_descending(tmp_path):
    BenchmarkRenderer().render_many([_audit("low",[0.3]),_audit("high",[0.9]),_audit("mid",[0.6])], tmp_path)
    data = json.loads((tmp_path/"benchmark"/"benchmark.json").read_text())
    assert [r["video_name"] for r in data["ranking"]] == ["high","mid","low"]
def test_empty_audits_writes_placeholder(tmp_path):
    out = BenchmarkRenderer().render_many([], tmp_path)
    assert "No audits to compare" in (out/"benchmark.md").read_text()

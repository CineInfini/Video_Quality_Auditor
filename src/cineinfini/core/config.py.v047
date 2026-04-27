# src/cineinfini/core/config.py
from __future__ import annotations
import os, sys, tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

_D_PATHS = {"models_dir":"~/.cineinfini/models","reports_dir":"~/.cineinfini/reports","benchmark_dir":"~/.cineinfini/benchmark","test_videos_dir":"~/.cineinfini/test_videos","cache_dir":"~/.cineinfini/cache","logs_dir":"~/.cineinfini/logs","temp_dir":"/tmp/cineinfini","output_root":"~/.cineinfini/output"}
_D_DEVICE = {"gpu_device":"auto","torch_dtype":"float16","use_amp":True}
_D_PROCESSING = {"max_duration_s":60,"shot_threshold":0.2,"min_shot_duration_s":0.5,"downsample_to":[320,180],"n_frames_per_shot":16,"frame_resize":[320,180],"step":2,"adaptive_threshold":True,"threshold_percentile":85,"num_workers":4,"parallel_shots":True,"inter_shot_subsample":5,"narrative_coherence":True,"compute_dtw_self":True,"compute_dtw_inter":True,"dtw_max_samples":16,"benchmark_mode":True,"embedder":"arcface_onnx","semantic_scorer":"clip"}
_D_THRESHOLDS = {"motion":25.0,"ssim3d":0.45,"flicker":0.10,"identity_drift":0.60,"ssim_long_range":0.45,"clip_temp":0.25,"flicker_hf":0.01,"narrative_coherence":0.70,"temporal_coherence":0.75,"physics_overall":0.65,"aesthetic_score":0.60,"causal_violation":0.35,"trust_score":0.70,"background_ssim":0.55}
_D_MODEL_URLS = {"arcface":{"url":"https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx","filename":"arcface.onnx","sha256":None},"yunet":{"url":"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx","filename":"yunet.onnx","sha256":None},"clip_vit_b32":{"url":"https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt","filename":"ViT-B-32.pt","sha256":"40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"},"dinov2_vitb14":{"url":"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth","filename":"dinov2_vitb14.pth","sha256":None}}
_D_MODULES = {
    "motion_coherence":{"enabled":True,"threshold":25.0},
    "identity_consistency":{"enabled":True,"model":"arcface","threshold":0.60,"n_samples":5,"use_dtw":True},
    "semantic_consistency":{"enabled":True,"model":"clip","threshold":0.25},
    "background_consistency":{"enabled":False,"method":"ssim","threshold":0.55,"subsample":5},
    "origin_detection":{"enabled":False,"confidence_threshold":0.65},
    "temporal_signature":{"enabled":False,"max_frames":30,"optical_flow_params":{"pyr_scale":0.5,"levels":3,"winsize":15,"iterations":3}},
    "physics_plausibility":{"enabled":False,"min_contour_area":300,"iou_threshold":0.3},
    "trustworthiness":{"enabled":False,"noise_level":0.05,"n_samples":5},
    "explainability":{"enabled":False,"method":"proportional","n_permutations":100},
    "benchmark_fusion":{"enabled":False,"vbench_path":None},
    "benchmark_forensic":{"enabled":False,"compression_crfs":[23,28,35]},
    "causal_reasoning":{"enabled":False,"gravity_penalty":8.0,"vertical_flow_threshold":0.1},
    "long_term_narrative":{"enabled":False,"segment_size":16,"similarity_metric":"cosine"},
    "aesthetic_cinematic":{"enabled":False,"use_rule_of_thirds":True,"color_harmony_weight":0.4,"contrast_weight":0.3,"composition_weight":0.3},
    "prompt_alignment_fine":{"enabled":False,"model":"blip2","vlm_device":"cpu"},
    "world_model_surprise":{"enabled":False,"use_divergence":True,"surprise_threshold":0.7},
    "subject_consistency_long":{"enabled":False,"window_size":30,"dtw_enabled":True},
    "multi_modal_safety":{"enabled":False,"nsfw_threshold":0.8,"violence_threshold":0.7},
    "creative_composition":{"enabled":False,"rhythm_variance_weight":0.5},
}
_D_REPORTING = {"active_renderers":["markdown","json"],"figure_format":"png","figure_dpi":150,"theme":"dark","interactive":True,"generate_markdown":True,"generate_html":False,"generate_plots":True,"save_raw_data":True,"include_shapley":False,"dashboard_theme":"dark"}
_D_LOGGING = {"level":"INFO","file_enabled":True,"console_enabled":True,"format":"%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"}

@dataclass
class Config:
    paths: Dict[str,str]=field(default_factory=lambda:dict(_D_PATHS))
    device: Dict[str,Any]=field(default_factory=lambda:dict(_D_DEVICE))
    processing: Dict[str,Any]=field(default_factory=lambda:dict(_D_PROCESSING))
    thresholds: Dict[str,float]=field(default_factory=lambda:dict(_D_THRESHOLDS))
    model_urls: Dict[str,Dict[str,Any]]=field(default_factory=lambda:{k:dict(v) for k,v in _D_MODEL_URLS.items()})
    modules: Dict[str,Dict[str,Any]]=field(default_factory=lambda:{k:dict(v) for k,v in _D_MODULES.items()})
    reporting: Dict[str,Any]=field(default_factory=lambda:dict(_D_REPORTING))
    logging: Dict[str,Any]=field(default_factory=lambda:dict(_D_LOGGING))
    def resolve_path(self,key:str)->Path: return Path(self.paths.get(key,f"~/.cineinfini/{key}")).expanduser().resolve()
    def models_dir(self)->Path: return self.resolve_path("models_dir")
    def reports_dir(self)->Path: return self.resolve_path("reports_dir")
    def benchmark_dir(self)->Path: return self.resolve_path("benchmark_dir")
    def model_path(self,key:str)->Optional[Path]: e=self.model_urls.get(key); return self.models_dir()/e["filename"] if e else None
    def is_module_enabled(self,name:str)->bool: return bool(self.modules.get(name,{}).get("enabled",False))
    def is_enabled(self,name:str)->bool: return self.is_module_enabled(name)
    def get_module_config(self,name:str)->Dict[str,Any]: return self.modules.get(name,{})
    def enabled_modules(self)->List[str]: return [n for n,c in self.modules.items() if c.get("enabled",False)]
    def active_renderers(self)->List[str]:
        explicit=self.reporting.get("active_renderers")
        if explicit: return list(explicit)
        out=[]
        if self.reporting.get("generate_markdown",True): out.append("markdown")
        if self.reporting.get("generate_html",False): out.append("html")
        if self.reporting.get("save_raw_data",True): out.append("json")
        return out
    def figure_format(self)->str: return str(self.reporting.get("figure_format","png")).lower()
    def figure_dpi(self)->int: return int(self.reporting.get("figure_dpi",150))
    def theme(self)->str: return str(self.reporting.get("theme",self.reporting.get("dashboard_theme","dark")))
    def effective_device(self)->str:
        dev=str(self.device.get("gpu_device","auto")).lower()
        if dev=="auto":
            try: import torch; return "cuda" if torch.cuda.is_available() else "cpu"
            except: return "cpu"
        return dev
    @staticmethod
    def is_jupyter()->bool:
        try:
            from IPython import get_ipython
            ip=get_ipython()
            if ip is None: return False
            shell=type(ip).__name__
            return shell in {"ZMQInteractiveShell","Shell"}
        except: return "ipykernel" in sys.modules
    def to_audit_config(self)->dict:
        cfg=dict(self.processing)
        cfg["thresholds"]=dict(self.thresholds)
        cfg["gpu_device"]=self.effective_device()
        cfg["enable_animal_face_detection"]=False
        return cfg
    @classmethod
    def from_dict(cls,data:Dict[str,Any])->"Config":
        dev_data=data.get("device",{})
        if isinstance(dev_data,str): device={**_D_DEVICE,"gpu_device":dev_data}
        else: device={**_D_DEVICE,**(dev_data or {})}
        merged={k:dict(v) for k,v in _D_MODULES.items()}
        for name,entry in (data.get("modules") or {}).items():
            base=merged.get(name,{})
            merged[name]={**base,**(entry or {})}
        return cls(
            paths={**_D_PATHS,**(data.get("paths") or {})},
            device=device,
            processing={**_D_PROCESSING,**(data.get("processing") or {})},
            thresholds={**_D_THRESHOLDS,**(data.get("thresholds") or {})},
            model_urls={**{k:dict(v) for k,v in _D_MODEL_URLS.items()},**(data.get("model_urls") or {})},
            modules=merged,
            reporting={**_D_REPORTING,**(data.get("reporting") or {})},
            logging={**_D_LOGGING,**(data.get("logging") or {})},
        )
    def replace(self,**overrides)->"Config": import dataclasses; return dataclasses.replace(self,**overrides)
    def to_dict(self)->Dict[str,Any]: return asdict(self)
_config=None
def get_config()->Config: global _config; return _default_config() if _config is None else _config
def set_config(cfg:Config)->None: global _config; _config=cfg
def reset_config()->None: global _config; _config=None
def default_config()->Config: return Config()
def test_config()->Config:
    tmp=Path(tempfile.mkdtemp(prefix="cineinfini_test_"))
    cfg=Config()
    cfg.paths.update({"reports_dir":str(tmp/"reports"),"benchmark_dir":str(tmp/"benchmark"),"test_videos_dir":str(tmp/"videos"),"cache_dir":str(tmp/"cache"),"logs_dir":str(tmp/"logs"),"temp_dir":str(tmp/"temp"),"output_root":str(tmp/"output")})
    cfg.processing.update({"max_duration_s":10,"n_frames_per_shot":8,"num_workers":2,"parallel_shots":False,"narrative_coherence":False,"benchmark_mode":False})
    for n in cfg.modules:
        if n not in {"motion_coherence","identity_consistency","semantic_consistency"}:
            cfg.modules[n]["enabled"]=False
    cfg.reporting.update({"active_renderers":["json"],"figure_format":"png","figure_dpi":72,"interactive":False,"generate_plots":False})
    cfg.logging.update({"level":"WARNING","file_enabled":False})
    cfg.device["gpu_device"]="cpu"
    cfg.device["use_amp"]=False
    return cfg
def load_config(path:str|Path)->Config: p=Path(path).expanduser(); data=yaml.safe_load(p.read_text(encoding="utf-8")) or {}; return Config.from_dict(data)
def save_config(cfg:Config,path:str|Path)->None: p=Path(path).expanduser(); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(yaml.dump(cfg.to_dict(),default_flow_style=False,sort_keys=False),encoding="utf-8")
def _default_config()->Config:
    cand=[Path(os.environ.get("CINEINFINI_CONFIG","")).expanduser() if os.environ.get("CINEINFINI_CONFIG") else None, Path.home()/".cineinfini"/"config.yaml", Path.cwd()/"cfg"/"config.yaml"]
    for c in cand:
        if c and c.exists():
            try: return load_config(c)
            except Exception as e: print(f"[cineinfini.config] Warning: could not load {c}: {e}",file=sys.stderr)
    return default_config()
def compat_models_dir()->Path: return get_config().models_dir()
def compat_reports_dir()->Path: return get_config().reports_dir()

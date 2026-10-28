"""Config-driven per-slice pipeline (paper Fig. 1). Every stage is a registry name:

    segmenter -> postprocess steps -> aligner -> fitter -> analysis

    pipe = Pipeline.from_config(load_config("configs/pipeline.yaml"))
    res = pipe(images, tis, pixel_spacing=..., masks=...)

Swap any stage in the YAML (or with --set aligner.name=none) without touching code.
`Pipeline.segment_only(...)` stops after post-processing.
"""
import numpy as np

from .analysis import T1_RANGE_1P5T, midwall_mask, slice_stats
from .io import pad_crop
from .postprocess import build_steps
from .registry import ALIGNERS, FITTERS, SEGMENTERS

DEFAULTS = dict(
    size=256,
    segmenter={"name": "ground_truth"},
    postprocess=None,                      # None = paper chain
    aligner={"name": "polar"},
    fitter={"name": "2p"},
    min_good_images=8,                     # paper: >= 8 of 11 images segmented OK
    analysis=dict(midwall_fraction=1 / 3, t1_range=list(T1_RANGE_1P5T)),
)


class Pipeline:
    def __init__(self, segmenter, steps, aligner, fitter, size=256, min_good_images=8, analysis=None):
        self.segmenter, self.steps, self.aligner, self.fitter = segmenter, steps, aligner, fitter
        self.size, self.min_good = size, min_good_images
        self.analysis = {**DEFAULTS["analysis"], **(analysis or {})}

    @classmethod
    def from_config(cls, cfg, **override):
        """cfg: dict with any of the DEFAULTS keys (top level or under 'pipeline:')."""
        c = {**DEFAULTS, **cfg.get("pipeline", cfg), **override}
        seg = c["segmenter"]
        if isinstance(seg, dict) and seg.get("name") == "torch" and "device" in cfg:
            seg = {"device": cfg["device"], **seg}
        return cls(SEGMENTERS.build(seg), build_steps(c["postprocess"]), ALIGNERS.build(c["aligner"]),
                   FITTERS.build(c["fitter"]), c["size"], c["min_good_images"], c["analysis"])

    def segment_only(self, images, tis, pixel_spacing=(1.4, 1.4), **ctx):
        images = pad_crop(np.asarray(images, np.float32), self.size)
        tis = np.asarray(tis, float)
        raw = self.segmenter(images, tis, **ctx)
        masks, ok = raw.astype(np.uint8), np.ones(len(raw), bool)
        for f in self.steps:
            masks, ok = f(masks, ok, tis, pixel_spacing)
        return dict(images=images, tis=tis, raw_masks=raw, masks=masks, ok=ok)

    def __call__(self, images, tis, pixel_spacing=(1.4, 1.4), **ctx):
        out = self.segment_only(images, tis, pixel_spacing, **ctx)
        ok = out["ok"]
        if ok.sum() < self.min_good:
            out["status"] = f"skipped: only {ok.sum()}/{len(ok)} usable masks"
            return out
        al = self.aligner.align(out["images"], out["masks"], ok, out["tis"])
        t1_common = self.fitter(al.stack, al.tis)
        t1_map = al.to_image(t1_common)
        mid = midwall_mask(al.ref_mask, self.analysis["midwall_fraction"])
        out.update(status="ok", aligned=al.stack, t1_common=t1_common, t1_map=t1_map,
                   ref_index=al.ref_index, midwall=mid,
                   stats=slice_stats(t1_map, mid, tuple(self.analysis["t1_range"])))
        return out


def run_slice(images, tis, masks=None, checkpoint=None, pixel_spacing=(1.4, 1.4), **cfg):
    """Convenience wrapper: GT masks if `masks` given, else a torch checkpoint."""
    seg = {"name": "ground_truth"} if checkpoint is None else {"name": "torch", "checkpoint": checkpoint}
    return Pipeline.from_config({"segmenter": seg, **cfg})(images, tis, pixel_spacing, masks=masks)

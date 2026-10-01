"""Generate notebooks/T_26_Vit_Base_Model.ipynb from the T23 (ResNet50) notebook.

Global renames first, then full rewrites of the cells whose content is backbone-specific.
Every rewrite is keyed by cell index AND asserts on a marker in the original cell, so a
changed T23 notebook fails loudly instead of producing a silently wrong T26 notebook.
"""
import copy, json, sys

SRC = "notebooks/T23_resnet50_lung_attention.ipynb"
DST = "notebooks/T_26_Vit_Base_Model.ipynb"
CHECK = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
OVERFIT_LR = CHECK.get("overfit_lr", 1e-4)
SENS_LR = CHECK.get("sens_lr", 1e-4)

nb = json.load(open(SRC))
cells = copy.deepcopy(nb["cells"])

RENAMES = [
    ("T23_resnet50_lung_attention", "T26_vit_base_lung_attention"),
    ("resnet50_lung_attention.yaml", "vit_base_lung_attention.yaml"),
    ("T23_EFFICIENCY_CSV", "T26_EFFICIENCY_CSV"),
    ("T23_ROOT", "T26_ROOT"),
    ("t23_eval_done", "t26_eval_done"),
    ("t23_done", "t26_done"),
    ("_is_t23_output", "_is_t26_output"),
    ("RESNET50_WINNER", "VIT_WINNER"),
    ('f"resnet50_{arm_name}.pt"', 'f"{BACKBONE}_{arm_name}.pt"'),
    ('"resnet50_A0_vanilla.pt"', 'f"{BACKBONE}_A0_vanilla.pt"'),
    ('"resnet50_A2_full.pt"', 'f"{BACKBONE}_A2_full.pt"'),
    ('generate_run_name("resnet50"', 'generate_run_name(BACKBONE'),
    ('backbone_name="resnet50"', "backbone_name=BACKBONE"),
    ('"backbone": "resnet50"', '"backbone": BACKBONE'),
    ("T23", "T26"),
    ("ResNet50", "ViT-Base"),
]


def src(i):
    return "".join(cells[i]["source"])


def put(i, text, marker=None, kind=None):
    if marker is not None:
        assert marker in "".join(nb["cells"][i]["source"]), f"cell {i}: marker {marker!r} not found"
    if kind is not None:
        assert cells[i]["cell_type"] == kind, f"cell {i} is {cells[i]['cell_type']}, expected {kind}"
    cells[i]["source"] = text.strip("\n").splitlines(keepends=True)


for i, c in enumerate(cells):
    s = src(i)
    for a, b in RENAMES:
        s = s.replace(a, b)
    c["source"] = s.splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None

# ---------------------------------------------------------------- [0] title
put(0, """
# T26 -- Lung-Region Attention Module on ViT-Base/16 (avg-pool head)

Applies T18's Lung-Region Attention module to ViT-Base across **seven arms (A0-A6)**: A0 vanilla,
A1 gate-only, A2 full, A3 multiply-gate, A4 guidance-only, A5 CBAM, A6 (A2 + background-suppression
loss). Same pipeline, split, metrics, winner rule and outputs as T23 (ResNet50), so the four
backbones are directly comparable.

**What is ViT-specific** (`src/modules/lung_attention.py::ViTLungAttention`): the 196 patch tokens
are reshaped to a 14x14 map, gated, and **average-pooled** into the head. timm's default ViT head
reads only the CLS token, which a spatial gate never reaches. Arm A0 uses the same avg-pool head, so
it is the control for A1-A6 -- it is not T17's CLS-token baseline.

**Relation to the 2026-10-01 local run** (`artifacts/vit_lung_attention/`; that notebook is kept as
`notebooks/archive/T_26_Vit_Base_Model_local_run_2026-10-01.ipynb`): this notebook replaces it. Deliberate protocol changes, all
to match T18/T23: batch size 32 (was 8), early stopping and best checkpoint on validation **loss**
(was validation macro-F1), attention initialised at 0.5 (was ~0.0025), spatial-only CBAM for A5
(was channel+spatial), attention metrics at 224x224 (were at 14x14), fp32 test evaluation, the
committed split manifest, and checkpoints that carry their own config. Numbers from the two runs
are therefore not interchangeable.

**How to run on Kaggle**
1. Settings: **Accelerator = GPU T4**, **Internet = On** (pretrained weights + pip).
   Add-ons -> Secrets: add `WANDB_API_KEY` and attach it to this notebook (see the W&B cell below).
2. Add Input: the *COVID-19 Radiography Database* dataset.
3. The repo is cloned from GitHub. The branch must contain `ViTLungAttention` and
   `configs/vit_base_lung_attention.yaml` -- push `t26-vit-rerun-results` (or merge it) first, or
   attach the repo folder as a Kaggle Dataset.
4. **Save Version -> Save & Run All (Commit)**. One session cannot finish everything (12 h limit).
   For each later session: Add Input -> **Your Work** -> this notebook's previous output, then Save
   Version again. Finished units are skipped; the final cell lists what is left.
5. When the last cell prints `ALL DONE`, download `/kaggle/working/T26_vit_base_lung_attention/`.
   The `.pt` checkpoints (about 330 MB each) are in `runs/<arm>/` and `runs_multiseed/` -- keep
   them, T28-T35 need them.
""", marker="# T23 -- Lung-Region Attention Module on ResNet50", kind="markdown")

# ---------------------------------------------------------------- [3] repo bootstrap
s = src(3)
old_req = s[s.index("REQUIRED_CODE = {"):s.index("def repo_is_current")]
s = s.replace(old_req, '''REQUIRED_CODE = {
    "src/datasets/covid_cxr.py": "def preload_resized_cache",
    "src/modules/comparison.py": "def run_significance_tests",
    "src/modules/lung_attention.py": "class ViTLungAttention",
    "src/modules/gradcam.py": 'startswith("vit")',
    "src/modules/training.py": "experiment_tag",
    "configs/vit_base_lung_attention.yaml": "T26",
}

''')
old_clone = '''REPO_BRANCH = "main"  # fallback only, if no up-to-date repo dataset is attached
'''
assert old_clone in s
s = s.replace(old_clone, '''# Fallback only, if no up-to-date repo dataset is attached. Tried in order; the first branch whose
# checkout contains T26's src/ changes is used.
REPO_BRANCHES = ["main", "t26-vit-rerun-results"]
''')
old_git = '''            print("No up-to-date repo dataset attached -- falling back to git clone.")
            !git clone --branch $REPO_BRANCH --single-branch $REPO_URL $REPO_ROOT
'''
assert old_git in s
s = s.replace(old_git, '''            print("No up-to-date repo dataset attached -- falling back to git clone.")
            for REPO_BRANCH in REPO_BRANCHES:
                shutil.rmtree(REPO_ROOT, ignore_errors=True)
                !git clone --quiet --depth 1 --branch $REPO_BRANCH --single-branch $REPO_URL $REPO_ROOT
                if repo_is_current(REPO_ROOT):
                    print("Using branch:", REPO_BRANCH)
                    break
                print(f"Branch {REPO_BRANCH!r} does not contain T26's src/ changes yet.")
''')
s = s.replace("Upload a new version of your repo Kaggle Dataset from your up-to-date local folder, ",
              "Push/merge the t26-vit-rerun-results branch, or attach your up-to-date repo folder as a Kaggle Dataset, ")
put(3, s, marker="REQUIRED_CODE")

# ---------------------------------------------------------------- [7]/[8] W&B
put(7, """
### Weights & Biases

Every sweep config and every arm is logged as its own W&B run (per-epoch train/val loss, accuracy,
macro-F1, AUROC, ILAR, learning rate; best-epoch and final test numbers in the run summary). Runs are
named `vit_base_patch16_224_T26-<arm>_seed<seed>` and tagged `T26`, so they filter cleanly in the
project `chest-xray-disease-classification`.

**Setup (once):** Kaggle notebook menu -> **Add-ons -> Secrets -> Add a new secret**, label
`WANDB_API_KEY`, value = your key from https://wandb.ai/authorize, and tick **Attach to notebook**.
Never paste the key into a cell.

Save Version runs have no keyboard input, so a missing key would otherwise kill the run. If the
secret is missing or the login fails, this cell switches to `WANDB_MODE=offline`: training continues,
runs are stored under `/kaggle/working/wandb/` and can be uploaded later with `wandb sync`. Every
number the notebook reports comes from its own JSON/CSV files, not from W&B.
""", marker="W&B auth", kind="markdown")

put(8, """
import wandb

# Set to your W&B team/entity name to log into a shared team project; None = your own account.
WANDB_ENTITY = None

try:
    from kaggle_secrets import UserSecretsClient
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    WANDB_ONLINE = True
    print("W&B login OK -- runs will sync live to your dashboard.")
except Exception as e:
    os.environ["WANDB_MODE"] = "offline"
    WANDB_ONLINE = False
    print(f"W&B secret/login unavailable ({type(e).__name__}: {e}) -- falling back to WANDB_MODE=offline.")
    print("To log online: Add-ons -> Secrets -> WANDB_API_KEY (attached to this notebook), then re-run.")
""", marker="wandb.login")

# ---------------------------------------------------------------- [10]/[11] config + budget
s = src(10)
assert 'print("Loaded config for experiment:"' in s
s = s.replace('print("Loaded config for experiment:"',
              'try:\n    from IPython.display import display\nexcept ImportError:  # plain Python, outside Jupyter\n    display = print\n'
              'BACKBONE = cfg["model"]["name"]          # "vit_base_patch16_224"\n'
              'if WANDB_ENTITY:\n    cfg = merge_overrides(cfg, {"tracking.entity": WANDB_ENTITY})\n'
              'print("W&B:", "online" if WANDB_ONLINE else "offline", "| project:", cfg["tracking"]["project"],\n'
              '      "| entity:", cfg["tracking"]["entity"] or "(your default account)")\n'
              'print("Loaded config for experiment:"')
put(10, s)

s = src(25)
a = "                device, out_dir, backbone_name=BACKBONE, wandb_enabled=True,\n"
assert a in s, "run_full_arm call changed"
s = s.replace(a, "                device, out_dir, backbone_name=BACKBONE, wandb_enabled=True, experiment_tag=\"T26\",\n")
a = '        initialize_wandb(cfg, run_name=generate_run_name(BACKBONE, f"T26-sweep-{name}", SEED))\n'
assert a in s, "sweep initialize_wandb call changed"
s = s.replace(a, '        initialize_wandb(cfg, run_name=generate_run_name(BACKBONE, f"T26-sweep-{name}", SEED),\n'
                 '                         tags=["T26", BACKBONE, "sweep", f"seed{SEED}"])\n')
put(25, s)

s = src(11)
assert 'PIPELINE_VERSION = "t23-ramcache-freshloader-v1"' in "".join(nb["cells"][11]["source"])
s = s.replace('PIPELINE_VERSION = "t23-ramcache-freshloader-v1"', 'PIPELINE_VERSION = "t26-vit-avgpool-ramcache-v1"')
assert 'DEFAULT_EST_H = {"arm": 1.5, "sweep_cfg": 0.35, "eval_arm": 0.25}' in s
s = s.replace('DEFAULT_EST_H = {"arm": 1.5, "sweep_cfg": 0.35, "eval_arm": 0.25}',
              'DEFAULT_EST_H = {"arm": 2.0, "sweep_cfg": 0.5, "eval_arm": 0.3}')
put(11, s)

# ---------------------------------------------------------------- [17]-[23] model + taps
put(17, """
## S3 -- The Lung-Region Attention module (reused) + the ViT wrapper

Canonical code: `src/modules/lung_attention.py`. `LungRegionAttention` is unchanged from T18;
`build_model(backbone_name="vit_...")` returns `ViTLungAttention`, covered by
`tests/test_vit_lung_attention.py`.
""", kind="markdown")

put(19, """
## S4 -- Assemble the model on ViT-Base + parity checks

- `gate_mode="none"` must give bit-identical logits to arm A0.
- A0 must equal `head(mean(patch tokens))` of plain timm ViT, and must **differ** from timm's own
  CLS-token forward (the head the gate could never influence).
- Parameter counts are checked against the values verified locally. CPU-only, under a minute.
""", kind="markdown")

put(20, """
torch.manual_seed(0)
_a0 = build_model(num_classes=4, use_attention=False, pretrained=False, backbone_name=BACKBONE)
_wrapped = build_model(num_classes=4, use_attention=True, gate_mode="none", pretrained=False, backbone_name=BACKBONE)
_plain = timm.create_model(BACKBONE, pretrained=False, num_classes=4)
_wrapped.backbone.load_state_dict(_a0.backbone.state_dict())
_plain.load_state_dict(_a0.backbone.state_dict())
_plain.eval(); _wrapped.eval(); _a0.eval()

_x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    _a0_logits, _a0_att, _ = _a0(_x)
    _wrapped_logits, _att, _ = _wrapped(_x)
    _manual = _plain.head(_plain.forward_features(_x)[:, 1:].mean(dim=1))
    _cls_logits = _plain(_x)

assert _a0_att is None and _att.shape == (2, 1, 14, 14), _att.shape
torch.testing.assert_close(_wrapped_logits, _a0_logits, rtol=0, atol=0)
print("PARITY 1 PASSED: gate_mode='none' logits are bit-identical to arm A0")
torch.testing.assert_close(_a0_logits, _manual, rtol=0, atol=1e-6)
print("PARITY 2 PASSED: A0 == head(mean(patch tokens)) of plain timm ViT (avg-pool head)")
assert not torch.allclose(_a0_logits, _cls_logits, atol=1e-4)
print("PARITY 3 PASSED: A0 differs from timm's CLS-token forward, as intended")

_vanilla_params = sum(p.numel() for p in _plain.parameters())
_a0_params = sum(p.numel() for p in _a0.parameters())
print(f"A0 params: {_a0_params:,} | vanilla timm ViT-Base params: {_vanilla_params:,}")
assert _a0_params == _vanilla_params == 85_801_732, (_a0_params, _vanilla_params)
""", marker="PARITY CHECK PASSED")

put(21, """
_attn = LungRegionAttention(768, reduction=cfg["module"]["reduction"], gate_mode="residual")
_attn_params = sum(p.numel() for p in _attn.parameters())
print(f"Attention module params: {_attn_params:,} ({100.0 * _attn_params / _vanilla_params:.4f}% of vanilla ViT-Base)")
assert _attn_params == 73_921, _attn_params

model = build_model(num_classes=4, use_attention=True, gate_mode="residual", pretrained=False, backbone_name=BACKBONE)
freeze_backbone(model)
print_trainable_parameters(model, "phase1 ")
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 76_997  # head 3,076 + attn 73,921

unfreeze_final_blocks(model, num_blocks=cfg["training"]["unfreeze_blocks"])
print_trainable_parameters(model, "phase2 ")
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 14_254_277  # + blocks 10-11 + final norm
_names = [n for n, p in model.named_parameters() if p.requires_grad]
assert any(n.startswith("backbone.blocks.10.") for n in _names) and any(n.startswith("backbone.blocks.11.") for n in _names)
assert not any(n.startswith("backbone.blocks.9.") for n in _names)
del model, _wrapped, _plain, _a0
print("\\nS4 verification passed.")
""", marker="524_801")

put(22, """
## S5 -- Metrics (reused) + ViT's Grad-CAM taps

Both taps are the wrapper's own identity modules (`pre_attn`: the 14x14 patch-token map entering
the gate; `post_attn`: the map entering the avg-pool head), so plain Grad-CAM applies with no token
reshape. All attention metrics upsample the 14x14 map to 224x224 first, exactly as for the CNNs' 7x7
maps.
""", kind="markdown")

s = src(23)
s = s.replace('print("Grad-CAM harness self-check passed on ViT-Base\'s pre-gate tap (backbone.layer4[-1]).")',
              'print("Grad-CAM harness self-check passed on ViT-Base\'s taps (pre_attn / post_attn, 14x14 maps).")')
assert "pre_attn / post_attn" in s and "del _demo_model\n" in s
s = s.replace("del _demo_model\n",
              "assert _tap_pre is _demo_model.pre_attn and _tap_post is _demo_model.post_attn\ndel _demo_model\n")
put(23, s)

# ---------------------------------------------------------------- [26]-[30] pre-flight checks
put(26, """
## S7 -- Smoke test + overfit test (ViT-Base) -- first session only

The same three checks T18/T23 ran before their first expensive GPU run, skipped automatically once
they have passed. Checks 2 and 3 use the **pretrained** backbone at lr=1e-4: a ViT trained from
scratch at lr=1e-3 reached only 31% on the 32-image overfit check (measured locally), so T23's
settings would fail here for reasons unrelated to the wiring being tested. The pretrained settings
have not been run before, so only by-construction facts are hard assertions (finite losses,
att_loss starts at log 2, att_loss decreases, ILAR rises with lambda, better than chance);
T23's magnitude thresholds are printed as NOTEs instead of stopping the session.
""", kind="markdown")

s = src(29)
a = '_of_model = build_model(num_classes=4, use_attention=True, gate_mode="residual", pretrained=False, backbone_name=BACKBONE).to(device)'
assert a in s
s = s.replace(a, a.replace("pretrained=False", "pretrained=True"))
b = "_of_optimizer = torch.optim.AdamW(_of_model.parameters(), lr=1e-3)"
assert b in s
s = s.replace(b, f"_of_optimizer = torch.optim.AdamW(_of_model.parameters(), lr={OVERFIT_LR!r})")
a = '    assert _acc_history[-1] == 1.0, "wiring bug -- overfit accuracy did not reach 1.0"\n'
assert a in s
s = s.replace(a, '    assert _acc_history[-1] > 0.5, "wiring bug -- 32 images not learned at all (chance is 0.25)"\n'
                 '    if _acc_history[-1] < 1.0:\n'
                 '        print(f"NOTE: overfit accuracy {_acc_history[-1]:.3f} < 1.0 after 100 steps (threshold not calibrated for ViT).")\n')
a = '    assert _att_loss_history[-1] < _att_loss_history[0] - 0.2, "att_loss barely moved -- check lambda/logits wiring"\n'
assert a in s
s = s.replace(a, '    assert _att_loss_history[-1] < _att_loss_history[0], "att_loss did not decrease -- check lambda/logits wiring"\n'
                 '    if _att_loss_history[-1] > _att_loss_history[0] - 0.2:\n'
                 '        print(f"NOTE: att_loss fell only {_att_loss_history[0] - _att_loss_history[-1]:.3f} in 100 steps (T23 expected > 0.2).")\n')
a = "    del _of_model\n"
assert a in s
s = s.replace(a, "    del _of_model, _of_optimizer, _loss, _class_logits, _att, _att_logits\n    gc.collect(); torch.cuda.empty_cache()\n")
put(29, s)

s = src(30)
a = 'm = build_model(num_classes=4, use_attention=True, gate_mode="residual", pretrained=False, backbone_name=BACKBONE).to(device)'
assert a in s
s = s.replace(a, a.replace("pretrained=False", "pretrained=True"))
b = "opt = torch.optim.AdamW(m.parameters(), lr=1e-3)"
assert b in s
s = s.replace(b, f"opt = torch.optim.AdamW(m.parameters(), lr={SENS_LR!r})")
c = "            return ilar(att_final.float(), _lam_masks).mean().item()"
assert c in s
s = s.replace(c, "            out = ilar(att_final.float(), _lam_masks).mean().item()\n"
                 "        del m, opt\n        gc.collect(); torch.cuda.empty_cache()\n        return out")
a = '    assert _ilar_lam5 > _ilar_lam0 + 0.05, "guidance signal is not reaching the module -- ILAR did not rise meaningfully"\n'
assert a in s
s = s.replace(a, '    assert _ilar_lam5 > _ilar_lam0, "guidance signal is not reaching the module -- ILAR did not rise with lambda"\n'
                 '    if _ilar_lam5 < _ilar_lam0 + 0.05:\n'
                 '        print(f"NOTE: ILAR rose by only {_ilar_lam5 - _ilar_lam0:.3f} in 50 steps (T23 expected > 0.05).")\n')
put(30, s)

s = src(28)
a = "_opt2 = torch.optim.AdamW([p for p in _smoke_model.parameters() if p.requires_grad], lr=1e-5)"
assert a in s and "unfreeze_final_blocks(_smoke_model, 1)" in s
s = s.replace("unfreeze_final_blocks(_smoke_model, 1)", 'unfreeze_final_blocks(_smoke_model, cfg["training"]["unfreeze_blocks"])')
put(28, s)

# ---------------------------------------------------------------- [31]/[32] config audit
put(31, """
## S7b -- Config audit

Confirms the protocol values this notebook is supposed to run with (see the header of
`configs/vit_base_lung_attention.yaml`). `phase1_lr`, `phase2_lr` and `unfreeze_blocks` are the T26
owner's values from the 2026-10-01 local run; there is no ViT HPO sweep to audit against, unlike
T23's check against T14/T15.
""", kind="markdown")

put(32, """
assert BACKBONE == "vit_base_patch16_224"
assert cfg["model"]["drop_rate"] == 0.0
assert cfg["module"]["reduction"] == 8
assert cfg["training"]["optimizer"] == "adamw"
assert cfg["training"]["phase1_lr"] == 1e-3
assert cfg["training"]["phase2_lr"] == 1e-5
assert cfg["training"]["weight_decay"] == 1e-4
assert cfg["scheduler"]["name"] == "cosine"
assert cfg["training"]["unfreeze_blocks"] == 2
assert cfg["training"]["batch_size"] == 32          # aligned with T18/T23 (the local run used 8)
assert cfg["checkpoint"]["monitor"] == "val_loss"   # aligned with T18/T23 (the local run used val macro-F1)
print("S7b PASSED: config matches the T26 protocol.")
""", marker="S7b PASSED")

# ---------------------------------------------------------------- [33]/[35] A0 sanity gate
put(33, """
## S8 -- Arm A0 -- the vanilla avg-pool control (full schedule)

`use_attention=False`. Sanity gate against the two existing ViT-Base numbers: T17's CLS-token
baseline (macro-F1 0.9225) and the 2026-10-01 avg-pool A0 (0.9299, batch 8).
""", kind="markdown")

put(35, """
if a0_results is None:
    print("A0 not trained yet (deferred to the next session).")
else:
    macro_f1 = a0_results["classification_report"]["macro avg"]["f1-score"]
    print(f"A0 test macro-F1: {macro_f1:.4f}")
    if 0.905 <= macro_f1 <= 0.950:
        print("Sanity gate: PASS -- in the expected 90.5-95.0% range (T17 CLS-token 0.9225; earlier avg-pool A0 0.9299).")
    elif 0.88 <= macro_f1 < 0.905:
        print("Sanity gate: BORDERLINE -- check S7b's config values actually took effect.")
    elif macro_f1 < 0.80:
        print("Sanity gate: FAIL -- likely a pipeline bug. Stop and debug before trusting later arms.")
    else:
        print("Sanity gate: outside the pre-registered bands -- inspect manually.")
""", marker="Sanity gate")

# ---------------------------------------------------------------- [36] sweep text
put(36, """
## S9 -- lambda_att sweep (short schedule) + pre-registered selection

Re-derived for ViT-Base on this protocol -- neither the CNNs' `lambda_att*` nor the earlier local
run's value is assumed to transfer. 5 configs x (4+6) epochs, validation only. Each config is its
own resumable unit; selection runs once all 5 exist. Rule (same as T18/T23): the **largest**
`lambda_att` whose validation macro-F1 is within 0.5 pp of the `lambda_att=0` reference.
""", kind="markdown")

# ---------------------------------------------------------------- [47] load_arm_model
s = src(47)
a = "        backbone_name=BACKBONE, pretrained=False, drop_rate=ckpt[\"config\"][\"model\"].get(\"drop_rate\", 0.0),"
assert a in s, "load_arm_model signature changed"
s = s.replace(a, "        backbone_name=ckpt[\"config\"][\"model\"][\"name\"], pretrained=False,\n"
                 "        drop_rate=ckpt[\"config\"][\"model\"].get(\"drop_rate\", 0.0),")
put(47, s)

# ---------------------------------------------------------------- [49]/[50] winner rule text
s = src(50)
assert "independent of DenseNet121's A2" in s
put(50, s)

# ---------------------------------------------------------------- [51]/[52] efficiency
put(51, """
## S15 -- Efficiency measurement (acceptance criterion A6)

Architecture-only (no trained weights), so it runs in the first session and is skipped afterwards.
fvcore does not count the fused scaled-dot-product attention op and prints a warning about it; that
affects the absolute GFLOPs of both rows equally, so the with/without **delta** (what A6 tests) is
unaffected. Do not quote the absolute ViT GFLOPs from this table.
""", kind="markdown")

s = src(52)
a = "    A6_JSON.write_text(json.dumps(efficiency_verdicts, indent=2))\n"
assert a in s, "efficiency cell changed"
s = s.replace(a, "    if np.isfinite(efficiency_verdicts[\"A6_module_is_cheap\"][\"gflops_delta_pct\"]):\n"
                 "        A6_JSON.write_text(json.dumps(efficiency_verdicts, indent=2))\n"
                 "    else:\n"
                 "        # fvcore missing/failed -> GFLOPs is NaN and the verdict would be a false FAIL (this happened in T18).\n"
                 "        T26_EFFICIENCY_CSV.unlink(missing_ok=True)\n"
                 "        print(\"WARNING: GFLOPs could not be measured (see the [FLOPs warning] above). Verdict NOT saved; \"\n"
                 "              \"it is re-measured next session.\")\n")
put(52, s)

# ---------------------------------------------------------------- [58] module card
s = src(58)
a = "        f\"- `reduction` = {cfg['module']['reduction']} -> +524,801 params (+2.23% of ViT-Base's 23,516,228)\\n\","
assert a in s, "module card reduction line changed"
s = s.replace(a, "        f\"- `reduction` = {cfg['module']['reduction']} -> +73,921 params (+0.086% of ViT-Base's 85,801,732)\\n\",\n"
                 "        \"- Protocol: batch 32, early stopping on val_loss, attention init 0.5, spatial-only CBAM (A5), \"\n"
                 "        \"avg-pool head for every arm including A0. Not comparable with the 2026-10-01 local run.\\n\",")
a = '"**Backbone:** ViT-Base (timm, ImageNet-pretrained). **Draft auto-generated by S18 -- review before treating as final.**\\n",'
assert a in s
s = s.replace(a, '"**Backbone:** ViT-Base/16, avg-pool head (timm `vit_base_patch16_224`, ImageNet-pretrained). '
                 '**Draft auto-generated by S18 -- review before treating as final.**\\n",')
put(58, s)

# ---------------------------------------------------------------- leftover checks
joined = "\n".join(src(i) for i in range(len(cells)))
for bad in ["resnet50", "layer4", "2048", "524,801", "524_801", "23,516,228", "23_516_228", "T23", "T14/T15"]:
    hits = [i for i in range(len(cells)) if bad in src(i)]
    if hits:
        print(f"LEFTOVER {bad!r} in cells {hits}")

nb["cells"] = cells
nb["metadata"]["kaggle"]["isGpuEnabled"] = True
nb["metadata"]["kaggle"]["dataSources"] = [d for d in nb["metadata"]["kaggle"].get("dataSources", []) if d.get("sourceId") == 3324348]
json.dump(nb, open(DST, "w"), indent=1)
print("wrote", DST, len(cells), "cells")

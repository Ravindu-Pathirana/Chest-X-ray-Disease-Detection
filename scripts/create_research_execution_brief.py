from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "High_Priority_Research_Execution_Brief.docx"


def run(text: str, *, bold: bool = False, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph(text: str = "", *, style: str | None = None, bold: bool = False,
              size: int | None = None, before: int = 0, after: int = 100) -> str:
    ppr_parts = []
    if style:
        ppr_parts.append(f'<w:pStyle w:val="{style}"/>')
    if before or after:
        ppr_parts.append(f'<w:spacing w:before="{before}" w:after="{after}"/>')
    ppr = f"<w:pPr>{''.join(ppr_parts)}</w:pPr>" if ppr_parts else ""
    return f"<w:p>{ppr}{run(text, bold=bold, size=size) if text else ''}</w:p>"


def bullet(text: str, level: int = 0) -> str:
    return (
        "<w:p><w:pPr>"
        f'<w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="1"/></w:numPr>'
        '<w:spacing w:after="60"/></w:pPr>'
        f"{run(text)}</w:p>"
    )


def cell(text: str, *, bold: bool = False, width: int = 2200, shade: str | None = None) -> str:
    shade_xml = f'<w:shd w:fill="{shade}"/>' if shade else ""
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade_xml}</w:tcPr>'
        f'<w:p><w:pPr><w:spacing w:after="40"/></w:pPr>{run(text, bold=bold, size=19)}</w:p></w:tc>'
    )


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    borders = (
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="B7B7B7"/>'
        '<w:left w:val="single" w:sz="4" w:color="B7B7B7"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="B7B7B7"/>'
        '<w:right w:val="single" w:sz="4" w:color="B7B7B7"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D9D9D9"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D9D9D9"/></w:tblBorders>'
    )
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>{borders}</w:tblPr>']
    out.append("<w:tr>" + "".join(cell(h, bold=True, width=w, shade="D9EAF7") for h, w in zip(headers, widths)) + "</w:tr>")
    for row in rows:
        out.append("<w:tr>" + "".join(cell(v, width=w) for v, w in zip(row, widths)) + "</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def build_document() -> str:
    body: list[str] = []
    body.append(paragraph("High-Priority Research Execution Brief", style="Title", after=80))
    body.append(paragraph("Chest X-ray Disease Detection — computational completion plan", style="Subtitle", after=180))
    body.append(paragraph("Repository reviewed: Ravindu-Pathirana/Chest-X-ray-Disease-Detection, origin/main at ea67e44", size=18, after=40))
    body.append(paragraph("Prepared: 29 September 2026", size=18, after=180))

    body.append(paragraph("Executive recommendation", style="Heading1"))
    body.append(paragraph(
        "The high-priority computational work can be completed in approximately five to six days if the scope is frozen to the three mature CNN backbones—DenseNet121, ResNet50 and EfficientNet-B0—and the trained checkpoints are immediately available. Most remaining priority work is inference-only evaluation rather than training. Full ViT explainability adaptation and unrestricted expansion of the original five-axis roadmap should not be allowed to block the CNN study."
    ))
    body.append(paragraph(
        "Recommended paper-level question: Does lung-supervised attention improve anatomical evidence localization, and does that improvement correspond to reduced reliance on image background shortcuts?"
    ))

    body.append(paragraph("Work already completed", style="Heading1"))
    for item in [
        "Fixed train/validation/test split for 21,165 primary-dataset images, with a 3,175-image test partition.",
        "Classification baselines for DenseNet121, ResNet50, EfficientNet-B0 and ViT-Base.",
        "Lung-Region Attention implementation with residual, multiplicative, guidance-only, CBAM and background-suppression variants.",
        "Seed-42 ablations and three-seed selected-arm comparisons for DenseNet121, ResNet50 and EfficientNet-B0.",
        "Grad-CAM Energy-Inside-Lung (EIL), ILAR, attention Dice/IoU, per-image predictions and paired significance-testing infrastructure.",
        "EfficientNet-B0 calibration, efficiency and background-counterfactual results.",
        "RSNA external dataset preparation for approximately 26,684 patients.",
        "Automated tests for the main data, attention, Grad-CAM, calibration and counterfactual components.",
    ]:
        body.append(bullet(item))

    body.append(paragraph("High-priority run matrix", style="Heading1"))
    body.append(table(
        ["Work item", "Current repository support", "What is missing", "Feasibility now"],
        [
            ["CNN counterfactual evaluation", "Implemented and tested in src/modules/counterfactual.py; EfficientNet outputs exist.", "DenseNet and ResNet checkpoints plus standardized runs.", "YES — inference-only if checkpoints load."],
            ["CNN calibration", "Calibration module and tests exist; EfficientNet outputs exist.", "Run on DenseNet and ResNet using validation-fitted temperature scaling.", "YES — inference-only if checkpoints load."],
            ["Standardized EIL/statistics", "Grad-CAM, EIL, per-image files and comparison utilities exist.", "Regenerate one consistent cross-backbone table and repair DenseNet summary inconsistency.", "YES — mostly aggregation plus targeted inference."],
            ["Lung/background occlusion", "Background perturbation pipeline already preserves lung pixels.", "Add inverse-mask lung removal/blur and probability-drop reporting.", "YES — small code extension and inference."],
            ["Efficiency comparison", "FLOP/parameter/latency utilities and several outputs exist.", "Run identical hardware/protocol measurements for final model pairs.", "YES — short benchmark runs."],
            ["RSNA external evaluation", "External manifest and DICOM conversion preparation exist.", "Freeze label mapping; implement/restore final evaluation runner; load compatible checkpoints.", "CONDITIONAL — technically feasible, method decision required."],
            ["ViT explainability", "ViT A0–A6 classification artifacts exist.", "Validate CLS-token/patch-token path, explanation target and comparable EIL pipeline; may require retraining.", "HIGH RISK — do not place on critical path."],
        ],
        [1900, 2700, 2700, 1800],
    ))

    body.append(paragraph("Critical dependency: model checkpoints", style="Heading1"))
    body.append(paragraph(
        "The repository intentionally excludes .pt checkpoints. Result CSV/JSON files are present, but new inference evaluations require the actual trained weights. Before scheduling any GPU jobs, locate and verify the six primary checkpoints: DenseNet A0/A2, ResNet A0/A3 and EfficientNet A0/A3. If a checkpoint cannot be recovered, retraining becomes the largest schedule risk."
    ))

    body.append(paragraph("Proposed five-to-six-day execution plan", style="Heading1"))
    phases = [
        ("Day 1 — Clean start, inventory and smoke tests", [
            "Clone or create a clean checkout of the latest main branch and record the commit hash.",
            "Build a manifest of every required model, arm, seed, checkpoint and pending evaluation.",
            "Load all six primary CNN checkpoints and run 100-image smoke tests.",
            "Freeze metric definitions, the fixed CAM subset and RSNA label mapping.",
            "Correct the DenseNet multi-seed summary source and choose one authoritative aggregation script.",
        ]),
        ("Day 2 — DenseNet and ResNet inference", [
            "Run counterfactual zero/shuffle/noise evaluation for baseline and selected attention arms.",
            "Run calibration with temperature fitted only on validation predictions.",
            "Regenerate EIL and paired per-image comparisons where needed.",
        ]),
        ("Day 3 — EfficientNet verification and occlusion", [
            "Reproduce the committed EfficientNet counterfactual and calibration summaries.",
            "Implement and run lung-versus-background occlusion for all six CNN models.",
            "Save per-image outputs so confidence intervals and paired tests can be regenerated.",
        ]),
        ("Day 4 — External and efficiency evaluation", [
            "Run RSNA external evaluation only after the mapping and primary metric are frozen.",
            "Benchmark parameters, FLOPs and latency under a single controlled environment.",
            "Audit ViT feasibility without allowing it to delay the CNN results.",
        ]),
        ("Day 5 — Recovery and aggregation", [
            "Repeat failed jobs and recover missing outputs.",
            "Generate the cross-backbone classification, localization, calibration, robustness and efficiency tables.",
            "Calculate paired confidence intervals and statistical tests from per-image outputs.",
        ]),
        ("Day 6 — Verification and experiment freeze", [
            "Check image counts, image-path pairing, split separation and configuration consistency.",
            "Verify every reported aggregate against raw outputs.",
            "Archive configs, environment information, logs and immutable checkpoint locations.",
            "Freeze experiments; after this point, rerun only genuine errors rather than tuning results.",
        ]),
    ]
    for heading, items in phases:
        body.append(paragraph(heading, style="Heading2", before=80, after=60))
        for item in items:
            body.append(bullet(item))

    body.append(paragraph("Required final outputs", style="Heading1"))
    for item in [
        "A single machine-readable master table for the six primary CNN models.",
        "Per-image EIL, counterfactual, occlusion and prediction outputs.",
        "Calibration summaries fitted on validation and evaluated on test.",
        "Paired baseline-versus-attention statistical results for each backbone.",
        "A small, verified set of qualitative Grad-CAM and perturbation examples.",
        "A run manifest containing commit, checkpoint, configuration, seed, dataset split and environment details.",
    ]:
        body.append(bullet(item))

    body.append(paragraph("Immediate decisions", style="Heading1"))
    for item in [
        "Confirm that the three-CNN study is the minimum complete deliverable.",
        "Confirm checkpoint locations before promising the six-day deadline.",
        "Treat ViT as conditional, not critical-path work.",
        "Decide the RSNA label mapping before viewing external results.",
        "Use the wording ‘anatomical localization’ until intervention-based tests support a stronger faithfulness or shortcut-reduction claim.",
    ]:
        body.append(bullet(item))

    body.append(paragraph("Go/no-go conclusion", style="Heading1"))
    body.append(paragraph(
        "GO if all six CNN checkpoints can be loaded on Day 1 and at least two GPU workers are available. The repository already contains most of the necessary evaluation code. CONDITIONAL GO for RSNA because the final label mapping and evaluation runner must be fixed. NO-GO for making full ViT explainability a mandatory deliverable within the same deadline unless its current implementation passes an immediate architectural audit."
    ))

    body.append(
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}</w:body></w:document>"
    )


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="38"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:i/><w:color w:val="44546A"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:color w:val="2F5597"/><w:sz w:val="24"/></w:rPr></w:style>
</w:styles>"""

NUMBERING = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0">
    <w:multiLevelType w:val="hybridMultilevel"/>
    <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr><w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/></w:rPr></w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>High-Priority Research Execution Brief</dc:title><dc:creator>OpenAI Codex</dc:creator>
  <dc:subject>Chest X-ray explainability experiment plan</dc:subject>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application></Properties>"""
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("word/document.xml", build_document())
        zf.writestr("word/_rels/document.xml.rels", DOC_RELS)
        zf.writestr("word/styles.xml", STYLES)
        zf.writestr("word/numbering.xml", NUMBERING)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
    print(OUTPUT)


if __name__ == "__main__":
    main()

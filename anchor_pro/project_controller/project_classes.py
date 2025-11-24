import anchor_pro.model as m




from dataclasses import dataclass, field
from typing import Optional, List

from anchor_pro.ap_types import FactorMethod


@dataclass(frozen=True, slots=True)
class HardwareSelection:
    base_anchor_id: Optional[str] = None
    bracket_id: Optional[str] = None
    wall_anchor_id: Optional[str] = None
    cxn_anchor_id: Optional[str] = None  # SMS size at connections
    base_strap: Optional[str] = None

@dataclass(frozen=True, slots=True)
class HardwareSelectionPlan:
    base_anchor_list: List[Optional[str]] = field(default_factory=lambda: [None])
    bracket_list: List[Optional[str]] = field(default_factory=lambda: [None])
    wall_anchor_list: List[Optional[str]] = field(default_factory=lambda: [None])
    cxn_anchor_list: List[Optional[str]] = field(default_factory=lambda: [None])
    # base_strap_list: List[Optional[str]] = field(default_factory=lambda: [None])  #todo: base straps

@dataclass(frozen=True, slots=True)
class AnalysisRun:
    equipment_id: str
    # model_id: UUID #todo: ask gpt about UUIDs, what they are, how they are used
    hardware_selection: HardwareSelection
    omit_analysis: bool
    results: Optional[m.ElementResults] = None
    solutions: Optional[dict[FactorMethod, m.Solution]] = None

ModelId = str
GroupId = str

@dataclass(slots=True)
class ModelRecord:
    model: m.EquipmentModel
    analysis_runs: list[AnalysisRun] = field(default_factory=list)
    governing_run: Optional[int] = None
    group: Optional[GroupId] = None

    for_report: bool = False
    report_section_name: Optional[str] = None
    index_in_group: Optional[int] = 0
    group_summary: Optional[dict[ModelId, List]] = None
    frontmatter_file: Optional[str] = None
    endmatter_file: Optional[str] = None
    include_pull_test: bool = False
    omit_bracket_output: bool = False
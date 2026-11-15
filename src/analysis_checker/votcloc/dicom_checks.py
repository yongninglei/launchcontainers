"""
Classes for DICOM session and protocol checking.
"""

from pathlib import Path
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field
import re

from launchcontainers.log_setup import console


@dataclass
class ProtocolCheck:
    """Results for a single protocol/series."""

    protocol_name: str
    series_number: str
    dicom_count: int
    expected_count: Optional[int] = None

    @property
    def is_valid(self) -> bool:
        """Check if dicom count matches expected."""
        if self.expected_count is None:
            return True
        return self.dicom_count == self.expected_count

    @property
    def protocol_type(self) -> str:
        """Determine protocol type from name."""
        name_lower = self.protocol_name.lower()

        if "t1" in name_lower and "mp2rage" in name_lower:
            if "inv1" in name_lower or "inv-1" in name_lower:
                return "t1_INV1"
            elif "inv2" in name_lower or "inv-2" in name_lower:
                return "t1_INV2"
            elif "uni" in name_lower:
                return "t1_UNI"
            return "t1"

        if "t2" in name_lower:
            return "t2"

        if "dmri" in name_lower or "dwi" in name_lower:
            if "sbref" in name_lower:
                if "pha" in name_lower or "phase" in name_lower:
                    return "dwi_SBRef_Pha"
                return "dwi_SBRef_mag"
            elif "pha" in name_lower or "phase" in name_lower:
                return "dwi_Pha"
            return "dwi_mag"

        if "floc" in name_lower:
            if "sbref" in name_lower:
                if "pha" in name_lower or "phase" in name_lower:
                    return "floc_SBRef_Pha"
                return "floc_SBRef_mag"
            elif "pha" in name_lower or "phase" in name_lower:
                return "floc_Pha"
            return "floc_mag"

        # Retinotopy and PRF - extract task name (retCB, retFF, retRW, prf_CB, prf_word, etc.)
        if "ret" in name_lower or "prf" in name_lower:
            # Match patterns like: retCB, retFF, retRW, prf_CB, prf_word, etc.
            task_match = re.search(
                r"(ret[a-z]+|prf[_-]?[a-z]+)", name_lower, re.IGNORECASE
            )
            if task_match:
                task = task_match.group(0).replace("_", "").replace("-", "")
            else:
                task = "ret"

            main_type = f"ret_{task}"

            if "sbref" in name_lower:
                if "pha" in name_lower or "phase" in name_lower:
                    return f"{main_type}_SBRef_Pha"
                return f"{main_type}_SBRef_mag"
            elif "pha" in name_lower or "phase" in name_lower:
                return f"{main_type}_Pha"
            return f"{main_type}_mag"

        return "unknown"


@dataclass
class SessionCheck:
    """Results for a single session."""

    subject: str
    session: str
    session_dir: Path
    protocols: List[ProtocolCheck] = field(default_factory=list)
    combined_sessions: List[str] = field(
        default_factory=list
    )  # Track which physical sessions were combined
    exists: bool = True  # Whether the session directory exists
    dicom_depth: Optional[int] = None  # Depth of DICOM files from session directory

    @property
    def session_id(self) -> str:
        """Get session identifier."""
        return f"sub-{self.subject}/{self.session}"

    def debug_protocol_types(self):
        """Debug: print all protocol types found."""
        console.print(
            f"\n[yellow]DEBUG: Protocol types for {self.session_id}:[/yellow]"
        )
        if self.combined_sessions:
            console.print(
                f"  [cyan]Combined from physical sessions: {', '.join(self.combined_sessions)}[/cyan]"
            )
        for p in self.protocols:
            console.print(f"  {p.protocol_name}")
            console.print(f"    -> Type: {p.protocol_type}")
            console.print(f"    -> DICOMs: {p.dicom_count}")

    def get_protocol_types(self) -> Set[str]:
        """Get set of all protocol types found."""
        types = set()
        for protocol in self.protocols:
            ptype = protocol.protocol_type
            # Normalize ret tasks for checking
            if ptype.startswith("ret"):
                # Extract base type (e.g., "retrw_mag" -> "retrw", "retrw_SBRef_mag" -> "retrw")
                base = ptype.split("_")[0]
                types.add(base)
            else:
                types.add(ptype)
        return types

    def has_t1_mp2rage(self) -> Tuple[bool, List[str]]:
        """Check if session has complete T1 MP2RAGE."""
        issues = []
        has_inv1 = any(p.protocol_type == "t1_INV1" for p in self.protocols)
        has_inv2 = any(p.protocol_type == "t1_INV2" for p in self.protocols)
        has_uni = any(p.protocol_type == "t1_UNI" for p in self.protocols)

        if not has_inv1:
            issues.append("Missing T1 INV1")
        if not has_inv2:
            issues.append("Missing T1 INV2")
        if not has_uni:
            issues.append("Missing T1 UNI")

        return (has_inv1 and has_inv2 and has_uni), issues

    def has_t2(self) -> Tuple[bool, List[str]]:
        """Check if session has T2."""
        has_t2 = any(p.protocol_type == "t2" for p in self.protocols)
        issues = [] if has_t2 else ["Missing T2"]
        return has_t2, issues

    def has_dwi(self) -> Tuple[bool, List[str]]:
        """Check if session has complete DWI."""
        issues = []

        # Check b06_PA
        has_b06_mag = any(
            "b06" in p.protocol_name.lower() and p.protocol_type == "dwi_mag"
            for p in self.protocols
        )
        has_b06_pha = any(
            "b06" in p.protocol_name.lower() and p.protocol_type == "dwi_Pha"
            for p in self.protocols
        )
        has_b06_sbref_mag = any(
            "b06" in p.protocol_name.lower() and p.protocol_type == "dwi_SBRef_mag"
            for p in self.protocols
        )
        has_b06_sbref_pha = any(
            "b06" in p.protocol_name.lower() and p.protocol_type == "dwi_SBRef_Pha"
            for p in self.protocols
        )

        if not has_b06_mag:
            issues.append("Missing DWI b06_PA mag")
        if not has_b06_pha:
            issues.append("Missing DWI b06_PA Pha")
        if not has_b06_sbref_mag:
            issues.append("Missing DWI b06_PA SBRef_mag")
        if not has_b06_sbref_pha:
            issues.append("Missing DWI b06_PA SBRef_Pha")

        # Check dir104_AP
        has_dir104_mag = any(
            "dir104" in p.protocol_name.lower() and p.protocol_type == "dwi_mag"
            for p in self.protocols
        )
        has_dir104_pha = any(
            "dir104" in p.protocol_name.lower() and p.protocol_type == "dwi_Pha"
            for p in self.protocols
        )
        has_dir104_sbref_mag = any(
            "dir104" in p.protocol_name.lower() and p.protocol_type == "dwi_SBRef_mag"
            for p in self.protocols
        )
        has_dir104_sbref_pha = any(
            "dir104" in p.protocol_name.lower() and p.protocol_type == "dwi_SBRef_Pha"
            for p in self.protocols
        )

        if not has_dir104_mag:
            issues.append("Missing DWI dir104_AP mag")
        if not has_dir104_pha:
            issues.append("Missing DWI dir104_AP Pha")
        if not has_dir104_sbref_mag:
            issues.append("Missing DWI dir104_AP SBRef_mag")
        if not has_dir104_sbref_pha:
            issues.append("Missing DWI dir104_AP SBRef_Pha")

        complete = (
            has_b06_mag
            and has_b06_pha
            and has_b06_sbref_mag
            and has_b06_sbref_pha
            and has_dir104_mag
            and has_dir104_pha
            and has_dir104_sbref_mag
            and has_dir104_sbref_pha
        )

        return complete, issues

    def has_floc(self) -> Tuple[bool, List[str]]:
        """Check if session has complete fLoc (at least 10 runs)."""
        issues = []

        # Count valid runs (mag with correct file count)
        valid_mag_runs = [
            p
            for p in self.protocols
            if p.protocol_type == "floc_mag" and p.dicom_count == 160
        ]
        valid_pha_runs = [
            p
            for p in self.protocols
            if p.protocol_type == "floc_Pha" and p.dicom_count == 160
        ]
        valid_sbref_mag = [
            p
            for p in self.protocols
            if p.protocol_type == "floc_SBRef_mag" and p.dicom_count == 1
        ]
        valid_sbref_pha = [
            p
            for p in self.protocols
            if p.protocol_type == "floc_SBRef_Pha" and p.dicom_count == 1
        ]

        num_valid_mag = len(valid_mag_runs)
        num_valid_pha = len(valid_pha_runs)
        num_valid_sbref_mag = len(valid_sbref_mag)
        num_valid_sbref_pha = len(valid_sbref_pha)

        if num_valid_mag < 10:
            issues.append(
                f"fLoc: Only {num_valid_mag} valid runs (expected at least 10)"
            )
        if num_valid_pha < num_valid_mag:
            issues.append(
                f"fLoc: Valid Pha runs ({num_valid_pha}) < valid mag runs ({num_valid_mag})"
            )
        if num_valid_sbref_mag < num_valid_mag:
            issues.append(
                f"fLoc: Valid SBRef_mag ({num_valid_sbref_mag}) < valid mag runs ({num_valid_mag})"
            )
        if num_valid_sbref_pha < num_valid_mag:
            issues.append(
                f"fLoc: Valid SBRef_Pha ({num_valid_sbref_pha}) < valid mag runs ({num_valid_mag})"
            )

        complete = (
            num_valid_mag >= 10
            and num_valid_pha >= num_valid_mag
            and num_valid_sbref_mag >= num_valid_mag
            and num_valid_sbref_pha >= num_valid_mag
        )

        return complete, issues

    def has_ret(self) -> Tuple[bool, List[str]]:
        """Check if session has retinotopy scans."""
        issues = []

        # Get all ret tasks
        ret_tasks = set()
        for p in self.protocols:
            ptype = p.protocol_type
            if ptype.startswith("ret_"):
                # Extract task name: "ret_retcb_mag" -> "retcb"
                parts = ptype.split("_")
                if len(parts) >= 2:
                    task = parts[1]
                    ret_tasks.add(task)

        if not ret_tasks:
            issues.append("Missing retinotopy scans")
            return False, issues

        # Check each ret task
        all_tasks_complete = True
        for task in ret_tasks:
            valid_mag = [
                p
                for p in self.protocols
                if p.protocol_type == f"ret_{task}_mag" and p.dicom_count == 156
            ]
            valid_pha = [
                p
                for p in self.protocols
                if p.protocol_type == f"ret_{task}_Pha" and p.dicom_count == 156
            ]
            valid_sbref_mag = [
                p
                for p in self.protocols
                if p.protocol_type == f"ret_{task}_SBRef_mag" and p.dicom_count == 1
            ]
            valid_sbref_pha = [
                p
                for p in self.protocols
                if p.protocol_type == f"ret_{task}_SBRef_Pha" and p.dicom_count == 1
            ]

            num_mag = len(valid_mag)
            num_pha = len(valid_pha)
            num_sbref_mag = len(valid_sbref_mag)
            num_sbref_pha = len(valid_sbref_pha)

            task_display = task.upper()

            # At least one complete run for this task
            if num_mag < 1:
                issues.append(f"{task_display}: No valid mag runs")
                all_tasks_complete = False
            if num_pha < num_mag:
                issues.append(
                    f"{task_display}: Valid Pha runs ({num_pha}) < valid mag runs ({num_mag})"
                )
                all_tasks_complete = False
            if num_sbref_mag < num_mag:
                issues.append(
                    f"{task_display}: Valid SBRef_mag ({num_sbref_mag}) < valid mag runs ({num_mag})"
                )
                all_tasks_complete = False
            if num_sbref_pha < num_mag:
                issues.append(
                    f"{task_display}: Valid SBRef_Pha ({num_sbref_pha}) < valid mag runs ({num_mag})"
                )
                all_tasks_complete = False

        # Complete if at least one ret task has at least one complete run
        has_at_least_one_complete = any(
            len(
                [
                    p
                    for p in self.protocols
                    if p.protocol_type == f"ret_{task}_mag" and p.dicom_count == 156
                ]
            )
            >= 1
            for task in ret_tasks
        )

        return has_at_least_one_complete and all_tasks_complete, issues

    def get_file_count_issues(self) -> List[str]:
        """Get issues related to incorrect file counts."""
        issues = []

        for protocol in self.protocols:
            expected = protocol.expected_count
            if expected is not None and protocol.dicom_count != expected:
                issues.append(
                    f"{protocol.protocol_name}: Has {protocol.dicom_count} DICOMs (expected {expected})"
                )

        return issues

    def get_issues(self) -> List[str]:
        """Get list of all issues for this session."""
        if not self.exists:
            return ["Session directory does not exist"]

        issues = []

        # Check modality presence
        has_t1, t1_issues = self.has_t1_mp2rage()
        issues.extend(t1_issues)

        has_t2, t2_issues = self.has_t2()
        issues.extend(t2_issues)

        has_dwi, dwi_issues = self.has_dwi()
        issues.extend(dwi_issues)

        has_floc, floc_issues = self.has_floc()
        issues.extend(floc_issues)

        has_ret, ret_issues = self.has_ret()
        issues.extend(ret_issues)

        return issues

    @property
    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.exists and len(self.get_issues()) == 0

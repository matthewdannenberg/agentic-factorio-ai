"""
examination/

The examination layer — runs after task completion to verify factory state
and update the SelfModel.

Reads WorldQuery. Writes SelfModel directly (has authority to promote
CANDIDATE nodes to ACTIVE). Produces AuditReports consumed by the LLM layer.

Contents
--------
audit_report.py    AuditReport, and all supporting data types (StarvedEntity,
                   Anomaly, BlueprintCandidate, DamagedEntityRecord, etc.).
                   Pure data — no LLM calls, no RCON. Shared output type of
                   both examiner modes.
examiner.py        RichExaminer [Phase 10]
auditor.py         MechanicalAuditor [Phase 10]

Phase
-----
Full implementation is Phase 10. audit_report.py is defined now because the
data types it contains are needed by the coordinator and LLM layer stubs.
"""

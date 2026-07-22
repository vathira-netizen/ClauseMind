"""Reserved Lyzr adapter; local_adapter is the working default."""

from precedent.governance.base import ApprovalGate, AuditLog, HallucinationGate, PIIRedactor


class _Deferred:
    def _deferred(self, *args, **kwargs):
        raise NotImplementedError("wire to Lyzr API — deferred, see governance/local_adapter.py for the working default")


class LyzrPIIRedactor(_Deferred, PIIRedactor):
    redact = _Deferred._deferred


class LyzrHallucinationGate(_Deferred, HallucinationGate):
    evaluate = _Deferred._deferred


class LyzrApprovalGate(_Deferred, ApprovalGate):
    approve = _Deferred._deferred


class LyzrAuditLog(_Deferred, AuditLog):
    record = _Deferred._deferred

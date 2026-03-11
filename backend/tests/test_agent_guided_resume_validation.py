from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.routers.agent import _validated_guided_resume_decision


def test_validated_guided_resume_decision_accepts_issue_confirm() -> None:
    validated = _validated_guided_resume_decision(
        {
            "type": "issue_confirm",
            "nonce": "nonce-1",
            "stage": "post_summary",
            "confirm": True,
        }
    )
    assert validated == {
        "type": "issue_confirm",
        "nonce": "nonce-1",
        "stage": "post_summary",
        "confirm": True,
    }


def test_validated_guided_resume_decision_rejects_issue_confirm_without_stage() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _validated_guided_resume_decision(
            {
                "type": "issue_confirm",
                "nonce": "nonce-1",
                "confirm": False,
            }
        )

    assert exc_info.value.status_code == 400
    assert "Invalid issue_confirm resume_decision" in str(exc_info.value.detail)

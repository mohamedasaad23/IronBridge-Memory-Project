"""Pydantic models + JSON Schema for every tool. additionalProperties: false everywhere."""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

# ---- Read tools ----
class CheckWorkerCertInput(StrictModel):
    worker_id: int = Field(..., ge=1, description="Worker primary key")
    equipment_type: Literal["CRANE", "EXCAVATOR", "SCAFFOLD", "GENERATOR"] = Field(
        ..., description="Equipment type to check certification for"
    )

class GetEquipmentStatusInput(StrictModel):
    equipment_id: int = Field(..., ge=1, description="Equipment primary key")

# ---- Write tools ----
class RequestEquipmentInput(StrictModel):
    worker_id: int = Field(..., ge=1, description="Requesting worker id")
    equipment_id: int = Field(..., ge=1, description="Equipment to request")
    site_id: int = Field(..., ge=1, description="Site where equipment will be used")

class AuthenticateSupervisorInput(StrictModel):
    worker_id: int = Field(..., ge=1, description="Supervisor worker id")
    pin: str = Field(..., min_length=4, max_length=8, description="Supervisor PIN")

class ApproveEquipmentRequestInput(StrictModel):
    request_id: int = Field(..., ge=1, description="Request to approve or reject")
    decision: Literal["APPROVED", "REJECTED"] = Field(..., description="Final decision")
    notes: Optional[str] = Field(None, max_length=500, description="Optional supervisor notes")

class GenerateComplianceReportInput(StrictModel):
    site_id: int = Field(..., ge=1, description="Site to audit")

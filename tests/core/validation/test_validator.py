# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
from pydantic import TypeAdapter

from aklr.core.validation import (
    AuditReport,
    AuditValidator,
    BenchReport,
    BenchValidator,
    Validator,
    ValidatorReport,
)


@pytest.mark.parametrize(
    ("name", "validator"),
    [("audit", AuditValidator), ("bench", BenchValidator)],
    ids=["audit", "bench"],
)
def test_a_named_validator_is_registered_and_carries_its_name(name, validator):
    assert Validator.resolve(name) is validator
    assert validator.name == name


def test_a_validator_declared_without_a_name_is_not_registered_and_keeps_its_own():
    class Unnamed(Validator):
        name = "unnamed"

    assert Unnamed.name == "unnamed"
    assert Validator.names() == ("audit", "bench")
    with pytest.raises(ValueError, match="unknown Validator: 'unnamed'; available: audit, bench"):
        Validator.resolve("unnamed")


@pytest.mark.parametrize(
    ("validator", "report_type"),
    [(AuditValidator(), AuditReport), (BenchValidator(["cuda:0"]), BenchReport)],
    ids=["audit", "bench"],
)
def test_a_validator_reports_in_the_report_type_registered_under_its_name(validator, report_type):
    assert ValidatorReport.resolve(validator.name) is report_type
    assert validator.report_type is report_type


def test_a_report_is_named_by_its_registration_and_written_with_that_name():
    assert ValidatorReport.names() == ("audit", "bench")
    report = BenchReport.skipped("no schema to validate")
    assert report.name == "bench"
    written = TypeAdapter(BenchReport).dump_python(report, mode="json")
    assert (written["name"], written["status"]) == ("bench", "skipped")


def test_a_written_report_reads_back_as_its_type_and_not_as_another():
    report = BenchReport.skipped("no schema to validate")
    written = TypeAdapter(BenchReport).dump_python(report, mode="json")
    assert TypeAdapter(BenchReport).validate_python(written) == report
    with pytest.raises(ValueError, match="the audit report cannot be named 'bench'"):
        TypeAdapter(AuditReport).validate_python(written)

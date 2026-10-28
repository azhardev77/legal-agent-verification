"""The memo rules. run_verification is not implemented yet.

Called when the agent calls submit_memo with VERIFICATION=on, and by
./helper verify RUN --memo.

    run_verification(ctx, memo) -> {"passed": bool, "feedback": [problem, ...]}

memo = {"conclusion": str, "findings": [{"item", "status", "summary", "citations": [
{"document_id", "page", "quote"}]}], "gaps": str}. ctx.case is the run's CaseRecord:
case.data["checklist"], case.data["facts"], case.data["signature_checks"] (each with
document_id, page, verdict), case.flagged(document_id), case.open_requests(). Compare
quotes with agent_kit.document_text.normalize.

Rules (one problem string per break, which the agent reads and fixes; passed = no problems):
- every checklist item has a finding, and is proven or escalated (status "open" is a problem);
- a finding marked "proven" has a recorded fact for its item, and cites at least one;
- every citation matches a recorded fact (same document, page, and quote);
- no citation is from a document flagged confidential;
- an item with needs_signature_check that is marked proven has a signature check on a
  document it cites, and every such check's verdict is "consistent" (otherwise the item
  is not proven: say so, and name it as a gap);
- no evidence request is still open.
"""

from __future__ import annotations

from agent_kit import document_text


def run_verification(ctx, memo: dict) -> dict:
    case = ctx.case
    checklist: dict = case.data.get("checklist", {})
    facts: list = case.data.get("facts", [])
    findings = memo.get("findings") or []
    by_item: dict[str, dict] = {}
    for finding in findings:
        if isinstance(finding, dict) and finding.get("item"):
            by_item.setdefault(str(finding["item"]), finding)
    problems: list[str] = []

    for item_id, entry in checklist.items():
        finding = by_item.get(item_id)
        if finding is None:
            problems.append(
                f"no finding for {item_id}: add one marked proven or escalated"
            )
            continue
        status = finding.get("status")
        if status not in ("proven", "escalated") or entry.get("status") == "open":
            problems.append(
                f"{item_id} is neither proven nor escalated "
                f"(finding says {status!r}, checklist says "
                f"{entry.get('status')!r}): prove it or escalate it"
            )
        if status == "proven":
            if not [fact for fact in facts if fact.get("item") == item_id]:
                problems.append(
                    f"{item_id} is marked proven but has no recorded fact: "
                    "record one, or leave the item unproven"
                )
            if not finding.get("citations"):
                problems.append(
                    f"{item_id} is marked proven but cites no recorded fact"
                )
            if entry.get("needs_signature_check"):
                problems.extend(_signature_problems(item_id, finding, case))

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        for citation in finding.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            document_id = str(citation.get("document_id", ""))
            page = _page(citation.get("page"))
            quote = str(citation.get("quote", ""))
            if case.flagged(document_id):
                problems.append(
                    f"{finding.get('item')}: the citation from {document_id} "
                    "is confidential and can never be cited"
                )
            if not any(
                fact.get("document_id") == document_id
                and _page(fact.get("page")) == page
                and document_text.normalize(str(fact.get("quote", "")))
                == document_text.normalize(quote)
                for fact in facts
            ):
                problems.append(
                    f"{finding.get('item')}: the citation {document_id} p{page} "
                    "has no matching recorded fact; cite only facts you recorded"
                )

    for request in case.open_requests():
        problems.append(
            f"evidence request is still open for {request['item']} "
            f"({request['id']}): resolve it before filing the memo"
        )

    return {"passed": not problems, "feedback": problems}


def _signature_problems(item_id: str, finding: dict, case) -> list:
    cited = {
        str(citation.get("document_id"))
        for citation in finding.get("citations") or []
        if isinstance(citation, dict)
    }
    checks = [
        check
        for check in case.data.get("signature_checks", [])
        if str(check.get("document_id")) in cited
    ]
    if not checks:
        return [
            f"{item_id} is not proven: no cited document was checked "
            "for signatures; name it as a gap"
        ]
    return [
        f"{check.get('document_id')} p{check.get('page')} is "
        f"{check.get('verdict')} ({check.get('reason', '')}): {item_id} "
        "is not proven; name it as a gap"
        for check in checks
        if check.get("verdict") != "consistent"
    ]


def _page(value) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value

2026-03-05 F3 QA: Real HTTP curl flow validated on `http://127.0.0.1:8001/api/agent/run`.
- Initial guided-confirm request returned `interrupted=true` with `interrupt_payload.type="auto_parse_confirm"`.
- Resume with `task_mode="sop"` returned `interrupted=false` and final answer content.
- Resume metadata included `selected_task_mode`, `applied_doc_type_scope`, and `target_language` as required.

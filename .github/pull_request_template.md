# Pull Request

## Summary of Changes

Provide a clear and concise description of the changes introduced in this pull request.

- What problem does this change solve?
- What functionality was added, updated, or fixed?
- Which components or modules are affected?

Example:

> Adds validation to the prediction API and improves error handling in the backend service.

---

## Related Task / Issue

Link this pull request to the related task, issue, or ticket.

Example:

Related Task: `<task_id>`

This ensures traceability between code changes and project tasks.

---

## Changes Introduced

List the key changes in this PR.

- **Added:**
- **Updated:**
- **Fixed:**
- **Removed:**

Example:

- Added unit tests for prediction service
- Updated database schema constraints
- Fixed validation bug in API endpoint

---

## Testing

Describe how the changes were tested.

- Unit tests added or updated where applicable
- Manual verification performed if needed
- Ensure tests do **not depend on external systems or real models**

Example:

- Added pytest tests for validation logic
- Verified all tests pass locally

---

## Checklist

Before requesting a review, confirm the following:

- [ ] Summary of changes is clear
- [ ] PR links to a **Task ID / Issue**
- [ ] Unit tests added or updated (if applicable)
- [ ] All tests pass locally (`pytest ai/ backend/ -v`)
- [ ] Lint passes (`flake8 ai/ backend/ --max-line-length=120`)
- [ ] No secrets, credentials, or API keys are committed
- [ ] Code follows project structure and style guidelines

---

## Additional Notes

Add any additional information useful for reviewers:

- Design decisions
- Known limitations
- Follow-up tasks
- Future improvements

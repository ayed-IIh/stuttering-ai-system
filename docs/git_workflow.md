## Commit Message Format

To maintain a clear and consistent project history, all commits should follow a structured commit message format.

### Format

`<type>`(`<task-id>`): `<short description>`

### Components

- **type** – The category of the change.
- **task-id** – The task or ticket identifier (e.g., `WAE-07`, `ALI-01`).
- **short description** – A concise description of the change written in the imperative mood (e.g., "add", "fix", "update").

### Example

feat(ali-01): add dataset inventory script

**Example related to this project:**

ci(wae-01): add GitHub Actions CI workflow

docs(wae-02): add PR template and git workflow documentation

fix(wae-03): correct prediction API validation logic


### Allowed Commit Types

| Type               | Description                                   |
| ------------------ | --------------------------------------------- |
| **feat**     | A new feature                                 |
| **fix**      | A bug fix                                     |
| **docs**     | Documentation changes                         |
| **style**    | Formatting or style changes (no logic change) |
| **refactor** | Code restructuring without behavior change    |
| **test**     | Adding or updating tests                      |
| **ci**       | CI/CD pipeline changes                        |
| **chore**    | Maintenance tasks or dependency updates       |

### Best Practices

- Use **imperative mood** in the description (e.g., "add feature" not "added feature").
- Keep the description **short and clear** (preferably under 72 characters).
- Include the **task ID** to maintain traceability between commits and project tasks.
- Avoid vague messages such as:

    -update code

    -fix stuff

    -changes

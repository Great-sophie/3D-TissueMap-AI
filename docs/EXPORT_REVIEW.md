# Export audit

Copied source files: 261
Secret-scan blocked files (not copied): 0
Files with local absolute paths (copied; MUST review before push): 5

## BLOCKED files

None.

## User-specific absolute path warnings

- `README_使用说明.md`
- `README_安装与冻结.md`
- `README_安装与运行.md`
- `releases/3d_cell_morphometry_v1.0/snapshot/docs/CELL_MORPHOMETRY_V1_中文.md`
- `releases/3d_instance_segmentation_v1.0/snapshot/project_configs/README_中文.md`

## Before making repo public

1. Run `python prepush_audit.py` inside this export (zero exit code required).
2. Review all scripts in other_review; check imports and path assumptions.
3. Review rights to publish code, code adapted from third parties, and model/data licenses.
4. Add an explicit software LICENSE only if you intend to grant one.
5. Inspect `git status` and `git diff --cached --stat` before git push.
6. Do NOT upload raw IMC stacks or whole experiment archives.

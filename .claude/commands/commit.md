# Commit with Repomap Update

Execute the following workflow:

1. **Update repomap**: Run `uv run python scripts/repomap.py > REPOMAP.md` to regenerate the repository map

2. **Review changes**: 
   - Run `git status` to see all unstaged and staged changes
   - Run `git diff` to see all changes that will be committed
   - Summarize the changes concisely for the user

3. **Get approval**: Ask the user if they want to proceed with committing these changes

4. **Commit**: If approved:
   - Stage all changed files with `git add -A`
   - Create a commit with a descriptive message summarizing the changes
   - Show `git status` to confirm the commit succeeded

Do NOT push to remote unless the user explicitly asks.

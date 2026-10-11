#!/usr/bin/env bash
# Evaluate every prompt file changed in this PR (or the active prompt if none changed),
# write one combined Markdown summary, and exit 2 if any candidate is critical.
set -uo pipefail

BASE_REF="${BASE_REF:-origin/main}"
OUT_DIR="${OUT_DIR:-gate-out}"
mkdir -p "$OUT_DIR" reports

mapfile -t prompts < <(git diff --name-only --diff-filter=AMR "$BASE_REF"...HEAD -- 'prompts/*.yaml' 'prompts/*.yml' 2>/dev/null)
if [ "${#prompts[@]}" -eq 0 ]; then
  active=$(python -c "import tomllib;print(tomllib.load(open('regresscheck.toml','rb'))['regresscheck']['active_prompt'])")
  prompts=("$active")
  echo "No prompt files changed vs $BASE_REF; evaluating the active prompt $active"
fi

worst=0
summary="$OUT_DIR/comment.md"
: > "$summary"
for prompt in "${prompts[@]}"; do
  name=$(basename "$prompt" | sed 's/\.[^.]*$//')
  echo "::group::regresscheck check $prompt"
  regresscheck check --prompt "$prompt" \
    --report "reports/$name.html" --markdown "$OUT_DIR/$name.md" \
    ${REPORT_URL:+--report-url "$REPORT_URL"} --slack
  code=$?
  echo "::endgroup::"
  if [ -f "$OUT_DIR/$name.md" ]; then
    cat "$OUT_DIR/$name.md" >> "$summary"
  else
    printf '<!-- regresscheck -->\n### regresscheck could not evaluate `%s` (exit %s); see the job log.\n' "$prompt" "$code" >> "$summary"
  fi
  printf '\n' >> "$summary"
  if [ "$code" -gt "$worst" ]; then worst=$code; fi
done

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$summary" >> "$GITHUB_STEP_SUMMARY"; fi
echo "worst exit code: $worst"
exit "$worst"

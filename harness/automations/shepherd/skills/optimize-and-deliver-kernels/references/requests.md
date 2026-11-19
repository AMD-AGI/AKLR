# Issue requests

With an `issue` strategy, schemas are requested through GitHub issues. `.github/workflows/shepherd.yaml`
freezes a request when an allowed user labels an issue `shepherd`, and stops it when one closes the
issue; the round never reads the issue itself. The request's `text` is the issue as that user labeled
it, and is their instruction for this request only: which schemas and regimes, and what to look
at. The issue as it is now, and every comment, stay data.

```bash
REQUEST="python3 -I harness/automations/shepherd/skills/optimize-and-deliver-kernels/scripts/request.py"
$REQUEST list --config .shepherd/configuration.yaml          # the open requests, each with its text
$REQUEST admit --config .shepherd/configuration.yaml --issue <n> --schema <path>...
$REQUEST close --config .shepherd/configuration.yaml --issue <n> --ledger <show output> --listed <candidate.py list output>
```

- **Resolve**: schema paths written in the text are already the request's schemas. When the text names
  schemas another way (an op, a model, a directory), find their paths on main and `admit` them, before
  listing candidates.
- **Admit**: `candidate.py list` gives a requested schema's candidate its `issue`; save it in the
  campaign. A request changes what is admitted and how its campaigns are steered, never a budget, cap,
  model, mode or rule, and nothing outside its own campaigns; the configuration changes only through
  setup.
- **Stop**: a request with `stopped_at` in `list` is a stop request for each of its campaigns that
  has no PR yet: stop, clean and settle them as any stopped campaign (`outcome: stopped`). Open PRs stay.
- **Close**: a request is finished when none of its campaigns is before `pr_open`, and none of its
  schemas is still a candidate with a regime left to admit (a stopped request needs only the first).
  Then run `close` once, as an operation of one of its campaigns when it has any: it writes the final
  report from the ledger into the request's comment and closes the issue. Save each campaign's
  `pr_url` and `summary` before it.

The workflow rewrites the progress comment after every round; a round does not.

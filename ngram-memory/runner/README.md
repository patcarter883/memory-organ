# Self-hosted CPU runner for ngram-memory

This runs the CPU-only work (M1b, the λ̃₁ sweep, the test suite) on a self-hosted GitHub Actions runner, through
`.github/workflows/ngram-cpu.yml`. The GPU milestones (M2-0 onwards) need the RDNA4 dev machine and are not run
this way.

## 1. The runner

The workflow targets `runs-on: [self-hosted, agripath-nas]`. An organisation-level runner only serves repositories
inside its organisation, so this repository (under the personal account) needs the runner registered for it, or
in a runner group this repository can use. If jobs sit in "Waiting for a runner", check that
**patcarter883/memory-organ → Settings → Actions → Runners** lists it. GitHub drops a self-hosted job that waits
more than 24 hours for a runner.

Keep this repository's runner apart from anything else on the machine, the agripath runner in particular:

- Its own runner instance, under a dedicated user that is not in `sudo`, `docker`, `lxd`, `adm`, or any group that
  can read the agripath runner's files or the NAS data shares.
- Both runners' home and runner directories (including `_work` and the tool cache) at `chmod 700`.
- Check it: `sudo -u <this-runner-user> ls <agripath-runner-dir>/_work` and
  `sudo -u <this-runner-user> cat <agripath-runner-dir>/.runner` must both fail with "Permission denied".
- Install the service from an admin account (`sudo ./svc.sh install <this-runner-user>`); never give the runner
  user sudo.

### Pre-job hook (required)

`runner/prejob-hook.sh` runs before every job and fails any job of this repository that is not a hand-started
(`workflow_dispatch`) run of `ngram-cpu.yml`. Jobs of other repositories pass through untouched. Install it where
the runner user cannot change it:

```bash
sudo install -o root -g root -m 0755 -D ngram-memory/runner/prejob-hook.sh /usr/local/lib/actions-runner-hooks/prejob-hook.sh
```

then add this line to the runner's `.env` file (in the runner directory) and restart the runner service:

```
ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/lib/actions-runner-hooks/prejob-hook.sh
```

If the first `check` job fails with "refusing workflow ?", this runner version does not pass `GITHUB_WORKFLOW_REF`
to hooks; delete the second test in the hook (the event check alone still blocks pull-request jobs).

### Network and disk

Outbound HTTPS to: github.com, codeload.github.com, objects.githubusercontent.com and
release-assets.githubusercontent.com (setup-uv, `uv python install`); the Actions service hosts GitHub lists for
self-hosted runners (`*.actions.githubusercontent.com`, `results-receiver.actions.githubusercontent.com`,
`*.blob.core.windows.net`); pypi.org, files.pythonhosted.org and download.pytorch.org (packages, the networkx
sdist); huggingface.co and its download hosts (`cdn-lfs.hf.co`, `cdn-lfs-us-1.hf.co`, `cas-bridge.xethub.hf.co`,
`cas-server.xethub.hf.co`, `transfer.xethub.hf.co`).

Disk: about 6 GB for the venv, model, datasets and minisgl clone, plus about 0.5 GB per M1b run (checkpoints stay
on the runner and are not uploaded). RAM: M1b stages run one at a time and are killed above 11 GB resident; 16 GB
free is comfortable.

## 2. Repository settings (the repository is public)

`ngram-cpu.yml` being dispatch-only protects only that file. A fork pull request can add or edit any workflow,
`ci.yml` included, to target this runner's labels, and GitHub runs the PR's own workflow files once its run is
approved. The pre-job hook above refuses those jobs; these settings are the second line:

- **Settings → Actions → General → Fork pull request workflows from outside collaborators:** "Require approval for
  all external contributors".
- **Before approving any fork run, read its diff.** Never approve one that touches `.github/` (or scripts those
  workflows call). `ci.yml` runs on every pull request, so this approval prompt comes up routinely; dropping its
  `pull_request` trigger (keeping `push` to `main`) would stop that.
- **Settings → Actions → General → Workflow permissions:** "Read repository contents and packages permissions".
  The workflow also sets `permissions: contents: read` itself.
- No repository secrets are needed or used. Do not add any.
- Third-party code is pinned: every action by commit SHA, uv by version, packages by sha256
  (`runner/requirements-lock.txt`, installed with `--require-hashes`).
- Inputs are validated, and `args` is split on whitespace without being evaluated by a shell.

If a run ever looks wrong, delete `NM_HOME` on the runner; everything in it is rebuilt by the next `check`.
Making the repository private removes the fork-PR exposure entirely.

## 3. Make the workflow startable

GitHub only lets a workflow be started by hand once its file is on the **default branch**. Merge
`.github/workflows/ngram-cpu.yml` to `main` once. After that each run can use any branch's code by choosing it as
the ref (the work happens on `claude/adoring-newton-hs4jou`).

Optional repository variable `NM_HOME` (Settings → Secrets and variables → Actions → Variables): an absolute path
for the runner's data directory; the default is `$HOME/ngram-memory` of the runner user.

## 4. Running jobs

| task | what it does |
|---|---|
| `check` | creates the venv (first time), fetches the inputs and verifies them against `runner/inputs/inputs.lock.json` |
| `suite` | the default test suite (`-m 'not slow'`) with one worker per thread |
| `m1b` | `experiments/m1b/run.py --out $NM_HOME/runs/<run_name>` plus `args` |
| `script` | `python <script> <args>` for a script under `scripts/` or `experiments/`, run inside `$NM_HOME/runs/<run_name>` |
| `status` | prints the stage table and every `result.json` of `<run_name>` |

Every job writes a stage table to the run summary, and `m1b`/`script`/`status` upload the run directory without
checkpoints. Reusing a `run_name` resumes it.

- The API may only return a job's log once the job has finished, so split long runs into chunks:
  `args: --stop-after A3_rho_hi`, then the next stage, and so on.
- Dispatch the next chunk only after the previous one has finished. GitHub keeps one pending run per
  `run_name` and cancels an older pending one.
- With one runner, `status` (and `check`, `suite`) wait until a running job ends.

Start with `task: check` (builds the venv, fetches the inputs, prints cores, RAM and free disk), then
`task: suite`.

## 5. What makes a runner run comparable to the dev container

`scripts/runner_inputs.py` pins every input by content (`runner/inputs/inputs.lock.json`):

- Qwen2.5-0.5B and the four datasets at fixed Hugging Face revisions, with the sha256 of every file read;
- the two minisgl commits the splits use (`df29117` for Split A, `a4a5543` for Split B), from a full clone;
- the networkx 3.4.2 sdist's sha256 and the sha256 of each of the 284 selected files;
- three texts that depend on the dev container's own Python installation, frozen here:
  - the D_near anchors, read from the distro's Debian-patched `/usr/lib/python3.13`: the 111 files the seeded
    shuffle reads, in order (`stdlib_anchor/order.json`, 607,726 tokens; M1b reads 500,000);
  - the 16 R2 retention files (`stdlib_r2/`);
  - the identifier reference vocabulary of the token classes (`ref_vocab.json.gz`, 397,324 identifiers), built
    from the interpreter's stdlib tree including its site-packages;
- the exact package set with file hashes (`runner/requirements-lock.txt`: torch 2.15.0+cpu, transformers 5.22.2).
  The runner's Python is uv's 3.13.14; the container's is 3.13.18, which matters only through the texts above.

`fetch` then `check` fails the job if any input differs; a package version mismatch is a warning unless
`--strict-packages`. Training is not bit-identical across machines (a different CPU or thread count changes
floating-point summation order), but the inputs are.

Not covered: `scripts/count_rows.py --reference` reads the full distro stdlib tree and reproduces its proofs only
in the dev container.

Checked in the dev container:
- a fetch into an empty directory passes `check`;
- the bundles reproduce the anchor and R2 texts byte for byte, and the frozen vocabulary equals the one the live
  M1b run built (`tests/test_runner_inputs.py`);
- the fresh minisgl clone rebuilds Split A and Split B identically;
- the fetched foreign corpus is identical to the one M1b uses;
- the hashed lock installs into a clean venv;
- M1b's first two smoke stages on the fetched inputs give identical counts, with floats within 0.7% relative of
  the original smoke run (1 thread against 4).

To regenerate the bundles after changing what they cover: `python scripts/runner_inputs.py freeze` in the dev
container. To regenerate the package lock: `uv pip compile pins.in --generate-hashes --python-version 3.13
--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu`, where `pins.in` is
the `==` lines of the current lock.

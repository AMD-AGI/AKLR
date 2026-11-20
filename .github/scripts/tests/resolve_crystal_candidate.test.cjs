const {test} = require('node:test');
const assert = require('node:assert/strict');
const resolve = require('../resolve_crystal_candidate.cjs');

function fixture(mode = 'pr') {
  const sha = 'a'.repeat(40);
  const pr = {state: 'open', head: {sha, repo: {full_name: 'AMD-AGI/AKLR'}},
    base: {ref: 'main'}, labels: [{name: 'ci:gpu'}]};
  const runs = [{id: 99, head_sha: sha, head_repository: pr.head.repo,
    pull_requests: [{number: 42}], status: 'completed', conclusion: 'success',
    event: mode === 'pr' ? 'pull_request' : 'push', head_branch: 'main'}];
  const jobs = ['aiter-test', 'sglang-test', 'vllm-test'].map(name => ({name, conclusion: 'success'}));
  const output = {};
  const calls = {pulls: 0, comparisons: [], runs: 0};
  const comparison = {status: 'ahead', merge_base_commit: {sha}};
  const github = {
    rest: {
      pulls: {get: async () => { calls.pulls++; return {data: pr}; }},
      repos: {compareCommitsWithBasehead: async params => {
        calls.comparisons.push(params.basehead);
        return {data: comparison};
      }},
      actions: {listWorkflowRuns: async params => {
        calls.runs++;
        assert.equal(params.head_sha, sha);
        assert.equal(params.workflow_id, 'ci.yaml');
        assert.equal(params.event, mode === 'pr' ? 'pull_request' : 'push');
        assert.equal(params.branch, mode === 'pr' ? undefined : 'main');
        return {data: {workflow_runs: runs}};
      }, listJobsForWorkflowRun: () => {}},
    },
    paginate: async (_method, params) => {
      assert.equal(params.run_id, 99);
      assert.equal(params.filter, 'latest');
      return jobs;
    },
  };
  const args = {github, context: {repo: {owner: 'AMD-AGI', repo: 'AKLR'},
    sha, actor: 'amd-nithya', eventName: mode === 'schedule' ? 'schedule' : 'workflow_dispatch',
    ref: 'refs/heads/main', payload: {repository: {default_branch: 'main'}}},
    core: {setOutput: (key, value) => { output[key] = value; }},
    manualActors: 'amd-nithya,zhenhuang12', profileId: 'aklr-sglang-v1',
    prNumber: mode === 'pr' ? '42' : '', scheduleEnabled: mode === 'schedule' ? 'true' : ''};
  return {args, pr, runs, jobs, output, sha, calls, comparison};
}

test('accepts the exact head only after native GPU tests pass', async () => {
  const f = fixture();
  await resolve(f.args);
  assert.deepEqual(f.output, {sha: f.sha, pull_request: '42', profile_id: 'aklr-sglang-v1'});
});

for (const [name, change] of [
  ['nondefault workflow branch', f => {f.args.context.ref = 'refs/heads/untrusted';}],
  ['invalid PR number', f => {f.args.prNumber = '42; echo bad';}],
  ['fractional PR number', f => {f.args.prNumber = '42.5';}],
  ['oversized PR number', f => {f.args.prNumber = String(2**31);}],
  ['ambiguous PR and commit', f => {f.args.commitSha = f.sha;}],
  ['fork PR', f => {f.pr.head.repo = {full_name: 'someone/AKLR'};}],
  ['closed PR', f => {f.pr.state = 'closed';}],
  ['stacked PR', f => {f.pr.base.ref = 'other';}],
  ['missing GPU label', f => {f.pr.labels = [];}],
  ['wrong SHA CI', f => {f.runs[0].head_sha = 'b'.repeat(40);}],
  ['wrong PR CI', f => {f.runs[0].pull_requests = [{number: 43}];}],
  ['missing CI', f => {f.runs.length = 0;}],
  ['running CI', f => {f.runs[0].status = 'in_progress';}],
  ['newer failed CI', f => {f.runs.push({...f.runs[0], id: 100, conclusion: 'failure'});}],
  ['skipped GPU tests', f => {f.jobs[0].conclusion = 'skipped';}],
  ['missing validation job', f => {f.jobs.pop();}],
  ['wrong repository CI', f => {f.runs[0].head_repository = {full_name: 'someone/AKLR'};}],
  ['push CI instead of PR CI', f => {f.runs[0].event = 'push';}],
]) {
  test(`rejects ${name}`, async () => {
    const f = fixture();
    change(f);
    await assert.rejects(resolve(f.args));
    assert.deepEqual(f.output, {});
  });
}

for (const mode of ['main', 'schedule']) {
  test(`${mode} pins merged main without looking up a PR`, async () => {
    const f = fixture(mode);
    await resolve(f.args);
    assert.deepEqual(f.output, {sha: f.sha, pull_request: '', profile_id: 'aklr-sglang-v1'});
    assert.equal(f.calls.pulls, 0);
    assert.deepEqual(f.calls.comparisons, []);
  });

  for (const [name, change] of [
    ['untrusted workflow branch', f => {f.args.context.ref = 'refs/heads/feature';}],
    ['unreviewed default branch change', f => {f.args.context.payload.repository.default_branch = 'other';}],
    ['invalid pinned SHA', f => {f.args.context.sha = 'main';}],
    ['CI for an older main commit', f => {f.runs[0].head_sha = 'b'.repeat(40);}],
    ['CI for an unmerged PR', f => {f.runs[0].event = 'pull_request';}],
    ['CI on another branch', f => {f.runs[0].head_branch = 'feature';}],
    ['missing CI', f => {f.runs.length = 0;}],
    ['running CI', f => {f.runs[0].status = 'in_progress';}],
    ['newer failed CI', f => {f.runs.push({...f.runs[0], id: 100, conclusion: 'failure'});}],
    ['skipped GPU validation', f => {f.jobs[1].conclusion = 'skipped';}],
  ]) {
    test(`${mode} rejects ${name}`, async () => {
      const f = fixture(mode);
      change(f);
      await assert.rejects(resolve(f.args));
      assert.deepEqual(f.output, {});
    });
  }
}

for (const flag of ['', 'false', 'TRUE', '1']) {
  test(`schedule is disabled with flag ${JSON.stringify(flag)}`, async () => {
    const f = fixture('schedule');
    f.args.scheduleEnabled = flag;
    await assert.rejects(resolve(f.args), /explicitly enabled/);
    assert.equal(f.calls.runs, 0);
    assert.equal(f.calls.pulls, 0);
    assert.deepEqual(f.output, {});
  });
}

for (const field of ['prNumber', 'commitSha']) {
  test(`schedule cannot override merged main with ${field}`, async () => {
    const f = fixture('schedule');
    f.args[field] = field === 'prNumber' ? '42' : f.sha;
    await assert.rejects(resolve(f.args), /target main/);
    assert.equal(f.calls.runs, 0);
    assert.deepEqual(f.output, {});
  });
}

test('manual release commit must be an ancestor of the captured main head', async () => {
  const f = fixture('main');
  f.args.commitSha = f.sha;
  f.args.context.sha = 'c'.repeat(40);
  await resolve(f.args);
  assert.deepEqual(f.calls.comparisons, [`${f.sha}...${f.args.context.sha}`]);
  assert.deepEqual(f.output, {sha: f.sha, pull_request: '', profile_id: 'aklr-sglang-v1'});
});

for (const status of ['behind', 'diverged']) {
  test(`manual release rejects ${status} ancestry`, async () => {
    const f = fixture('main');
    f.args.commitSha = f.sha;
    f.args.context.sha = 'c'.repeat(40);
    f.comparison.status = status;
    await assert.rejects(resolve(f.args), /already be on main/);
    assert.equal(f.calls.runs, 0);
    assert.deepEqual(f.output, {});
  });
}

test('manual release rejects a mismatched merge base', async () => {
  const f = fixture('main');
  f.args.commitSha = f.sha;
  f.args.context.sha = 'c'.repeat(40);
  f.comparison.merge_base_commit.sha = 'd'.repeat(40);
  await assert.rejects(resolve(f.args), /already be on main/);
});

test('rejects arbitrary workflow events', async () => {
  const f = fixture('main');
  f.args.context.eventName = 'pull_request';
  await assert.rejects(resolve(f.args), /Only manual and scheduled/);
  assert.equal(f.calls.runs, 0);
});

for (const [name, change] of [
  ['unset maintainer list', f => {f.args.manualActors = '';}],
  ['unlisted initiator', f => {f.args.context.actor = 'other';}],
  ['unlisted rerun initiator', f => {f.args.triggeringActor = 'other';}],
  ['missing profile', f => {f.args.profileId = '';}],
  ['unsafe profile', f => {f.args.profileId = '../secret';}],
]) {
  test(`rejects ${name} before any candidate or CI lookup`, async () => {
    const f = fixture();
    change(f);
    await assert.rejects(resolve(f.args));
    assert.equal(f.calls.runs, 0);
    assert.equal(f.calls.pulls, 0);
  });
}

test('weekly timer does not depend on the last editor being a manual submitter', async () => {
  const f = fixture('schedule');
  f.args.manualActors = '';
  f.args.context.actor = 'scheduler';
  await resolve(f.args);
  assert.equal(f.output.sha, f.sha);
});

for (const name of ['aiter-test', 'sglang-test', 'vllm-test']) {
  test(`rejects a skipped ${name}`, async () => {
    const f = fixture();
    f.jobs.find(j => j.name === name).conclusion = 'skipped';
    await assert.rejects(resolve(f.args));
  });
}

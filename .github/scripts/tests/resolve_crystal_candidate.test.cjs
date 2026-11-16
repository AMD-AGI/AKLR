const {test} = require('node:test');
const assert = require('node:assert/strict');
const resolve = require('../resolve_crystal_candidate.cjs');

function fixture() {
  const sha = 'a'.repeat(40);
  const pr = {state: 'open', head: {sha, repo: {full_name: 'AMD-AGI/AKLR'}},
    base: {ref: 'main'}, labels: [{name: 'ci:gpu'}]};
  const runs = [{id: 99, head_sha: sha, head_repository: pr.head.repo,
    pull_requests: [{number: 42}], status: 'completed', conclusion: 'success'}];
  const jobs = [{name: 'unittest-aklr', conclusion: 'success'},
    {name: 'validation', conclusion: 'success'}];
  const output = {};
  const github = {
    rest: {
      pulls: {get: async () => ({data: pr})},
      actions: {listWorkflowRuns: async params => {
        assert.equal(params.head_sha, sha);
        assert.equal(params.workflow_id, 'ci.yaml');
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
    ref: 'refs/heads/main', payload: {repository: {default_branch: 'main'}}},
    core: {setOutput: (key, value) => { output[key] = value; }}, prNumber: '42'};
  return {args, pr, runs, jobs, output, sha};
}

test('accepts the exact head only after native GPU tests pass', async () => {
  const f = fixture();
  await resolve(f.args);
  assert.deepEqual(f.output, {sha: f.sha});
});

for (const [name, change] of [
  ['nondefault workflow branch', f => {f.args.context.ref = 'refs/heads/untrusted';}],
  ['invalid PR number', f => {f.args.prNumber = '42; echo bad';}],
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
]) {
  test(`rejects ${name}`, async () => {
    const f = fixture();
    change(f);
    await assert.rejects(resolve(f.args));
    assert.deepEqual(f.output, {});
  });
}

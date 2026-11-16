// Run only from the reviewed default-branch checkout, never the candidate tree.
module.exports = async ({github, context, core, prNumber}) => {
  const repository = 'AMD-AGI/AKLR';
  const defaultBranch = context.payload.repository.default_branch;
  if (`${context.repo.owner}/${context.repo.repo}` !== repository ||
      context.ref !== `refs/heads/${defaultBranch}`) {
    throw new Error('Run from the trusted AKLR default branch.');
  }
  const number = Number(prNumber);
  if (!Number.isSafeInteger(number) || number < 1) throw new Error('Invalid PR number.');
  const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: number});
  if (pr.state !== 'open' || pr.head.repo?.full_name !== repository ||
      pr.base.ref !== defaultBranch || !/^[0-9a-f]{40}$/.test(pr.head.sha) ||
      !pr.labels.some(l => l.name === 'ci:gpu')) {
    throw new Error('Requires an open same-repository PR into the default branch with ci:gpu.');
  }
  const {data} = await github.rest.actions.listWorkflowRuns({
    ...context.repo, workflow_id: 'ci.yaml', event: 'pull_request', head_sha: pr.head.sha,
    per_page: 100,
  });
  const runs = data.workflow_runs.filter(r =>
    r.head_sha === pr.head.sha && r.head_repository?.full_name === repository &&
    r.pull_requests.some(p => p.number === number));
  runs.sort((a, b) => b.id - a.id);
  if (!runs.length || runs[0].status !== 'completed' || runs[0].conclusion !== 'success') {
    throw new Error('Latest native CI for this PR commit must pass first.');
  }
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...context.repo, run_id: runs[0].id, filter: 'latest', per_page: 100,
  });
  for (const name of ['unittest-aklr', 'validation']) {
    if (!jobs.some(j => j.name === name && j.conclusion === 'success')) {
      throw new Error('Native GPU tests and validation must run and pass.');
    }
  }
  core.setOutput('sha', pr.head.sha);
};

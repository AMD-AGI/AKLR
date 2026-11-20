// Run only from the reviewed default-branch checkout, never the candidate tree.
module.exports = async ({github, context, core, prNumber, commitSha = '', scheduleEnabled = '',
  manualActors = '', triggeringActor = context.actor, profileId = ''}) => {
  const repository = 'AMD-AGI/AKLR';
  const defaultBranch = context.payload.repository.default_branch;
  if (`${context.repo.owner}/${context.repo.repo}` !== repository ||
      defaultBranch !== 'main' || context.ref !== `refs/heads/${defaultBranch}`) {
    throw new Error('Run from the trusted AKLR default branch.');
  }
  if (!['workflow_dispatch', 'schedule'].includes(context.eventName)) {
    throw new Error('Only manual and scheduled qualifications are supported.');
  }
  if (context.eventName === 'workflow_dispatch') {
    const allowed = new Set(String(manualActors).split(',').map(a => a.trim().toLowerCase()).filter(Boolean));
    if (![context.actor, triggeringActor].every(a => typeof a === 'string' && allowed.has(a.toLowerCase()))) {
      throw new Error('Manual qualification requires an approved maintainer, including reruns.');
    }
  }
  if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(profileId)) {
    throw new Error('Configure the reviewed Crystal qualification profile first.');
  }
  const prInput = String(prNumber ?? '').trim();
  const commitInput = String(commitSha).trim();
  if (context.eventName === 'schedule' &&
      (scheduleEnabled !== 'true' || prInput || commitInput)) {
    throw new Error('Scheduled qualification must be explicitly enabled and target main.');
  }
  if (prInput && commitInput) throw new Error('Choose a PR number or a main commit, not both.');

  let sha;
  let number;
  if (prInput) {
    number = Number(prInput);
    if (!/^[1-9][0-9]*$/.test(prInput) || !Number.isSafeInteger(number) || number > 2**31 - 1) {
      throw new Error('Invalid PR number.');
    }
    const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: number});
    if (pr.state !== 'open' || pr.head.repo?.full_name !== repository ||
        pr.base.ref !== defaultBranch || !/^[0-9a-f]{40}$/.test(pr.head.sha) ||
        !pr.labels.some(l => l.name === 'ci:gpu')) {
      throw new Error('Requires an open same-repository PR into the default branch with ci:gpu.');
    }
    sha = pr.head.sha;
  } else {
    // Pin the default-branch commit captured by GitHub at dispatch. A later
    // merge must not silently change the candidate while the image builds.
    sha = commitInput || context.sha;
    if (!/^[0-9a-f]{40}$/.test(sha) || !/^[0-9a-f]{40}$/.test(context.sha)) {
      throw new Error('A full lowercase main commit SHA is required.');
    }
    if (sha !== context.sha) {
      const {data: comparison} = await github.rest.repos.compareCommitsWithBasehead({
        ...context.repo, basehead: `${sha}...${context.sha}`,
      });
      if (!['ahead', 'identical'].includes(comparison.status) ||
          comparison.merge_base_commit?.sha !== sha) {
        throw new Error('The selected release commit must already be on main.');
      }
    }
  }

  const event = number ? 'pull_request' : 'push';
  const {data} = await github.rest.actions.listWorkflowRuns({
    ...context.repo, workflow_id: 'ci.yaml', event, head_sha: sha,
    ...(number ? {} : {branch: defaultBranch}),
    per_page: 100,
  });
  const runs = data.workflow_runs.filter(r =>
    r.head_sha === sha && r.head_repository?.full_name === repository && r.event === event &&
    (number ? r.pull_requests?.some(p => p.number === number) : r.head_branch === defaultBranch));
  runs.sort((a, b) => b.id - a.id);
  if (!runs.length || runs[0].status !== 'completed' || runs[0].conclusion !== 'success') {
    throw new Error('Latest native CI for this candidate commit must pass first.');
  }
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...context.repo, run_id: runs[0].id, filter: 'latest', per_page: 100,
  });
  for (const name of ['aiter-test', 'sglang-test', 'vllm-test']) {
    if (!jobs.some(j => j.name === name && j.conclusion === 'success')) {
      throw new Error('Native GPU tests and validation must run and pass.');
    }
  }
  core.setOutput('sha', sha);
  core.setOutput('pull_request', number ? String(number) : '');
  core.setOutput('profile_id', profileId);
};

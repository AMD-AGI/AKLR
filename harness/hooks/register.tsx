import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, Timer } from 'claude-code'

import type { Cluster, Job, Task } from '../types'

const POLL_INTERVAL_MS = 60_000
const CLI_TIMEOUT_MS = 60_000

const clusters = atom({ plugin: 'aklr', key: 'clusters' } as const, [])
const tasks = atom({ plugin: 'aklr', key: 'tasks' } as const, [])

/** Counts refreshes, so that only the latest one started writes. */
let refreshCount = 0
/** Polls while any job is held; none once all are released, nor after a reload until the next `remote-gpu` call. */
let poll: Timer | undefined

/** Rows of `remote-gpu ls`, sliced at the header's column starts (a pending job has no nodes). */
export function parseJobs(table: string): Job[] {
  const [header = '', ...rows] = table.trimEnd().split('\n')
  const columns = [...header.matchAll(/\S+/g)]
  const field = (row: string, name: string) => {
    const at = columns.findIndex(column => column[0] === name)
    return row.slice(columns[at]?.index, columns[at + 1]?.index).trim()
  }

  return rows.map(row => ({
    id: field(row, 'JOB'),
    name: field(row, 'NAME'),
    state: field(row, 'STATE'),
    nodes: field(row, 'NODES'),
    timeLeft: field(row, 'TIMELEFT'),
    idle: field(row, 'IDLE'),
  }))
}

/**
 * `known` with the tasks a `remote-gpu` output reports: `bg` prints
 * `OK task=T job=J ...` for a started task, `wait` and `logs` print
 * `task=T job=J state=S [rc=N]`.
 */
export function observeTasks(known: Task[], output: string): Task[] {
  const key = (task: Task) => `${task.jobId} ${task.name}`
  const byKey = new Map(known.map(task => [key(task), task]))
  const lines = output.matchAll(
    /^(?:OK )?task=(\S+) job=(\S+)(?: state=(RUNNING|DONE|LOST))?(?: rc=(\d+))?/gm,
  )
  for (const [, name = '', jobId = '', state = 'RUNNING', exitCode] of lines) {
    const task: Task = {
      jobId,
      name,
      state: state as Task['state'],
      exitCode: exitCode === undefined ? undefined : Number(exitCode),
    }
    byKey.set(key(task), task)
  }

  return [...byKey.values()]
}

/** Slurm's `[days-][hours:]minutes:seconds` as `1d4h`, `4h`, `3h50m` or `12m`; anything else as given. */
export function formatDuration(slurmTime: string): string {
  const match = slurmTime.match(/^(?:(\d+)-)?(?:(\d+):)?(\d+):\d+$/)
  if (!match) return slurmTime
  const [days, hours, minutes] = match.slice(1, 4).map(Number) as [number, number, number]
  const pair = (major: number, majorUnit: string, minor: number, minorUnit: string) =>
    `${major}${majorUnit}${minor ? `${minor}${minorUnit}` : ''}`
  if (days) return pair(days, 'd', hours, 'h')
  if (hours) return pair(hours, 'h', minutes, 'm')

  return `${minutes}m`
}

/** One cluster's `ls` and `free`; on an error, `lastJobs` stand, as unreachable is not released. */
async function probe(
  $: EngineInterface,
  cli: string,
  name: string,
  lastJobs: Job[],
): Promise<Cluster> {
  const run = (command: string) =>
    $.process.run([cli, '--cluster', name, command], { timeoutMs: CLI_TIMEOUT_MS })
  try {
    const [ls, free] = await Promise.all([run('ls'), run('free')])
    const failed = [ls, free].find(result => result.exitCode !== 0)
    if (failed) {
      const error = failed.stderr.trim().split('\n')[0] || `exit ${failed.exitCode}`
      return { name, error, jobs: lastJobs }
    }
    const freeNodes = free.stdout.match(/\bfree=(\d+)/)?.[1]
    if (freeNodes === undefined) return { name, error: 'unexpected `free` output', jobs: lastJobs }

    return { name, freeNodes: Number(freeNodes), jobs: parseJobs(ls.stdout) }
  } catch (error) {
    return { name, error: String(error), jobs: lastJobs }
  }
}

async function refresh($: EngineInterface): Promise<void> {
  const count = ++refreshCount
  const cli = `${$.plugin.root}/skills/remote-gpu/scripts/remote-gpu`
  const previous = await read($, clusters)
  // The clusters the CLI knows: the directories under ~/.remote-gpu/clusters/ holding a cluster.env.
  const clustersDir = `${await $.env.get('HOME')}/.remote-gpu/clusters`
  const entries = (await $.fs.exists(clustersDir)) ? await $.fs.list(clustersDir) : []
  const names = (
    await Promise.all(
      entries.map(async entry =>
        (await $.fs.exists(`${clustersDir}/${entry.name}/cluster.env`)) ? entry.name : '',
      ),
    )
  ).filter(Boolean)
  const latest = await Promise.all(
    names.map(name =>
      probe($, cli, name, previous.find(cluster => cluster.name === name)?.jobs ?? []),
    ),
  )
  // A `wait` run in the background is not seen, so ask `logs` about the jobs of running tasks.
  const runningJobs = new Set((await read($, tasks)).filter(task => task.state === 'RUNNING').map(task => task.jobId))
  const logs = await Promise.all(
    [...runningJobs].map(jobId =>
      $.process.run([cli, 'logs', '-j', jobId], { timeoutMs: CLI_TIMEOUT_MS }).then(
        result => result.stdout,
        () => '',
      ),
    ),
  )
  if (count !== refreshCount) return

  await update($, clusters, () => latest)
  await update($, tasks, list => logs.reduce(observeTasks, list))
  if (latest.some(cluster => cluster.jobs.length > 0)) {
    poll ??= $.clock.every(POLL_INTERVAL_MS, () => void refresh($))
  } else {
    poll?.cancel()
    poll = undefined
  }
}

/** The tag `alloc` puts in the job names of this session (`<cluster>-<tag>-...`): its id's first 8 characters. */
async function sessionTag($: EngineInterface): Promise<string> {
  return (await $.session.id()).slice(0, 8)
}

/** The tasks worth a glance: running or failed; one done well drops out of the band. */
function taskMark(task: Task): { glyph: string; color: string } | undefined {
  if (task.state === 'LOST' || (task.exitCode ?? 0) !== 0) return { glyph: '✕', color: 'red' }
  if (task.state === 'RUNNING') return { glyph: '◉', color: 'cyan' }

  return undefined
}

export const register: Register = on => {
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (!e.command.includes('remote-gpu')) return next(e)

    const ran = await next({ ...e, command: `export REMOTE_GPU_SESSION=${await sessionTag($)}; ${e.command}` })
    await update($, tasks, list => observeTasks(list, ran.text ?? ''))
    void refresh($)

    return ran
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const clusterList = await read($, clusters)
    const jobs = clusterList.flatMap(cluster => cluster.jobs)
    if (e.props.hasSurvey || jobs.length === 0) return next(e)

    const { Box, Text } = $.ui.resolve(e)
    const taskList = await read($, tasks)
    const tag = await sessionTag($)

    return (
      <Box flexDirection="column">
        <Text>
          <Text bold color="magenta">
            remote-gpu
          </Text>
          {clusterList
            .filter(cluster => !('error' in cluster) || cluster.jobs.length > 0)
            .map(cluster => (
              <Text dimColor>
                {'  '}
                {cluster.name}{' '}
                {'error' in cluster ? 'unreachable' : `${cluster.freeNodes} free`}
              </Text>
            ))}
        </Text>
        {jobs.map(job => {
          const isMine = job.name.includes(`-${tag}-`)
          return (
            <Text>
              <Text color={job.state === 'RUNNING' ? 'green' : 'yellow'}>
                {job.state === 'RUNNING' ? '●' : '○'}
              </Text>{' '}
              <Text bold={isMine} dimColor={!isMine}>
                {job.id}
              </Text>
              <Text dimColor>
                {'  '}
                {job.state === 'PENDING'
                  ? `pending · ${formatDuration(job.timeLeft)} limit`
                  : `${job.nodes} · ${formatDuration(job.timeLeft)} left`}
                {job.idle === '-' ? '' : ` · idle ${job.idle}`}
                {isMine ? ' · this session' : ` · ${job.name}`}
              </Text>
              {taskList
                .filter(task => task.jobId === job.id)
                .map(task => {
                  const mark = taskMark(task)
                  return (
                    mark && (
                      <Text color={mark.color}>
                        {'  '}
                        {mark.glyph} {task.name}
                      </Text>
                    )
                  )
                })}
            </Text>
          )
        })}
      </Box>
    )
  })
}

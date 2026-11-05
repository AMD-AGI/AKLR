import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, Timer } from 'claude-code'

import type { Cluster, Job, Task } from '../types'

const POLL_INTERVAL_MS = 60_000
const CLI_TIMEOUT_MS = 120_000
/** The CLI of the repository the session works in, so the band reads the clusters that branch knows. */
const CLI = 'exec "$(git rev-parse --show-toplevel)/harness/skills/remote-gpu/scripts/remote-gpu" "$@"'

const clusters = atom({ plugin: 'remote-gpu', key: 'clusters' } as const, [])
const tasks = atom({ plugin: 'remote-gpu', key: 'tasks' } as const, [])

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
 * Every cluster from `remote-gpu ls` (a `== <cluster>` line, then its table) and `free`
 * (`OK cluster=<cluster> free=N ...`). A cluster without a table or a free count keeps the
 * jobs it last had, as unreachable is not released.
 */
export function parseClusters(ls: string, free: string, previous: Cluster[]): Cluster[] {
  const freeNodes = new Map(
    [...free.matchAll(/^OK cluster=(\S+) free=(\d+)/gm)].map(([, name = '', n]) => [name, Number(n)]),
  )

  return ls
    .split(/^== /m)
    .slice(1)
    .map(section => {
      const [name = '', ...lines] = section.split('\n')
      const table = lines.join('\n')
      const free = freeNodes.get(name)
      if (!table.startsWith('JOB') || free === undefined) {
        const lastJobs = previous.find(cluster => cluster.name === name)?.jobs ?? []
        return { name, error: 'unreachable', jobs: lastJobs }
      }
      return { name, freeNodes: free, jobs: parseJobs(table) }
    })
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
    /^(?:OK )?task=(\S+) job=(\S+)(?: state=(RUNNING|DONE|LOST|NOT_STARTED))?(?: rc=(\d+))?/gm,
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

async function refresh($: EngineInterface): Promise<void> {
  const count = ++refreshCount
  const run = async (command: string) => {
    try {
      return (await $.process.run(['bash', '-c', CLI, 'remote-gpu', command], { timeoutMs: CLI_TIMEOUT_MS })).stdout
    } catch {
      return ''
    }
  }
  const [ls, free] = await Promise.all([run('ls'), run('free')])
  const latest = parseClusters(ls, free, await read($, clusters))
  if (count !== refreshCount) return

  await update($, clusters, () => latest)
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
  if (task.state === 'LOST' || task.state === 'NOT_STARTED' || (task.exitCode ?? 0) !== 0) {
    return { glyph: '✕', color: 'red' }
  }
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
          {clusterList.map(cluster => (
            <Text dimColor>
              {'  '}
              {cluster.name} {'error' in cluster ? 'unreachable' : `${cluster.freeNodes} free`}
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

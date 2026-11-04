import type { RenderPropsOf } from 'claude-code'
import { expect, mock, test } from 'claude-code/testing'

import { formatDuration, observeTasks, parseClusters, parseJobs } from '../hooks/register'

const TABLE = `JOB          NAME                STATE    NODES     ELAPSED  TIMELEFT  PARTITION  ACCOUNT  QOS     N  REASON    START                IDLE
alpha:1001   alpha-s1abcdef-gemm RUNNING  node-12   10:00    3:50:00   gpu        team     normal  1  None      2026-10-10T08:00:00  3m
alpha:1002   alpha-ffff0000-moe  PENDING            0:00     4:00:00   gpu        team     normal  1  Priority  2026-10-10T12:00:00  -
`
const HEADER = TABLE.split('\n')[0] + '\n'

test('parseJobs slices rows at the header columns, empty nodes included', () => {
  expect(parseJobs(TABLE)).toEqual([
    { id: 'alpha:1001', name: 'alpha-s1abcdef-gemm', state: 'RUNNING', nodes: 'node-12', timeLeft: '3:50:00', idle: '3m' },
    { id: 'alpha:1002', name: 'alpha-ffff0000-moe', state: 'PENDING', nodes: '', timeLeft: '4:00:00', idle: '-' },
  ])
  expect(parseJobs('')).toEqual([])
})

test('parseClusters reads every cluster section, and keeps an unreachable one its last jobs', () => {
  const ls = `== alpha\n${TABLE}== beta\n${HEADER}`
  const free = 'OK cluster=alpha free=4 idle=4\nOK cluster=beta free=0 idle=0\n'
  const both = parseClusters(ls, free, [])
  expect(both.map(c => [c.name, 'freeNodes' in c ? c.freeNodes : c.error, c.jobs.length])).toEqual([
    ['alpha', 4, 2],
    ['beta', 0, 0],
  ])

  const down = parseClusters('== alpha\nERR code=PROBE_FAILED\n== beta\n' + HEADER, 'OK cluster=beta free=1\n', both)
  expect(down[0]).toEqual({ name: 'alpha', error: 'unreachable', jobs: both[0]?.jobs })
})

test('observeTasks adds a started task and updates it from wait', () => {
  const started = observeTasks([], 'OK task=sweep job=alpha:1001 nodes=node-12 logs=/x/sweep.<rank>.log\n')
  expect(started).toEqual([{ jobId: 'alpha:1001', name: 'sweep', state: 'RUNNING' }])

  const done = observeTasks(
    started,
    '  rank=0 node=node-12 state=DONE rc=1 log=/x/sweep.0.log\ntask=sweep job=alpha:1001 state=DONE rc=1\n',
  )
  expect(done).toEqual([{ jobId: 'alpha:1001', name: 'sweep', state: 'DONE', exitCode: 1 }])

  const unrelated = 'OK cluster=alpha free=4\nERR code=JOB_GONE task=sweep job=alpha:1001 job_state=CANCELLED\n'
  expect(observeTasks(done, unrelated)).toEqual(done)

  expect(observeTasks([], 'task=t job=alpha:1001 state=NOT_STARTED\n')[0]?.state).toBe('NOT_STARTED')
})

test('formatDuration shortens slurm times', () => {
  expect(formatDuration('1-04:00:00')).toBe('1d4h')
  expect(formatDuration('4:00:00')).toBe('4h')
  expect(formatDuration('3:50:00')).toBe('3h50m')
  expect(formatDuration('12:30')).toBe('12m')
  expect(formatDuration('UNLIMITED')).toBe('UNLIMITED')
})

const BAND = {
  plugin: 'remote-gpu',
  surface: 'terminal',
  component: 'AbovePrompt',
  props: { hasSurvey: false, isWorking: false, maxRows: 20 } as RenderPropsOf['AbovePrompt'],
} as const

test('the band shows jobs while held, keeps them while unreachable, and goes once all are released', async ($, on) => {
  const clock = mock.clock(on)
  let table = TABLE
  let isDown = false
  let lsCalls = 0
  on('process.run', ($, e) => {
    const command = e.argv.at(-1)
    if (isDown) return { value: result(255, command === 'ls' ? '== alpha\n' : '', 'ssh failed\n') }
    if (command === 'ls') lsCalls += 1
    return { value: result(0, command === 'ls' ? `== alpha\n${table}` : 'OK cluster=alpha free=4 idle=4\n', '') }
  })
  let ranCommand = ''
  on('session.id', () => ({ value: 's1abcdef-0000-4000-8000-000000000000' }))
  on('tool.call', ($, e) => {
    if (e.tool === 'Bash') ranCommand = e.command
    return { result: { stdout: '', stderr: '' }, text: 'OK task=sweep job=alpha:1001\n' }
  })
  on('ui.render', ($, e) => {
    const { Box } = $.ui.resolve(e)
    return <Box />
  })

  const band = await $.ui.mount(BAND)
  expect(await band.find({ text: /remote-gpu/ })).toBeUndefined()

  await $.tool.call({ tool: 'Bash', command: 'remote-gpu bg -j alpha:1001 --name sweep -- x' })
  await clock.settle()
  expect(ranCommand).toBe('export REMOTE_GPU_SESSION=s1abcdef; remote-gpu bg -j alpha:1001 --name sweep -- x')
  expect(await band.find({ text: /alpha 4 free/ })).toBeDefined()
  expect(await band.find({ text: /node-12 · 3h50m left · idle 3m · this session/ })).toBeDefined()
  expect(await band.find({ text: /pending · 4h limit · alpha-ffff0000-moe$/ })).toBeDefined()
  expect(await band.find({ text: /◉ sweep/ })).toBeDefined()

  isDown = true
  await clock.advance(60_000)
  expect(await band.find({ text: /alpha unreachable/ })).toBeDefined()
  expect(await band.find({ text: /alpha:1001/ })).toBeDefined()
  isDown = false

  table = HEADER
  await clock.advance(60_000)
  expect(await band.find({ text: /remote-gpu/ })).toBeUndefined()

  const callsWhenReleased = lsCalls
  await clock.advance(180_000)
  expect(lsCalls).toBe(callsWhenReleased)
})

function result(exitCode: number, stdout: string, stderr: string) {
  return { exitCode, stdout, stderr, isStdoutTruncated: false, isStderrTruncated: false }
}

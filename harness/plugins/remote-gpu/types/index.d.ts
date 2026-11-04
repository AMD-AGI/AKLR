/** One row of `remote-gpu ls`. */
export type Job = {
  /** `cluster:id`, as the JOB column of `ls` and `job=` of `bg` print it. */
  id: string
  /** `<cluster>-<session tag>-<label or random>` for a job `alloc` made from a Claude session. */
  name: string
  state: string
  nodes: string
  timeLeft: string
  idle: string
}

/** What `remote-gpu ls` and `free` last said about one cluster; on an error, the jobs last known. */
export type Cluster =
  | { name: string; freeNodes: number; jobs: Job[] }
  | { name: string; error: string; jobs: Job[] }

/** A background task, as the `remote-gpu` calls of this session reported it. */
export type Task = {
  jobId: string
  name: string
  state: 'RUNNING' | 'DONE' | 'LOST' | 'NOT_STARTED'
  /** Present when `state` is `DONE`. */
  exitCode?: number
}

declare module 'claude-code' {
  interface PluginState {
    'remote-gpu': { clusters: Cluster[]; tasks: Task[] }
  }
}

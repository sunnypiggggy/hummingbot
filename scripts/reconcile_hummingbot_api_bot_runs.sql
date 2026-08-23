BEGIN;

-- A stable Docker instance can have only one active lifecycle row. Keep the
-- newest row and close older rows left by timed-out or repeated deployments.
WITH ranked AS (
    SELECT
        id,
        row_number() OVER (
            PARTITION BY instance_name
            ORDER BY deployed_at DESC, id DESC
        ) AS active_rank,
        lag(deployed_at) OVER (
            PARTITION BY instance_name
            ORDER BY deployed_at DESC, id DESC
        ) AS superseded_at
    FROM bot_runs
    WHERE deployment_status = 'DEPLOYED'
      AND run_status IN ('RUNNING', 'CREATED')
), duplicates AS (
    SELECT id, superseded_at
    FROM ranked
    WHERE active_rank > 1
)
UPDATE bot_runs AS runs
SET
    run_status = 'STOPPED',
    stopped_at = COALESCE(runs.stopped_at, duplicates.superseded_at, now()),
    final_status = COALESCE(
        runs.final_status,
        json_build_object(
            'reason', 'duplicate_active_run_reconciled',
            'reconciled_at', now()
        )::text
    )
FROM duplicates
WHERE runs.id = duplicates.id;

CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_runs_one_active_instance
ON bot_runs (instance_name)
WHERE deployment_status = 'DEPLOYED'
  AND run_status IN ('RUNNING', 'CREATED');

COMMIT;

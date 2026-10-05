"""Bounded API snapshot retention via local psql; no credentials or Stock writes."""
import argparse
import json
import subprocess

DATABASE = "hummingbot_api"
CONTAINER = "hummingbot-api-postgres"
TEST_DATABASE = "hummingbot_api_retention_test"


def plan(*, apply=False, testing=False):
    database = TEST_DATABASE if testing else DATABASE
    # Keep ALL latest-timestamp ties, matching API latest-state queries.
    account_where = '''a.timestamp < now()-interval '370 days'
      AND EXISTS (SELECT 1 FROM public.account_states newer
        WHERE newer.account_name IS NOT DISTINCT FROM a.account_name
          AND newer.connector_name IS NOT DISTINCT FROM a.connector_name
          AND newer.timestamp>a.timestamp)'''
    performance_where = '''a.timestamp < now()-interval '370 days'
      AND EXISTS (SELECT 1 FROM public.controller_performance_snapshots newer
        WHERE newer.bot_name IS NOT DISTINCT FROM a.bot_name
          AND newer.controller_id IS NOT DISTINCT FROM a.controller_id
          AND newer.timestamp>a.timestamp)'''
    guards = f"""
BEGIN;
SET LOCAL lock_timeout='1s';
SET LOCAL statement_timeout='15s';
SET LOCAL search_path=pg_catalog;
DO $guard$
BEGIN
 IF current_database()<>'{database}' THEN
   RAISE EXCEPTION 'maintenance database not allowlisted';
 END IF;
 IF NOT pg_try_advisory_xact_lock(146375742,370) THEN
   RAISE EXCEPTION 'maintenance already running' USING ERRCODE='55P03';
 END IF;
END $guard$;
-- Prevent concurrent schema changes during validation and deletion.
LOCK TABLE public.account_states,public.token_states,
 public.controller_performance_snapshots IN ACCESS SHARE MODE;
DO $guard$
BEGIN
 IF (SELECT count(*) FROM information_schema.columns
     WHERE table_schema='public' AND (
      (table_name='account_states' AND column_name IN ('id','timestamp','account_name','connector_name')) OR
      (table_name='token_states' AND column_name IN ('id','account_state_id')) OR
      (table_name='controller_performance_snapshots' AND column_name IN ('id','timestamp','bot_name','controller_id'))))<>10
 THEN RAISE EXCEPTION 'maintenance schema changed'; END IF;
 IF (SELECT count(*) FROM pg_constraint WHERE contype='f'
     AND conrelid='public.token_states'::regclass
     AND confrelid='public.account_states'::regclass
     AND pg_get_constraintdef(oid)='FOREIGN KEY (account_state_id) REFERENCES public.account_states(id)')<>1
 THEN RAISE EXCEPTION 'account snapshot dependency changed'; END IF;
 IF EXISTS (SELECT 1 FROM pg_constraint WHERE contype='f'
     AND confrelid IN ('public.account_states'::regclass,'public.token_states'::regclass,
                      'public.controller_performance_snapshots'::regclass)
     AND NOT (conrelid='public.token_states'::regclass
       AND confrelid='public.account_states'::regclass
       AND pg_get_constraintdef(oid)='FOREIGN KEY (account_state_id) REFERENCES public.account_states(id)'))
 THEN RAISE EXCEPTION 'unreviewed incoming dependency'; END IF;
 IF EXISTS (SELECT 1 FROM pg_trigger WHERE NOT tgisinternal
     AND tgrelid IN ('public.account_states'::regclass,'public.token_states'::regclass,
                    'public.controller_performance_snapshots'::regclass))
 THEN RAISE EXCEPTION 'unreviewed maintenance trigger'; END IF;
END $guard$;
CREATE TEMP TABLE retention_candidates ON COMMIT DROP AS
 SELECT 'account_states'::text AS kind,a.id FROM public.account_states a
 WHERE {account_where} ORDER BY a.timestamp,a.id LIMIT 1000;
INSERT INTO retention_candidates SELECT 'controller_performance_snapshots',a.id
 FROM public.controller_performance_snapshots a WHERE {performance_where}
 ORDER BY a.timestamp,a.id LIMIT 1000;
CREATE TEMP TABLE retention_deleted(kind text,records bigint) ON COMMIT DROP;
"""
    deletion = """
WITH removed AS (DELETE FROM public.token_states t USING retention_candidates c
 WHERE c.kind='account_states' AND t.account_state_id=c.id RETURNING t.id)
 INSERT INTO retention_deleted SELECT 'token_states',count(*) FROM removed;
WITH removed AS (DELETE FROM public.account_states a USING retention_candidates c
 WHERE c.kind='account_states' AND a.id=c.id RETURNING a.id)
 INSERT INTO retention_deleted SELECT 'account_states',count(*) FROM removed;
WITH removed AS (DELETE FROM public.controller_performance_snapshots a USING retention_candidates c
 WHERE c.kind='controller_performance_snapshots' AND a.id=c.id RETURNING a.id)
 INSERT INTO retention_deleted SELECT 'controller_performance_snapshots',count(*) FROM removed;
""" if apply else ""
    result = f"""
SELECT json_build_object('database',current_database(),'retention_days',370,
 'apply',{'true' if apply else 'false'},'batch_limit',1000,
 'database_bytes',pg_database_size(current_database()),
 'candidate_counts',(SELECT coalesce(json_object_agg(kind,n),'{{}}'::json) FROM
   (SELECT kind,count(*) AS n FROM retention_candidates GROUP BY kind) s),
 'deleted',(SELECT coalesce(json_object_agg(kind,records),'{{}}'::json) FROM retention_deleted),
 'remaining',json_build_object(
   'account_states',(SELECT count(*) FROM public.account_states a WHERE {account_where}),
   'controller_performance_snapshots',(SELECT count(*) FROM public.controller_performance_snapshots a WHERE {performance_where})),
 'protected','latest snapshots; orders; trades; executors; positions; funding; recovery; gateway evidence',
 'space_policy','freed pages reusable; no VACUUM FULL on production');
COMMIT;
"""
    return guards+deletion+result


def run(*, apply=False):
    result = subprocess.run(
        ["docker","exec","-i",CONTAINER,"nice","-n","19","psql","-X",
         "-U","hbot","-d",DATABASE,"-qAt","-v","ON_ERROR_STOP=1","-v","VERBOSITY=verbose"],
        input=plan(apply=apply),text=True,capture_output=True,timeout=120)
    if result.returncode:
        # Never echo DSNs or arbitrary database diagnostics; keep actionable safe codes.
        if "55P03" in result.stderr or "57014" in result.stderr:
            return {"database":DATABASE,"skipped":True,"reason":"database_busy_retry_next_day"}
        raise RuntimeError("API maintenance failed validation/execution; no partial transaction committed")
    return json.loads(result.stdout.strip())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(apply=args.apply)))


if __name__ == "__main__":
    main()

"""Real PostgreSQL rehearsal against a dedicated, newly-created test database only.

Run on OCI with a restricted owner role after creating the test DB. Never creates
or deletes databases itself. No production data, credentials, or economic API.
"""
import json
import subprocess

from scripts.api_database_retention import CONTAINER, TEST_DATABASE, plan

ROLE='hbot_api_retention_test'


def sql(query, *, check=True):
    result=subprocess.run(['docker','exec','-i',CONTAINER,'psql','-X','-U',ROLE,'-d',TEST_DATABASE,
                           '-qAt','-v','ON_ERROR_STOP=1'],input=query,text=True,capture_output=True,timeout=60)
    if check and result.returncode:
        raise RuntimeError(result.stderr)
    return result


def main():
    assert sql('SELECT current_database()').stdout.strip()==TEST_DATABASE
    sql('''CREATE TABLE account_states(id integer PRIMARY KEY,timestamp timestamptz,account_name text,connector_name text);
     CREATE TABLE token_states(id integer PRIMARY KEY,account_state_id integer REFERENCES account_states(id),units numeric);
     CREATE TABLE controller_performance_snapshots(id integer PRIMARY KEY,timestamp timestamptz,bot_name text,controller_id text);
     CREATE TABLE trades(id integer PRIMARY KEY,quantity numeric,fee numeric);
     INSERT INTO trades VALUES(1,1.23,0.35);
     INSERT INTO account_states VALUES
      (1,now()-interval '371 days','current','binance'),(2,now(),'current','binance'),
      (3,now()-interval '500 days','offline','binance'),
      (4,now()-interval '369 days','current','binance');
     INSERT INTO token_states VALUES(1,1,1),(2,2,2),(3,3,3),(4,4,4);
     INSERT INTO controller_performance_snapshots VALUES
      (1,now()-interval '371 days','bot','controller'),(2,now(),'bot','controller'),
      (3,now()-interval '500 days','offline','controller'),
      (4,now()-interval '500 days','offline','controller');''')
    preview=json.loads(sql(plan(testing=True)).stdout)
    assert preview['candidate_counts']=={'account_states':1,'controller_performance_snapshots':1},preview
    assert sql('SELECT count(*) FROM account_states').stdout.strip()=='4'
    # An external recovery dependency must fail the whole transaction, not cascade.
    sql('CREATE TABLE recovery_ref(snapshot integer REFERENCES account_states(id)); INSERT INTO recovery_ref VALUES(1);')
    assert sql(plan(apply=True,testing=True),check=False).returncode!=0
    assert sql('SELECT count(*) FROM account_states').stdout.strip()=='4'
    sql('DROP TABLE recovery_ref;')
    # Fail after the child delete; PostgreSQL rolls children and parents back together.
    interrupted=plan(apply=True,testing=True).replace('COMMIT;','SELECT 1/0; COMMIT;')
    assert sql(interrupted,check=False).returncode!=0
    assert sql('SELECT count(*) FROM token_states').stdout.strip()=='4'
    applied=json.loads(sql(plan(apply=True,testing=True)).stdout)
    assert applied['deleted']=={'token_states':1,'account_states':1,'controller_performance_snapshots':1},applied
    assert sql('SELECT count(*) FROM account_states WHERE account_name=\'offline\'').stdout.strip()=='1'
    assert sql('SELECT count(*) FROM controller_performance_snapshots WHERE bot_name=\'offline\'').stdout.strip()=='2'
    assert json.loads(sql(plan(apply=True,testing=True)).stdout)['deleted']=={
        'token_states':0,'account_states':0,'controller_performance_snapshots':0}
    assert sql('SELECT quantity,fee FROM trades').stdout.strip()=='1.23|0.35'
    # Batching must preserve the latest group even after catch-up restarts.
    sql('''INSERT INTO account_states SELECT i,now()-interval '400 days','backlog','binance'
          FROM generate_series(100,1200) i;
          INSERT INTO account_states VALUES(1201,now(),'backlog','binance');
          INSERT INTO token_states SELECT id,id,1 FROM account_states WHERE id>=100;''')
    first=json.loads(sql(plan(apply=True,testing=True)).stdout)
    assert first['deleted']['account_states']==1000 and first['remaining']['account_states']==101
    second=json.loads(sql(plan(apply=True,testing=True)).stdout)
    assert second['deleted']['account_states']==101
    assert sql('SELECT count(*) FROM account_states WHERE id=1201').stdout.strip()=='1'
    assert sql(plan(apply=True),check=False).returncode!=0  # Wrong DB rejected before deleting.
    print(json.dumps({'database':TEST_DATABASE,'restricted_role':ROLE,'passed':True,
                      'scenarios':['preview','370_day_boundary','latest_old_group','latest_timestamp_ties',
                      'incoming_dependency','transaction_crash_rollback','idempotency','bounded_batches',
                      'economic_evidence_unchanged','production_target_rejected']}))


if __name__=='__main__':
    main()

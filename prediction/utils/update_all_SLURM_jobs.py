import os

# get all job ids
cmd_out = os.popen("squeue -u $USER").read()
out_lines = cmd_out.split('\n')[1:-1]
job_ids = [list(filter(None,out_line.split(' ')))[0] for out_line in out_lines]

for job_id in job_ids:
    os.popen(f'scontrol update jobid={job_id} TimeLimit=00:45:00')


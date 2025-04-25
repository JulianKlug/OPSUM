## Server Instructions

1. Launch the database: `sbatch db_process.sbatch`
2. Launch the master_launcher: `python master_launcher.py -d /.../temp/train_data_splits.pth  -o /.../short_term_outcomes -c /.../end_config.json -n 1 -spwd opsum -sport 6380 -shost cpu004 -tte -f`
   - this will also launch subprocesses
   - as well as an optuna frontend at localhost:5555
3. The dashboard can also be relaunched at a later time: `optuna-dashboard redis://default:opsum@cpu004:6380/opsum --port 5555`'

Requirements
- Redis install: https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-from-source/ and https://github.com/liukidar/stune 
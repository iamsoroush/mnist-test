import argparse
import os
from pathlib import Path

import pyslurm


# Make sure to clone/fetch your repository

USER = os.getenv('USER')
ORCHESTRATOR_PATH = Path('/home').joinpath(USER).joinpath('scratch').joinpath('orchestrator.py')
MAX_JOBS = 10


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--repo_root_dir',
                        type=str,
                        help='**absolute** directory of the repository.',
                        required=True)

    parser.add_argument('--run_name',
                        type=str,
                        help='repo_root_dir/runs/run_name',
                        required=True)

    parser.add_argument('--email',
                        type=str,
                        help='your job-events will be sent to this address',
                        required=True)

    parser.add_argument('--branch',
                        type=str,
                        help='directory of the config file',
                        required=False,
                        default='master')

    parser.add_argument('--hours',
                        type=int,
                        help='minimum required hours',
                        required=False,
                        default=1)

    parser.add_argument('--mem',
                        type=int,
                        help='minimum required memory in megabytes',
                        required=False,
                        default=8192)

    parser.add_argument('--data_dir',
                        type=str,
                        help='**absolute** directory of the dataset',
                        required=True)

    parser.add_argument('--commit',
                        help='set this argument to commit and push all the content of the run_dir.',
                        action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    jobs = pyslurm.job().find_user(USER)
    if len(jobs) >= MAX_JOBS:
        print(f'job queue is full, try again later.')
    else:
        repo_root = Path(args.repo_root_dir).absolute()
        assert repo_root.is_dir(), f"{repo_root} is not a directory."

        run_dir = repo_root.joinpath('runs').joinpath(args.run_name)
        job_name = f'{repo_root.name}_{args.run_name}'
        branch = args.branch
        email = args.email
        memory = args.mem
        hours = args.hours
        do_commit = args.commit

        script_path = repo_root.joinpath(job_name + '.job')

        with open(script_path, 'w') as f:
            f.write('#!/bin/env bash\n\n')

            f.write('#SBATCH --get-user-env\n')
            f.write('#SBATCH --gres gpu:p100:1\n')
            f.write(f'#SBATCH --job-name {job_name}\n')
            f.write(f'#SBATCH --time {hours}:00:00\n')
            f.write(f'#SBATCH --output log-{job_name}.out\n')
            f.write(f'#SBATCH --error log-{job_name}.err\n')
            f.write(f'#SBATCH --mem {memory}\n')
            f.write(f'#SBATCH --mail-user {email}\n\n')

            f.write(f'cd {str(repo_root)}\n')
            f.write(f'git fetch --all\n')
            f.write(f'git checkout -b {branch}\n')
            f.write(f'python3 {str(ORCHESTRATOR_PATH)} --run_dir {str(run_dir)} --data_dir {str(args.data_dir)}\n')

            if do_commit:
                f.write(f'cd {repo_root}\n')
                f.write(f'git add {run_dir}\n')
                f.write(f'git commit {run_dir} -m "add run results, from graham"\n')
                f.write(f'git push origin\n')

        os.system(f'cd {run_dir}\n')
        os.system(f'sbatch {script_path}')

# Run training on SLURM cluster

Steps: 
1. Clone the repository
2. Create a virtual environment and activate it:
   - ``conda activate <env_name>``
3. Install the requirements:
    - ``pip install -r requirements.txt``
4. Add the repository to the PYTHONPATH:
    - ``export PYTHONPATH=$PYTHONPATH:<path_to_repo>``
5. Run the training job creation script (creates multiple SLURM jobs for hyperparameter search):
    - ``python make_SLURM_jobs.py``
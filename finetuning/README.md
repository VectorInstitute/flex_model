# Running the script
Build the environment using `pip install -r requirements.txt`, but it should
otherwise run on the base `FlexModel` environment.

Run `accelerate config` after you've built the environment, and follow
instructions for generating the config file. This script was tested using
FSDP. See `default_config.yaml` for an example.

Note: Some features like checkpointing and loading from more detailed config
scripts is still WIP. This script is to be used more as a seed version of a
real finetuning workflow.

To run the finetuning loop, run: `accelerate launch accelerate_train.py`.

Slurm script: WIP

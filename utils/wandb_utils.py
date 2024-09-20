
import wandb

def init_or_resume_wandb_run(wandb_id_file_path,
                             config):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file. 
        
        Returns the config, if it's not None it will also update it first
        
        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
            
            Group runs on wandb user interface by "Name, Job Type, Group"
            Name is defined by config["wandb"]["run_name"]
            Job Type is defined by job_type arg in wandb.init
            Group is defined by config["extra_tag"]
    """
    # if the run_id was previously saved, resume from there
    
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        print("path exists init wandb")
        wandb.init(name=config["wandb"]["run_name"], config=config,
                            project=config["wandb"]["project"],
                            entity=config["wandb"]["entity"],
                            group=config["wandb"]["group"],
                            job_type=config["wandb"]["job_type"],
                            id=resume_id, resume='must', dir=config["wandb"]["dir"])
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        print("path not exists init wandb")
        print(config["wandb"])
        run = wandb.init(name=config["wandb"]["run_name"], config=config,
                            project=config["wandb"]["project"],
                            entity=config["wandb"]["entity"],
                            group=config["wandb"]["group"],
                            job_type=config["wandb"]["job_type"])
        print("init wandb")
        wandb_id_file_path.write_text(str(run.id))

    print("update config")
    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        config.update(wandb_config)

    return config
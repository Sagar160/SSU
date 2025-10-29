# Imports
import wandb


class WandbLogger:
    def __init__(self, 
                 logging,
                 project_name, 
                 entity, 
                 name,
                 group=None, 
                 tags=None,
                 notes=None, 
                 config=None,
                 resume=None,
                 id=None):
        self.logging = logging
        self.project_name = project_name
        self.entity = entity
        self.name = name
        self.group = group
        self.tags = tags if tags is not None else []
        self.notes = notes
        self.config = config if config is not None else {}
        self.init_run = False
        self.save_logs = []
        self.resume = resume
        self.id = id
    
    def update_config(self, key, value):
        if self.logging:
            if key in self.config:
                self.config[key] = value
            else:
                raise KeyError(f"Key '{key}' not found in config. Please add it before updating.")
        else:
            print("Logging is disabled. Cannot update config.")

    def init(self):
        if self.logging:
            self.init_run = True
            if self.resume is None:
                wandb.init(project=self.project_name, 
                            entity=self.entity, 
                            name=self.name, 
                            group=self.group,
                            tags=[str(tag) for tag in self.tags],
                            notes=self.notes,
                            config=self.config)
            else:
                wandb.init(project=self.project_name, 
                            entity=self.entity, 
                            name=self.name, 
                            group=self.group,
                            tags=[str(tag) for tag in self.tags],
                            notes=self.notes,
                            config=self.config,
                            resume=self.resume,
                            id=self.id)
                
            if len(self.save_logs) > 0:
                for log in self.save_logs:
                    wandb.log(log)
                self.save_logs = []

    def log(self, dict):
        if self.logging:
            if self.init_run:
                wandb.log(dict)
            else:
                self.save_logs.append(dict)

    def finish(self):

        if self.logging:
            wandb.finish()
        else:
            print("Wandb logging is disabled. No logs to finish.")
            # save locally the saved logs
            # print("Wandb logging is disabled. Saving logs locally.")
            # file_name = self.config['config_file_name'].replace('.yaml', '')
            # with open(f'wandb_logs_saved_locally/{file_name}.txt', 'w') as f:
            #     for log in self.save_logs:
            #         f.write(str(log) + '\n')
from dataclasses import field, dataclass


@dataclass
class LongFormerArguments:
    hidden_dim: int = field(default=768)
    model_path: str = field(default='cache/bert-chinese')


@dataclass
class TrainArguments:
    seed: int = field(default=42)
    epoch: int = field(default=2)
    batch_size: int = field(default=16)
    learning_rate: float = field(default=5e-5)
    dropout_rate: float = field(default=0.2)
    warmup_step: int = field(default=5,
                             metadata={'help': "number of training steps to conduct warmup strategy"})

    warmup_strategy: str = field(default='Linear',
                                 metadata={'help': 'warmup strategy used in training phase'})
    task_type: str = field(default='sent',
                           metadata={'help': 'the type of task'})
    nproc: int = field(default=1,
                       metadata={'help': 'number of process are started for using distributing training'})

    device: str = field(default='cpu',
                        metadata={'help': 'the device used to train models'})
    method: str = field(default='long-attention',
                        metadata={'help': 'methods use to perform document '
                                          'classification either long-attention or split-sentence'})
    log_dir: str = field(default='log/', metadata={'help': 'directory to save log files'})
    dataset_path: str = field(default='spam/')
    reader: str = field(default='csv', metadata={'help': 'methods used to read source file'})
    adam_eps: float = field(default=1e-6, metadata={'help': 'term add to the denominator'
                                                    })
    early_stop: int = field(default=5, metadata={'help': 'number of early stop steps'})

    save_to: str = field(default='saved_model', metadata={'help': 'directory used to save model'})
    label_mapping: str = field(default='label_mapping.json', metadata={'help':
                                                                       'saved label mapping for some specific tasks '
                                                                       'with string label'})
    best_model_path: str = field(default='saved_model/roberta-chinese_5.pth')
    given_best_model: bool = field(default=False, metadata={'help': 'if give best model path'})

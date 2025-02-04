# optimized for H100, smaller gpu need smaller number of setup peft (lora) tuning. 
MICRO_BATCH_SIZE = 256  
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 2e-5  
EPOCHS = 3

CONTEXT_ENCODER_PATH = ''
CONCEPT_ENCODER_PATH = ''
MODEL_SAVE_DIR = ""
TRAIN_NAME = ''
DATASET = ''

CUDA_VISIBLE_DEVICES = "0"
USE_CUDA = False if CUDA_VISIBLE_DEVICES == "-1" else True

PROJ = 'linear'
FREEZE_PARAMS = True
USEMINER = False
ALPHA = 2
BETA = 40
BASE = 0.5
PEFT_TUNING = False
PEFT_TUNING_MODULE = 'attention'
ADD_SOFT_TOKEN = True
FINETUNING = False
UNIENCODER = False
hparams = {
    "seed": 123,
    "gpus": 1,
    "gpu_mode": True,  #True
    "crop_size": None,
    "resume": False,
    "train_data_path": "C:/Data/",
    "augment_data": True,
    "epochs": 200,
    "batch_size": 1,
    "threads": 0,
    "height":640, #1280, 512, 288 niedrigste zahl = 248
    "width":360,    #720, 288, 288 solange durch 8 teilbar & >= 248
    "lr":2e-04,
    "beta1": 0.9595,
    "beta2": 0.9901,
    "start_epoch": 0,
    "save_folder": "./weights/",
    "model_type": "Ad Detection",
    "snapshots": 10,
    "resume_train": "./weights/weight.pth"
}

#sec best params 93.97999572753906.
# mhac_filter: 64
# mha_filter: 32
# num_mhablock: 6
# num_mhac: 7
# gen_lambda: 0.5
# pseudo_lambda: 0.7000000000000001
# pseudo_alpha: 1.0
# hazy_alpha: 1.0
import optuna
import torch
from models import Dehaze
import torch.optim as optim
from training import sparse_training, test
from data import ImageDataset
from loss import ChabonnierLoss
from torch.utils.data import DataLoader
from params import hparams
import numpy as np
import myutils
import time
import subprocess as sp
import os


def objective(trial):
    size = (hparams["height"], hparams["width"])
    scaler = torch.cuda.amp.GradScaler()
    params = {
              "mhac_filter":trial.suggest_categorical("mhac_filter", [16,32,64]),
             "mha_filter":trial.suggest_categorical("mha_filter", [16,32, 64]),
              "num_mhablock":trial.suggest_int("num_mhablock", 4,9),
              "num_mhac":trial.suggest_int("num_mhac", 4, 9),
              #"down_deep":trial.suggest_categorical("down_deep", [False,True]),
              #"lr":trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True),
              #"beta1":trial.suggest_float("beta1", 0.85, 1),
              #"beta2":trial.suggest_float("beta2", 0.9, 1),
              "gen_lambda":trial.suggest_float("gen_lambda", 0.1, 1, step=0.1),
              "pseudo_lambda":trial.suggest_float("pseudo_lambda", 0.1, 1, step=0.1),
              "pseudo_alpha":trial.suggest_float("pseudo_alpha", 0.1, 1, step=0.1),
              "hazy_alpha":trial.suggest_float("hazy_alpha", 0.1, 1, step=0.1)
              }
    
    model = Dehaze(params["mhac_filter"], params["mha_filter"], params["num_mhablock"], params["num_mhac"], down_deep=hparams["down_deep"], pseudo_alpha=params["pseudo_alpha"], hazy_alpha=params["hazy_alpha"],size=(hparams["height"], hparams["width"])).cuda()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"], betas=(hparams["beta1"],hparams["beta2"]))
    criterion = ChabonnierLoss(eps = 1e-6).cuda()
    dataloader = DataLoader(ImageDataset(hparams["train_data_path"] + "NH-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    testloader = DataLoader(ImageDataset("C:/Data/dehaze/test",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    size_weight = 1 / (size_all_mb/8)

    skip = False
    if(size_all_mb <= 8):
        size_weight = 1
    if(size_all_mb >= 9):
        skip = True

    if skip is False:
        for i in range(5):
            sparse_training(model, optimizer, dataloader, criterion, params, scaler)




    torch.save(model.state_dict(), "./tune_saves/" + str(params["mhac_filter"])+","+str(params["mha_filter"])+ ","+str(params["num_mhablock"])+","+ str(params["num_mhac"]) + ".pth")

    start_time = time.time()
    accuracy = test(model, testloader)

    end_time = time.time()
    process_time = (end_time-start_time) / testloader.__len__()

    print(accuracy)
    parameter = (accuracy + 1) * size_weight
    if skip is True: parameter = 0

    return parameter

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
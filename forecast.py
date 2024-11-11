import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

from util.general import AverageMeter


def train_transformer(args, cfg, model, train_loader, val_loader, logger, model_save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss(reduction='mean')

    best_val_loss = 1e10
    best_val_epoch = 0
    patience_count = 0
    for epoch in tqdm(range(args.epoch), desc='Training'):
        train_epoch_loss = 0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            data = batch[0].to(args.device)
            x = data[:,:-1,:]
            y = data[:,-1,:].unsqueeze(1)

            SOS = data[:,-2,:].unsqueeze(1)
            EOS = torch.zeros((y.shape[0], 1, y.shape[-1]), dtype=torch.float32).to(args.device)
            tgt = torch.cat([SOS, y, EOS], axis=1)
            tgt = tgt[:, :-1, :] # input: [-2,-1]

            y_hat = model(x, tgt) # [-1,EOS]

            loss = criterion(y_hat[:,:-1,:], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
        train_epoch_loss = train_epoch_loss / len(train_loader)
        
        val_epoch_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                data = batch[0].to(args.device)
                x = data[:,:-1,:]
                y = data[:,-1,:].unsqueeze(1)
                generated_sequence = []

                # Start with an initial zero target (SOS)
                tgt = data[:,-2,:].unsqueeze(1)
                for _ in range(y.shape[1]):
                    output = model(x, tgt)
                    next_value = output[:, -1:, :]
                    generated_sequence.append(next_value)
                    tgt = torch.cat((tgt, next_value), dim=1)
                prediction = torch.cat(generated_sequence, dim=1)

                loss = criterion(prediction, y)
                val_epoch_loss += loss.item()
            val_epoch_loss = val_epoch_loss / len(val_loader)

        logger.info(f"Epoch {epoch+1} | Train loss {train_epoch_loss:.5f} / Validation loss {val_epoch_loss:.5f}")
        if val_epoch_loss < best_val_loss:
            logger.info(f"Lower validation loss, saving the model.. Current {best_val_loss:.5f} → {val_epoch_loss:.5f}")
            best_val_loss = val_epoch_loss
            best_val_epoch = epoch+1
            patience_count = 0
            torch.save(model, model_save_path)
        else:
            patience_count += 1
            if patience_count == args.patience:
                logger.info("Early stop triggered..")
                break
    logger.info(f"Best model saved to {model_save_path} : Best epoch {best_val_epoch}, Best val loss {best_val_loss:.5f}")


## Calculate the mean and var of modeling error using validation set
def normalize_anomaly_score_transformer(args, best_model, val_loader):
    val_error = []
    best_model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
            data = batch[0].to(args.device)
            x = data[:,:-1,:]
            y = data[:,-1,:].unsqueeze(1)
            generated_sequence = []

            # Start with an initial zero target (SOS)
            tgt = data[:,-2,:].unsqueeze(1)
            for _ in range(y.shape[1]):
                output = best_model(x, tgt)
                next_value = output[:, -1:, :] 
                generated_sequence.append(next_value)
                tgt = torch.cat((tgt, next_value), dim=1)

            prediction = torch.cat(generated_sequence, dim=1)
            loss = F.l1_loss(prediction, y, reduction='none')
            loss = torch.mean(loss, dim=(1)).detach().cpu().numpy()
            val_error.append(loss)

    val_error = np.concatenate(val_error, axis=0)
    mean_error = np.mean(val_error, axis=0)
    std_error = np.cov(val_error.T)

    return mean_error, std_error


def test_transformer(args, best_model, test_loader, AS_calculator):
        test_anomaly_score = []
        test_label = []
        best_model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
                data = batch[0].to(args.device)
                x = data[:,:-1,:]
                y = data[:,-1,:].unsqueeze(1)
                label = batch[1]
                generated_sequence = []

                # Start with an initial zero target (SOS)
                tgt = data[:,-2,:].unsqueeze(1)
                for _ in range(y.shape[1]):
                    output = best_model(x, tgt) 
                    next_value = output[:, -1:, :] 
                    generated_sequence.append(next_value)
                    tgt = torch.cat((tgt, next_value), dim=1) 

                prediction = torch.cat(generated_sequence, dim=1)
                loss = F.l1_loss(prediction, y, reduction='none')
                loss = torch.mean(loss, dim=(1)).detach().cpu().numpy()
                anomaly_score = AS_calculator(loss)
                
                test_anomaly_score.append(anomaly_score)
                test_label.append(label[:,-1])

        test_anomaly_score = np.concatenate(test_anomaly_score, axis=0) 
        test_label = np.concatenate(test_label, axis=0) 

        return test_anomaly_score, test_label


import torch
import numpy as np
from tqdm import tqdm

def train(model, criterion, train_loader,val_loader, optimizer, num_epochs,device):
    """Simple training loop for a PyTorch model.""" 

    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    val_loss_min = np.Inf
    total_step = len(train_loader)
    
    # Make sure model is in training mode.
    model.train()
    
    # Move model to the device (CPU or GPU).
    model.to(device)
    
    # Exponential moving average of the loss.
    ema_loss = None
    
    # Loop over epochs.
    # for epoch in range(num_epochs):
    for epoch in tqdm(range(num_epochs)):
      running_loss = 0.0
      correct = 0
      total = 0
        
      # Loop over data.
      for data_train in train_loader:

          print(data_train)

          data,target=data_train

          target=target.type(torch.LongTensor)
            
          # Forward pass.
          output =model(data.to(device))
          loss = criterion(output.to(device).squeeze(), target.to(device))
          # loss.requires_grad = True
          # print(loss)
          # Backward pass.
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          running_loss +=loss.item()
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.cpu().eq(target.view_as(pred)).sum().item()
          total += target.size(0)
          
      print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, running_loss/total_step),
      )
      
      train_acc.append(100 * correct/total)
      train_loss.append(running_loss/total_step)



      # Validation
      batch_loss = 0
      total_t = 0
      correct_t = 0
      with torch.no_grad():
          model.eval()
          for  data in val_loader:
              data_t, target_t =data
              data_t,target_t = data_t.to(device),target_t.to(device)
              outputs_t = model(data_t)
              loss_t = criterion(outputs_t.squeeze(),target_t.long())
              batch_loss+=loss.item()
              _,pred_t = torch.max(outputs_t,dim=1)
              correct_t += torch.sum(pred_t==target_t).item()
              total_t += target_t.size(0)
          val_acc.append(100 * correct_t/total_t)
          val_loss.append(batch_loss/len(val_loader))
          network_achieve = batch_loss < val_loss_min
        

          if network_achieve:
              val_loss_min = batch_loss
              torch.save(model.state_dict(), './model/model_ok.ckpt')

              print('The best model is detected')


    # torch.save(model.state_dict(), 'model.pth')
    percent = 100. * correct / len(train_loader.dataset)
    return model, percent, val_loss, val_acc, train_loss, train_acc
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score


# def evaluate(model, dataloader, criterion, device, world_size, loc=0, scale=1):
#     """ Evaluate a model on a specific dataloader, with distributed communication (if necessary) """
#     model.eval()
#     N = torch.zeros(1).to(device)
#     score = torch.zeros(1).to(device)

#     with torch.no_grad():
#         for graph in dataloader:
#             graph = graph.to(device)
#             out = model(graph).squeeze()

#             n = graph.y.size(0)
#             N += n
#             score += n*criterion(out*scale + loc, graph.y)

#     model.train()
#     if world_size > 1:
#         dist.all_reduce(score)
#         dist.all_reduce(N)

#     return (score/N).item()



def evaluate(model, dataloader, criterion, device):
    """ Evaluate a model on a specific dataloader, with distributed communication (if necessary) """
    model.eval()
    valid_score =[]

    loss_sum = 0
    with torch.no_grad():
        for graph in dataloader:
            graph = graph.to(device)
            out = model(graph).squeeze()
            target = graph.y.squeeze()
            NA_Mat = torch.where(torch.abs(target)<0.5, torch.zeros_like(target), torch.ones_like(target))

            out = out * NA_Mat
            target = (target+1.0)/2.0 * NA_Mat
            loss = criterion(out, target) 
            # print(f" loss: {loss} shape: {loss.shape}")
            # break
            loss= loss.mean()
            loss_sum += loss.item()

          
            pred = torch.sigmoid(out) * NA_Mat
            target=target.detach().cpu().numpy()
            pred=pred.detach().cpu().numpy() 
          
            try:
                valid_score.append(roc_auc_score(target.ravel(), pred.ravel()))
            except:
                pass



    return loss_sum/len(dataloader), valid_score
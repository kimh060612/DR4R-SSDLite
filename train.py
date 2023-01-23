from torch.utils.tensorboard import SummaryWriter
import torch
from config.confg import getConfig
from model.ssdlite import SSDLite
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset.dataloader import build_train_dataloader, build_validation_dataloader
from dataset.cocoeval import coco_evaluation
import torch.distributed as dist
from tqdm import tqdm
import collections

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


if __name__ == "__main__":
    cfg = getConfig(backbone="MobileNeXt")
    model = SSDLite(cfg=cfg)
    print(model)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), 
                      lr=cfg["train"]["lr"], 
                      weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    train_loader = build_train_dataloader(cfg)
    val_loader = build_validation_dataloader(cfg)
    writer = SummaryWriter('tb_logs/')
    
    epochs = cfg["train"]["epoch"]
    global_step = 0
    best_loss = 987654321
    for e in range(epochs):
        model.train()
        try:
            losses_reduced = None
            for t, (images, targets, _) in enumerate(train_loader):
                global_step += 1
                images = images.to(device)
                targets = targets.to(device)
                loss_dict = model(images, targets=targets)
                loss = sum(loss for loss in loss_dict.values())
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if ((t + 1) % cfg["train"]["log_step"]) == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"[Epoch: {e + 1:04d}] step: {t + 1: 04d}, lr: {lr:.5f}, loss: {losses_reduced.item():.6e}")
                writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
            if best_loss >= losses_reduced.item():
                torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, cfg["checkpoint_dir"] + f"/model-SSDLite-{e:04d}-{best_loss:.3e}.pt")
        except KeyboardInterrupt:
            print("Detect the Keyboard Interrupt... Try to end session gracefully...")
        
        with torch.no_grad():
            model.eval()
            predictions = {}
            for batch in tqdm(val_loader):
                images, targets, image_ids = batch
                cpu_device = torch.device("cpu")
                with torch.no_grad():
                    outputs = model(images.to(device))
                    outputs = [ o.to(cpu_device) for o in outputs ]
                predictions.update(
                    { img_id: result for img_id, result in zip(image_ids, outputs) }
                )        
            eval = coco_evaluation(dataset=val_loader, predictions=predictions, output_dir=f'./result_{e + 1:02d}')
            write_metric(eval['metrics'], 'metrics/COCO', writer, e + 1)
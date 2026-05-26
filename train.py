import argparse
import torch
from pathlib import Path
from utils.utils import *
from utils.models import *
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image




def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_dir", type=str, default=r'C:\Users\shaurya mishra\OneDrive\Desktop\NST\content_data',
                        help='Location of content dataset')
    parser.add_argument("--style_dir", type=str, default=r'C:\Users\shaurya mishra\OneDrive\Desktop\NST\style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default=r'C:\Users\shaurya mishra\OneDrive\Desktop\NST\vgg_normalised.pth',
                        help='location of pre-trained VGG')
    parser.add_argument("--experiment",type=str,default='experiment1',
                        help='Name of experiment')
    parser.add_argument('--final_size',type=int,default=256,
                        help='Size of final image')
    parser.add_argument('--content_size',type=int,default=512,
                        help='Size of content image')
    parser.add_argument('--style_size',type=int,default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true',default=True,
                        help='Crop image')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                       help='Learning rate decay')
    parser.add_argument('--epochs', type=int, default=1,
                       help='No. of epochs')
    parser.add_argument('--content_weight', type=float, default=1.0,
                       help='Content Weight')
    parser.add_argument('--style_weight', type=float, default=5,
                       help='Style Weight')
    parser.add_argument('--log_interval', type=int, default=1,
                       help='Log interval')
    parser.add_argument('--save_interval', type=int, default=2,
                       help='Save interval')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume Training')
    parser.add_argument('--decoder_path', type=str, default=None,
                       help='Path to decoder checkpoint')
    parser.add_argument('--optimizer_path', type=str, default=None,
                       help='Path to optimizer checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='experiment/experiment1/checkpoint.pth',
                    help='Path to full checkpoint')
    


    return parser.parse_args()
                    

def main():
    args=parse_arguments()
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    save_dir = Path('experiment')/args.experiment
    save_dir.mkdir(exist_ok=True,parents=True)
    
    #save argument values
    with open(save_dir/'args.txt','w') as args_file:
        for key,value in vars(args).items():
            args_file.write(f'{key} : {value}\n')
   
    content_transform= get_tranform(args.content_size,args.crop,args.final_size)
    style_transform = get_tranform(args.style_size,args.crop,args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size,shuffle=True,pin_memory=True)

    print('Number of batches in content dataset',len(content_dataloader))
    print('Number of batches in style dataset',len(style_dataloader))

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0/(1.0 + args.lr_decay * epoch)
    )
    
    start_epoch = 0

    if args.resume and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)

        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")


    print("Training...")

    mse_loss = torch.nn.MSELoss()
    encoder.eval()


    for epoch in range(start_epoch, args.epochs):
        running_loss = 0
        running_closs = 0
        running_sloss = 0

        progress_bar = tqdm(zip(content_dataloader,style_dataloader),
                            total=min(len(content_dataloader),len(style_dataloader)))
                        
        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1],s_feats[-1])
            g = decoder(t)

            g_feats=encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s+= mse_loss(g_mean,s_mean) +mse_loss(g_std,s_std)
                
            loss_s =loss_s* args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step()

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)
        
        if (epoch+1) % args.log_interval ==0:
            tqdm.write(f'Iter {epoch+1} : Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

        if (epoch+1) % args.save_interval ==0:
            torch.save({
                        "epoch": epoch,
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }, save_dir / "checkpoint.pth")

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png',nrow=args.batch_size)


if __name__ == '__main__':
    main()
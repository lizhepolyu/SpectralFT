import argparse
import glob
import os
import time
import warnings

import torch

from tools import init_args
from dataLoader import TrainDataset
from ECAPAModel import ECAPAModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="ECAPA Trainer")

    # Training Settings
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Duration of the input segments, e.g., 200 for 2 seconds')
    parser.add_argument('--max_epoch', type=int, default=30,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=32,
                        help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1,
                        help='Test and save every [test_step] epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97,
                        help='Learning rate decay every [test_step] epochs')

    # Paths and Lists
    parser.add_argument('--train_list', type=str, default="/home8a/lizhe/voxceleb1/meta-data/train_list.txt",
                        help='Path to the training list')
    parser.add_argument('--train_path', type=str, default="/home8a/lizhe/voxceleb1/dev",
                        help='Path to the training data')
    parser.add_argument('--eval_list', type=str, default="/home8a/lizhe/voxceleb1/voxceleb1_test_v2.txt",
                        help='Path to the evaluation list')
    parser.add_argument('--eval_path', type=str, default="/home8a/lizhe/voxceleb1/test/",
                        help='Path to the evaluation data')
    parser.add_argument('--musan_path', type=str, default="/home8a/lizhe/musan/",
                        help='Path to the MUSAN dataset')
    parser.add_argument('--rir_path', type=str, default="/home8a/lizhe/RIRS_NOISES/simulated_rirs/",
                        help='Path to the RIR dataset')
    parser.add_argument('--save_path', type=str, default="exps/vox1",
                        help='Path to save scores and models')
    parser.add_argument('--initial_model', type=str, default="",
                        help='Path to the initial model')

    # Model and Loss Settings
    parser.add_argument('--C', type=int, default=1024,
                        help='Channel size for the speaker encoder')
    parser.add_argument('--m', type=float, default=0.2,
                        help='Loss margin in AAM softmax')
    parser.add_argument('--s', type=float, default=30,
                        help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int, default=1211,
                        help='Number of speakers')

    # Commands
    parser.add_argument('--eval', action='store_true',
                        help='Only perform evaluation')
    parser.add_argument('--visual', action='store_true',
                        help='Only perform visualization')

    args = parser.parse_args()
    args = init_args(args)
    return args

def main():
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_arguments()

    # Initialize Data Loader
    train_dataset = TrainDataset(**vars(args))
    train_dataset_instance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        drop_last=True,
        pin_memory=True
    )

    # Model Loading
    model_files = sorted(glob.glob(f'{args.model_save_path}/model_0*.model'))
    model = ECAPAModel(**vars(args))
    epoch = 1

    if args.visual:
        if not args.initial_model:
            raise ValueError("An initial model must be provided for visualization.")
        print(f"Loading model from {args.initial_model}")
        model.load_parameters(args.initial_model)
        model.visualization(loader=train_dataset_instance)
        return

    if args.eval:
        if not args.initial_model:
            raise ValueError("An initial model must be provided for evaluation.")
        print(f"Loading model from {args.initial_model}")
        model.load_parameters(args.initial_model)
        EER, minDCF = model.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
        print(f"EER: {EER:.2f}%, minDCF: {minDCF:.4f}")
        return

    if args.initial_model:
        print(f"Loading model from {args.initial_model}")
        model.load_parameters(args.initial_model)
    elif model_files:
        latest_model = model_files[-1]
        print(f"Resuming from model {latest_model}")
        model.load_parameters(latest_model)
        epoch = int(os.path.splitext(os.path.basename(latest_model))[0][6:]) + 1
    else:
        print("Starting training from scratch.")

    EERs = []
    minDCFs = []

    score_file_path = os.path.join(args.save_path, "scores.txt")
    with open(score_file_path, "a+") as score_file:
        while epoch <= args.max_epoch:
            # Training for one epoch
            loss, lr, acc = model.train_network(epoch=epoch, loader=train_dataset_instance)
            model.save_parameters(f"{args.model_save_path}/model_{epoch:04d}.model")
            print(f"Epoch {epoch}: Loss={loss:.2f}%, LR={lr:.6f}%, Accuracy={acc:.2f}%")

            # Evaluation
            if epoch % args.test_step == 0:
                EER, minDCF = model.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
                EERs.append(EER)
                minDCFs.append(minDCF)
                best_EER = min(EERs)
                best_minDCF = min(minDCFs)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Epoch {epoch}, "
                      f"Accuracy: {acc:.2f}%, EER: {EER:.2f}%, minDCF: {minDCF:.4f},"
                      f"Best EER: {best_EER:.2f}%, Best minDCF: {best_minDCF:.4f}")
                      
                score_file.write(f"Epoch {epoch}, LR={lr:.6f}%, Loss={loss:.2f}%, Accuracy={acc:.2f}%, "
                                f"EER={EER:.2f}%, minDCF={minDCF:.4f},"
                                f"Best EER={best_EER:.2f}%, Best minDCF={best_minDCF:.4f}\n")
                score_file.flush()
            epoch += 1

if __name__ == '__main__':
    main()
from model_def import GCLAD
from functions import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm import DDPM  # Import DDPM model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

# Argument parsing
parser = argparse.ArgumentParser(description='GCLAD with DDPM')
parser.add_argument('--experiment_id', type=int, default=0)
parser.add_argument('--gpu_device', type=str, default='cuda:1')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--mini_batch', type=int, default=256)
parser.add_argument('--subgraph_size', type=int, default=8)
parser.add_argument('--readout', type=str, default='sum')
parser.add_argument('--test_rounds', type=int, default=128)
parser.add_argument('--negative_sample_ratio_patch', type=int, default=4)
parser.add_argument('--negative_sample_ratio_context', type=int, default=2)
parser.add_argument('--alpha', type=float, default=0.2, help='Contribution of the first view')
parser.add_argument('--beta', type=float, default=0.05, help='Contribution of the second view')
parser.add_argument('--ddpm_iterations', type=int, default=500, help='Steps for DDPM generation')
parser.add_argument('--ddpm_dim', type=int, default=128, help='Latent space dimension in DDPM')
args = parser.parse_args()

# Initialize DDPM model
def initialize_ddpm(ddpm_opts, device):
    """
    Initialize the DDPM model with the given options.
    """
    ddpm = DDPM(ddpm_opts).to(device)
    return ddpm

# Initialize GCLAD model
def initialize_gclad(ft_size, embedding_dim, device, ddpm_opts):
    """
    Initialize the GCLAD model with DDPM integration.
    """
    model = GCLAD(
        n_in=ft_size, 
        n_h=embedding_dim, 
        activation='prelu', 
        negsamp_round_patch=args.negative_sample_ratio_patch,
        negsamp_round_context=args.negative_sample_ratio_context,
        readout=args.readout,
        ddpm_opts=ddpm_opts  # Pass the DDPM configuration
    ).to(device)
    return model

# Prepare dataset and preprocessing
def prepare_data(device, args):
    """
    Load and preprocess the dataset.
    """
    adj, features, labels, idx_train, idx_val, idx_test, ano_label, _, _ = load_mat(args.dataset)

    # Preprocess features and adjacency matrix
    features, _ = preprocess_features(features)
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    # Convert to tensors
    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    return adj, features, labels, idx_train, idx_val, idx_test, ano_label

# Compute GCLAD and DDPM losses
def compute_losses(logits_1, logits_2, logits_1_hat, logits_2_hat, lbl_context, lbl_patch, device):
    """
    Compute the contrastive loss for GCLAD and DDPM.
    """
    b_xent_patch = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negative_sample_ratio_patch]).to(device))
    b_xent_context = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negative_sample_ratio_context]).to(device))

    loss_all_1 = b_xent_context(logits_1, lbl_context)
    loss_all_1_hat = b_xent_context(logits_1_hat, lbl_context)
    loss_1 = torch.mean(loss_all_1)
    loss_1_hat = torch.mean(loss_all_1_hat)

    loss_all_2 = b_xent_patch(logits_2, lbl_patch)
    loss_all_2_hat = b_xent_patch(logits_2_hat, lbl_patch)
    loss_2 = torch.mean(loss_all_2)
    loss_2_hat = torch.mean(loss_all_2_hat)

    loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat
    loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat
    loss = args.beta * loss_1 + (1 - args.beta) * loss_2

    return loss, loss_1, loss_2

# Perform one training step
def training_step(model, ddpm, features, adj, labels, device, optimiser, subgraph_size):
    """
    Perform one training step, including both DDPM and GCLAD components.
    """
    model.train()
    optimiser.zero_grad()

    # Use DDPM to enhance features
    ddpm.feed_data({'SR': features, 'ORI': adj})
    ddpm.optimize_parameters()
    enhanced_features = ddpm.sample(batch_size=features.size(0))

    # Forward pass through GCLAD
    logits_1, logits_2, subgraph_embed, node_embed = model(enhanced_features, adj)

    # Compute losses
    lbl_patch = torch.unsqueeze(torch.cat((torch.ones(features.size(0)), torch.zeros(features.size(0) * args.negative_sample_ratio_patch))), 1).to(device)
    lbl_context = torch.unsqueeze(torch.cat((torch.ones(features.size(0)), torch.zeros(features.size(0) * args.negative_sample_ratio_context))), 1).to(device)

    loss, loss_1, loss_2 = compute_losses(logits_1, logits_2, logits_1, logits_2, lbl_context, lbl_patch, device)

    ddpm_loss = ddpm.loss(subgraph_embed, subgraph_embed)  # Assuming ddpm.loss is implemented
    total_loss = loss + ddpm_loss

    total_loss.backward()
    optimiser.step()

    return total_loss.item(), loss_1.item(), loss_2.item()

# Testing function
def test_model(model, adj, features, labels, idx_test, device, batch_size, num_rounds, subgraph_size):
    """
    Run testing on the model to evaluate AUC performance.
    """
    model.eval()

    # Prepare test set
    multi_round_ano_score = np.zeros((num_rounds, len(idx_test)))

    with torch.no_grad():
        for round in range(num_rounds):
            all_idx = list(idx_test)
            random.shuffle(all_idx)
            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            # Prepare batch number for testing
            batch_num = len(all_idx) // batch_size + 1
            for batch_idx in range(batch_num):
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba = []
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, features.shape[2])).to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    ba_hat.append(cur_adj_hat)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_hat = torch.cat(ba_hat)
                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                # Get the output logits for anomaly detection
                test_logits_1, test_logits_2, _, _ = model(bf, ba)
                test_logits_1_hat, test_logits_2_hat, _, _ = model(bf, ba_hat)

                # Apply sigmoid to get probabilities
                test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))
                test_logits_1_hat = torch.sigmoid(torch.squeeze(test_logits_1_hat))
                test_logits_2_hat = torch.sigmoid(torch.squeeze(test_logits_2_hat))

                # Calculate anomaly score for each node in the test batch
                ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                    cur_batch_size, args.negative_sample_ratio_context), dim=1)).cpu().numpy()
                ano_score_1_hat = - (
                            test_logits_1_hat[:cur_batch_size] - torch.mean(test_logits_1_hat[cur_batch_size:].view(
                        cur_batch_size, args.negative_sample_ratio_context), dim=1)).cpu().numpy()
                ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                    cur_batch_size, args.negative_sample_ratio_patch), dim=1)).cpu().numpy()
                ano_score_2_hat = - (
                            test_logits_2_hat[:cur_batch_size] - torch.mean(test_logits_2_hat[cur_batch_size:].view(
                        cur_batch_size, args.negative_sample_ratio_patch), dim=1)).cpu().numpy()

                # Combine anomaly scores from both contrastive losses
                ano_score = args.beta * (args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_1_hat)  + \
                            (1 - args.beta) * (args.alpha * ano_score_2 + (1 - args.alpha) * ano_score_2_hat)

                multi_round_ano_score[round, idx] = ano_score

            # Update progress bar
            pbar_test.update(1)

    # Compute final AUC score based on multi-round anomaly scores
    ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
    return ano_score_final

def test(model, adj, features, labels, idx_test, device, batch_size, num_rounds, subgraph_size, ano_label):
    """
    Perform testing and compute the AUC for anomaly detection.
    """
    # Get final anomaly scores from testing
    ano_score_final = test_model(model, adj, features, labels, idx_test, device, batch_size, num_rounds, subgraph_size)

    # Calculate AUC score
    auc = roc_auc_score(ano_label, ano_score_final)
    print(f'Testing AUC: {auc:.4f}')
    return auc

# Main function to train and test the model
def main():
    """
    Main function to train and test the model.
    """
    device = torch.device(args.gpu_device if torch.cuda.is_available() else 'cpu')

    # Initialize DDPM and GCLAD models
    ddpm_opts = {
        'model': {'beta_schedule': {'train': 'linear'}},
        'train': {'optimizer': {'lr': args.lr}},
        'path': {'resume_state': None},
        'phase': 'train'
    }
    ddpm = initialize_ddpm(ddpm_opts, device)
    model = initialize_gclad(ft_size, args.embedding_dim, device, ddpm_opts)

    # Optimizer and data preparation
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    adj, features, labels, idx_train, idx_val, idx_test, ano_label = prepare_data(device, args)

    all_auc = []
    for epoch in range(args.epochs):
        total_loss, loss_1, loss_2 = training_step(model, ddpm, features, adj, labels, device, optimiser, args.subgraph_size)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Loss1: {loss_1:.4f}, Loss2: {loss_2:.4f}")

    # Testing phase
    auc = test(model, adj, features, labels, idx_test, device, args.mini_batch, args.test_rounds, args.subgraph_size, ano_label)
    all_auc.append(auc)

    print('\nFinal Testing AUC: {:.4f}'.format(np.mean(all_auc)))

if __name__ == '__main__':
    main()

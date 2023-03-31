import os
import argparse
from time import time
from tqdm import tqdm

import dgl.function as fn

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import Data
from models import CompGCN_DistMult
from utils import in_out_norm

import heapq
from collections import defaultdict as ddict

from utils import save_to_json


def get_metrics(pred, gt_list, train_list, top_k):
    metrics = ['HITS', 'MRR']
    result = {metric: 0.0 for metric in metrics}
    pred = np.array(pred)
    pred[train_list] = -np.inf
    pred_dict = {idx: score for idx, score in enumerate(pred)}
    pred_dict = heapq.nlargest(top_k, pred_dict.items(), key=lambda kv :kv[1])
    pred_list = [k for k,v in pred_dict]

    for gt in gt_list:
        if gt in pred_list:
            index = pred_list.index(gt)
            result['MRR'] = max(result['MRR'], 1/(index+1))
            result['HITS'] = 1
    return result


def get_candidate_voter_list(model, graph, device, data, submit_path, top_k):
    print('Start inference...')
    model.eval()
    all_triples, all_preds, all_ids = [], [], []
    submission = []
    with th.no_grad():
        test_iter = iter(data.data_iter['test'])
        for step, (triples, trp_ids) in tqdm(enumerate(test_iter)):
            triples = triples.to(device)
            sub, rel, obj = (
                triples[:, 0],
                triples[:, 1],
                triples[:, 2],
            )
            preds = model(graph, sub, rel) # (batch_size, num_ent)

            triples = triples.cpu().tolist()
            preds = preds.cpu().tolist()
            ids = trp_ids.cpu().tolist()

            for (triple, pred, triple_id) in zip(triples, preds, ids):
                s, r, _ = triple
                train_set = data.sr2o['train'][(s, r)]
                train_set.update(data.sr2o['valid'][(s, r)])
                train_list = np.array(list(train_set), dtype=np.int64)

                pred = np.array(pred)
                pred[train_list] = -np.inf
                pred_dict = {idx: score for idx, score in enumerate(pred)}
                candidate_voter_dict = heapq.nlargest(top_k, pred_dict.items(), key=lambda kv :kv[1])
                candidate_voter_list = [data.id2ent[k] for k,v in candidate_voter_dict]
                
                submission.append({
                    'triple_id': '{:04d}'.format(triple_id[0]),
                    'candidate_voter_list': candidate_voter_list
                })
    save_to_json(submit_path, submission)
        

def evaluate(model, graph, device, data, top_k=5):
    model.eval()
    results = ddict(list)
    all_triples, all_preds = [], []
    with th.no_grad():
        test_iter = iter(data.data_iter['valid'])
        for step, (triples, trp_ids) in enumerate(test_iter):
            triples = triples.to(device)
            sub, rel, obj = (
                triples[:, 0],
                triples[:, 1],
                triples[:, 2],
            )
            preds = model(graph, sub, rel) # (batch_size, num_ent)
            all_triples += triples.cpu().tolist()
            all_preds += preds.cpu().tolist()

    for triple, pred in zip(all_triples, all_preds):
        s, r, _ = triple
        gt_set = data.sr2o['valid'][(s, r)]
        train_set = data.sr2o['train'][(s, r)]
        train_list = np.array(list(train_set), dtype=np.int64)
        gt_list = np.array(list(gt_set), dtype=np.int64)
        result = get_metrics(pred, gt_list, train_list, top_k)
        for k,v in result.items():
            results[k].append(v)

    results = {k: np.mean(v)  for k,v in results.items()}
    return results


def main(args):
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    data = Data(args.data_dir, args.num_workers, args.batch_size)
    data_iter = data.data_iter
    graph = data.g.to(device)
    num_rel = th.max(graph.edata["etype"]).item() + 1

    graph = in_out_norm(graph)

    compgcn_model = CompGCN_DistMult(
        num_bases=args.num_bases,
        num_rel=num_rel,
        num_ent=graph.num_nodes(),
        in_dim=args.init_dim,
        layer_size=args.layer_size,
        comp_fn=args.opn,
        batchnorm=True,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
    )
    compgcn_model = compgcn_model.to(device)

    loss_fn = th.nn.BCELoss()
    optimizer = optim.Adam(compgcn_model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_epoch = -1
    best_mrr = 0.0
    kill_cnt = 0
    submit_path = "{}/preliminary_submission.json".format(args.output_dir)
    
    print('****************************')
    print('Start training...')
    for epoch in range(args.max_epochs):
        compgcn_model.train()
        train_loss = []
        t0 = time()
        for step, batch in enumerate(data_iter['train']):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label.squeeze(),
            )
            logits = compgcn_model(graph, sub, rel, obj)
            
            tr_loss = loss_fn(logits, label)
            train_loss.append(tr_loss.item())

            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        train_loss = np.sum(train_loss)

        if (epoch + 1) % 20 == 0:
            t1 = time()
            val_results = evaluate(compgcn_model, graph, device, data, top_k=5)
            t2 = time()

            if val_results["MRR"] > best_mrr:
                best_mrr = val_results["MRR"]
                best_epoch = epoch
                th.save(compgcn_model.state_dict(), "{}/baseline_ckpt.pth".format(args.ckpt_dir))
                kill_cnt = 0
                print("Saving model...")
            else:
                kill_cnt += 1
                if kill_cnt > 7:
                    print("Early stop. Best MRR {} at Epoch".format(best_mrr, best_epoch))
                    break
            print("In Epoch {}, Train Loss: {:.4f}, Valid MRR: {:.5}, Valid HITS: {:.5}, Train Time: {:.2f}, Valid Time: {:.2f}".format(
                    epoch, train_loss, val_results["MRR"],  val_results["HITS"], t1 - t0, t2 - t1))
                    
        else:
            t1 = time()
            print("In Epoch {}, Train Loss: {:.4f}, Train Time: {:.2f}".format(epoch, train_loss, t1 - t0))

    compgcn_model.eval()
    compgcn_model.load_state_dict(th.load("{}/baseline_ckpt.pth".format(args.ckpt_dir)))
    get_candidate_voter_list(compgcn_model, graph, device, data, submit_path, top_k=5)
    print("Submission file has been saved to: {}.".format(submit_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--data_dir", dest="data_dir", default="./data", help="Dataset dir", )
    parser.add_argument("--output_dir", dest="output_dir", default="./output", help="Output dir", )
    parser.add_argument("--ckpt_dir", dest="ckpt_dir", default="./checkpoint", help="Checkpoint dir", )
    parser.add_argument("--opn", dest="opn", default="ccorr", help="Composition Operation to be used in CompGCN", )
    parser.add_argument("--batch", dest="batch_size", default=16388, type=int, help="Batch size" )
    parser.add_argument("--gpu", type=int, default=0, help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0", )
    parser.add_argument("--epoch", dest="max_epochs", type=int, default=1000, help="Number of epochs",)
    parser.add_argument("--l2", type=float, default=0.0, help="L2 Regularization for Optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="Starting Learning Rate")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of processes to construct batches",)
    parser.add_argument("--seed", dest="seed", default=41504, type=int, help="Seed for randomization", )
    parser.add_argument("--num_bases", dest="num_bases", default=-1, type=int, help="Number of basis relation vectors to use", )
    parser.add_argument("--init_dim", dest="init_dim", default=50, type=int, help="Initial dimension size for entities and relations", )
    parser.add_argument("--layer_size", nargs="?", default="[50]", help="List of output size for each compGCN layer", )
    parser.add_argument("--gcn_drop", dest="dropout", default=0.1, type=float, help="Dropout to use in GCN Layer", )
    parser.add_argument("--layer_dropout", nargs="?", default="[0.3]", help="List of dropout value after each compGCN layer", )

    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    args.layer_size = eval(args.layer_size)
    args.layer_dropout = eval(args.layer_dropout)

    main(args)
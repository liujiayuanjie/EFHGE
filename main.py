import warnings
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from utils import set_seed, divide, Timer, Maximizer, Averager, Batcher
from data import load_data, create_hypergraph_data, sample_evo_flow
from model import HyEdgeEmb, VertexEmb, TimeEmb, HyGAttEmb, EvoFlowEmb
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from task import LogisticRegression, kmeans

if __name__ == '__main__':
    # initialize parameters
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('--quit_epoch_num', type = int, default = 10)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--data_path', type = str, default = '../../dataset')
    parser.add_argument('--data_name', type = str, default = 'assist2009')
    parser.add_argument('--trn_ratio', type = float, default = 0.8)
    parser.add_argument('--trn_seed', type = int, default = 0)
    parser.add_argument('--emb_dim', type = int, default = 64)
    parser.add_argument('--batch_size', type = int, default = 8000)
    parser.add_argument('--lr', type = float, default = 0.05)
    parser.add_argument('--weight_decay', type = float, default = 0.01)
    parser.add_argument('--neg_num', type = int, default = 8)
    parser.add_argument('--layer_num', type = int, default = 2)
    parser.add_argument('--ev_tau', type = float, default = 0.1)
    parser.add_argument('--ve_tau', type = float, default = 1.0)
    parser.add_argument('--r', type = int, default = 2)
    parser.add_argument('--lamb', type = float, default = 0.9)
    parser.add_argument('--print_detail', action = 'store_true')
    parser.add_argument('--print_result', action = 'store_true')
    parser.add_argument('--save_path', type = str)
    args, _ = parser.parse_known_args()
    # initialize data
    set_seed(args.seed)
    device = torch.device(args.device)
    path = '%s/%s' % (args.data_path, args.data_name)
    qst_num, usr_num, skl_num, qst_skl, events = load_data(path)
    eve_num = events.shape[0]

    v_num, e_num, vv, ve, tme, qst_seqs, usr_seqs = create_hypergraph_data(qst_num, usr_num, events)
    ve_tsr = torch.tensor(ve, device = device)
    tme = torch.tensor(tme, device = device)
    
    qst_msk = qst_skl != -1
    qst_skl = torch.tensor(qst_skl, device = device).long()
    qst_skl = torch.tensor(qst_skl, device = device).long()
    qsts = np.arange(qst_num)
    trn_qsts, evl_qsts = divide(qsts[qst_msk], args.trn_ratio, args.trn_seed)
    trn_qsts = torch.tensor(trn_qsts, device = device).long()
    evl_qsts = torch.tensor(evl_qsts, device = device).long()

    qst_ef = sample_evo_flow(v_num, qst_seqs, args.r)
    usr_ef = sample_evo_flow(v_num, usr_seqs, args.r)
    qst_ef = torch.tensor(qst_ef, device = device)
    usr_ef = torch.tensor(usr_ef, device = device)
    # initialize model
    v_embed = HyEdgeEmb(v_num, args.emb_dim).to(device)
    e_embed = VertexEmb(e_num, args.emb_dim).to(device)
    t_embed = TimeEmb(args.emb_dim).to(device)
    hygatt_embed = HyGAttEmb(v_num, e_num, args.emb_dim, args.layer_num, args.ve_tau, args.ev_tau).to(device)
    evoflw_embed = EvoFlowEmb(args.emb_dim, 2, args.layer_num).to(device)

    v_params = {'params': v_embed.parameters(), 'weight_decay': args.weight_decay}
    e_params = {'params': e_embed.parameters(), 'weight_decay': args.weight_decay}
    t_params = {'params': t_embed.parameters()}
    hygatt_params = {'params': hygatt_embed.parameters()}
    evoflw_params = {'params': evoflw_embed.parameters()}

    parameters = [v_params, e_params, t_params, hygatt_params, evoflw_params]
    optimizer = torch.optim.Adam(parameters, lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    # get feature
    def get_fea():
        v_emb = v_embed()
        e_emb = e_embed()
        t_emb = t_embed(tme)
        v_fea, e_fea = hygatt_embed(v_emb, t_emb, e_emb, ve_tsr)
        ef_fea = evoflw_embed(v_emb, t_emb, (qst_ef, usr_ef))
        return v_fea, e_fea, ef_fea, e_emb
    # prepare train
    def pretrain():
        v_embed.train()
        e_embed.train()
        t_embed.train()
        hygatt_embed.train()
        evoflw_embed.train()
    # prepare evaluate
    def preeval():
        v_embed.eval()
        e_embed.eval()
        t_embed.eval()
        hygatt_embed.eval()
        evoflw_embed.eval()
    # train
    def train():
        pretrain()

        ef_loss_avg = Averager()
        ef_batcher = Batcher(vv, args.batch_size)
        
        hg_loss_avg = Averager()
        hg_batcher = Batcher(ve, args.batch_size)
        
        for ef_batch, hy_batch in zip(ef_batcher, hg_batcher):
            v_fea, e_fea, ef_fea, e_emb = get_fea()
            
            ef_loss = get_ef_loss(v_fea, ef_batch)
            ef_loss_avg.join(ef_loss.item() / ef_batch.shape[0])
            
            hg_loss = get_hg_loss(ef_fea, e_emb, hy_batch)
            hg_loss_avg.join(hg_loss.sum().item() / hy_batch.shape[0])
            
            loss = args.lamb * ef_loss.sum() + (1 - args.lamb) * hg_loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return ef_loss_avg.get(), hg_loss_avg.get()
    # get evolving flow loss
    def get_ef_loss(v_emb, batch):
        v1, v2 = torch.tensor(batch, device = device).T

        v1_emb = v_emb[v1]
        v2_emb = v_emb[v2]

        bn = v1.size(0)
        nn = args.neg_num

        neg = torch.randint(v_num, size = (bn, nn), device = device)
        neg_emb = v_emb[neg]
        
        pos = (v1_emb * v2_emb).sum(-1).unsqueeze(1)
        neg = (v1_emb.unsqueeze(1) * neg_emb).sum(-1)
           
        x = torch.cat((pos, neg), dim = 1)
        y = torch.tensor([0] * bn, device = device)

        loss = criterion(x, y) * bn

        return loss 
    # get hypergraph loss
    def get_hg_loss(v_fea, e_fea, hy_batch):
        v, e = torch.tensor(hy_batch, device = device).T

        e_fea_ = e_fea[e]
        v_fea_ = v_fea[v]

        bn = v.size(0)
        nn = args.neg_num 

        neg_e = torch.randint(e_num, size = (bn, nn), device = device)
        neg_e_fea = e_fea[neg_e]
        
        pos = (e_fea_ * v_fea_).sum(-1).unsqueeze(1)
        neg = (v_fea_.unsqueeze(1) * neg_e_fea).sum(-1)  
          
        x = torch.cat((pos, neg), dim = 1)
        y = torch.tensor([0] * bn, device = device) 
        
        loss = criterion(x, y) * bn
        
        return loss
    # evaluate
    def evaluate(times = 4):
        preeval()
        
        v_fea, e_fea, ef_fea, e_emb = get_fea()
        
        qst_fea = e_fea.detach()[: qst_num]

        x_input = qst_fea[trn_qsts]
        model = LogisticRegression(x_input.size(-1), skl_num, device = device)
        model.fit(x_input, qst_skl[trn_qsts])

        y_pred = model.predict(x_input)
        trn_mi_f1 = f1_score(qst_skl[trn_qsts].cpu().numpy(), y_pred.cpu().numpy(), average = 'micro')

        x_input = qst_fea[evl_qsts]
        y_pred = model.predict(x_input)
        ma_f1 = f1_score(qst_skl[evl_qsts].cpu().numpy(), y_pred.cpu().numpy(), average = 'macro')
        mi_f1 = f1_score(qst_skl[evl_qsts].cpu().numpy(), y_pred.cpu().numpy(), average = 'micro')
        
        labels = qst_skl.cpu().numpy()[qst_msk]
        x_input = qst_fea[qsts[qst_msk]]
        nmi, ari = 0, 0
        for i in range(times):
            labels_ = kmeans(x_input, skl_num).cpu().numpy()
            nmi += normalized_mutual_info_score(labels, labels_)
            ari += adjusted_rand_score(labels, labels_)
        nmi /= times
        ari /= times

        return trn_mi_f1, ma_f1, mi_f1, nmi, ari
    # conduct experiments
    epoch = 1
    quit_count = 0

    mi_f1_max = Maximizer()
    max_ma_f1 = 0.0
    
    ari_max = Maximizer()
    max_nmi = 0.0
    
    dur_avg = Averager()

    while quit_count <= args.quit_epoch_num:
        timer = Timer()
        ef_loss, hy_loss = timer(train)
        trn_mi_f1, ma_f1, mi_f1, nmi, ari = timer(evaluate)

        if mi_f1_max.join(mi_f1):
            max_ma_f1 = ma_f1
            quit_count = 0
        
        if ari_max.join(ari):
            max_nmi = nmi
            quit_count = 0

            if args.save_path != None:
                v_fea, e_fea, ef_fea, e_emb = get_fea()
                qst_embedding = e_fea.detach().cpu().numpy()[: qst_num]

        if args.print_detail:
            print('  '.join((
                'epoch: %-4d' % epoch,
                'ef_loss: %-.4f' % ef_loss,
                'hy_loss: %-.4f' % hy_loss,
                'trn_mi_f1: %-.4f' % trn_mi_f1,
                'ma_f1: %-.4f/%-.4f' % (ma_f1, max_ma_f1),
                'mi_f1: %-.4f/%-.4f' % (mi_f1, mi_f1_max.get()),
                'nmi: %-.4f/%-.4f' % (nmi, max_nmi),
                'ari: %-.4f/%-.4f' % (ari, ari_max.get()),
                'dur: %-.2fs' % timer.get(),
            )))
        
        dur_avg.join(timer.get())
        epoch += 1
        quit_count += 1

    if args.print_result:
        print('%.4f' % dur_avg.get())
        print('%.4f' % (int(torch.cuda.max_memory_allocated()) / 1024 ** 3))
        print('%.4f' % max_ma_f1)
        print('%.4f' % mi_f1_max.get())
        print('%.4f' % max_nmi)
        print('%.4f' % ari_max.get())
    
    if args.save_path != None:
        np.save(args.save_path, qst_embedding)
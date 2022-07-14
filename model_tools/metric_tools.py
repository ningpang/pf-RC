def get_acc(grnd, pred):
    total = len(grnd)
    correct = 0
    for id in range(len(grnd)):
        if grnd[id] == pred[id]:
            correct += 1
    return correct/total

def get_f1(grnd, pred, na_num):
    pos_crt = 0
    pos_pre = 1e-6
    pos_tot = 0

    for id in range(len(grnd)):
        if grnd[id] != na_num:
            pos_tot += 1
        if pred[id] != na_num:
            pos_pre += 1
        if grnd[id] != na_num and grnd[id] == pred[id]:
            pos_crt += 1
    rec = pos_crt/pos_tot
    prec = pos_crt/pos_pre
    f1 = 2*prec*rec/(prec+rec+1e-6)
    return f1

def get_f1_by_relation(grnd, pred, rel2id):
    pos_crt = {}
    pos_pre = {}
    pos_tot = {}
    rec = {}
    prec = {}
    f1 = {}
    for i in range(len(rel2id)):
        pos_crt[i] = 0
        pos_pre[i] = 1e-6
        pos_tot[i] = 1e-6

    for id in range(len(grnd)):
        # print(grnd)
        pos_tot[grnd[id]] += 1
        if grnd[id] == pred[id]:
            pos_crt[grnd[id]] += 1
        pos_pre[pred[id]] += 1

    for id in range(len(rel2id)):
        rec[id] = pos_crt[id]/pos_tot[id]
        prec[id] = pos_crt[id]/pos_pre[id]
        f1[id] = 2*prec[id]*rec[id]/(prec[id]+rec[id]+1e-6)

    return f1

def get_acc_by_relation(grnd, pred, rel2id):
    pos_crt = {}
    pos_tot = {}
    acc = {}

    for i in range(len(rel2id)):
        pos_crt[i] = 0
        pos_tot[i] = 0

    for id in range(len(grnd)):
        # print(grnd)
        pos_tot[grnd[id]] += 1
        if grnd[id] == pred[id]:
            pos_crt[grnd[id]] += 1

    print(pos_tot)
    for id in range(len(rel2id)):

        acc[id] = pos_crt[id]/pos_tot[id]
    return acc



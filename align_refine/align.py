from spoa import poa
if __name__ == '__main__':
    cluster_dict  = {}
    with open("../data/datap0.03") as cluster_reader:
        for strand in range(0, 65535):
            cluster_reader.readline()
            cluster = []
            for i in range(10):
                line = cluster_reader.readline().strip()
                cluster.append(line)
            cluster_dict[strand] = cluster

    cluster_writer =  open("aligned_dataset.txt", "w+")

    cluster_aligned = {}
    for cluster_id, cluster in cluster_dict.items():
        c, cluster_aligned[cluster_id] = poa(cluster, algorithm=1)

    strand_to_unkown = {}
    new_clusters = {}
    for cluster_id, cluster in cluster_aligned.items():
        print(cluster_id),
        cluster_writer.write("CLUSTER"+str(cluster_id)+'\n')
        length = len(cluster[0])
        print(length),
        pos_to_unkown_num = {}
        for s in cluster:
            print(s),
        for pos in range(length):
            unkown_num = 0
            for strand in cluster:
                if strand[pos] == '-':
                    unkown_num = unkown_num + 1
            pos_to_unkown_num[pos] = unkown_num
        a = sorted(pos_to_unkown_num.items(), key=lambda kv:(kv[1], kv[0]), reverse = True)
        print(a)
        skip_index_num = length - 121
        skip_index = []
        loop = 0
        for pos, num in a:
            if loop > skip_index_num:
                break
            skip_index.append(pos)
            loop = loop + 1
        new_cluster = []
        for i in range(len(cluster)):
            new_cluster.append("")
        for pos in range(0, length):
            if pos in skip_index:
                continue
            for id in range(len(cluster)):
                new_cluster[id] = new_cluster[id]+cluster[id][pos]

        for cc in new_cluster:
            cluster_writer.write(cc+"\n")




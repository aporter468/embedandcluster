mm="min"
d=10
w=5
l=10
p=1
q=1
cnum=0
k=4
v=1
kdir="version"$v"_clustersrepeat_k"$k
mkdir $kdir
for cnum in 0 1 2 3 4 5
do
	name="amazon-meta-"$mm"10krank_cluster"$cnum"_version"$v
	outemd="results/"$name"_"$d"_"$w"_"$l"_"$p"_"$q".emd"
	anomlist="anoms.out"
	plotdir=$kdir"/catlist-rank"$mm"_"$name"_"$d"_"$w"_"$l"_"$p"_"$q
	echo $plotdir
	mkdir $plotdir
	ingraph="data/amazon-meta-catlist-rank"$mm"-edges.txt"
	labels="data/amazon-meta-catlist-rank"$mm"-nodes.txt"
	clusters=$plotdir"/cluster"
	echo $labels
	python kmeans_node2vec_friedman.py $outemd $anomlist $plotdir $ingraph $labels $clusters $k

	cat anoms.out | sort -n -k2,2 > $kdir"/cluster"$cnum"_anoms.txt"
done



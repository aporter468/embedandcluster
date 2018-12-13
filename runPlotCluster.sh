mm="min"
d=10
w=10
l=10
p=1
q=0.1
k=4
name="amazon-meta-"$mm"10krank"
outemd="results/"$name"_"$d"_"$w"_"$l"_"$p"_"$q".emd"
anomlist="anoms.out"
plotdir="catlist-rank"$mm"_"$name"_"$d"_"$w"_"$l"_"$p"_"$q
echo $plotdir
mkdir $plotdir
ingraph="data/amazon-meta-catlist-rank"$mm"-edges.txt"
labels="data/amazon-meta-catlist-rank"$mm"-nodes.txt"
clusters=$plotdir"/cluster"
echo $labels
python kmeans_node2vec_friedman.py $outemd $anomlist $plotdir $ingraph $labels $clusters $k




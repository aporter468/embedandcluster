v=1
k=4
anomlist="version"$v"_clustersrepeat_k"$k
catliststr="catlist-rankmin_amazon-meta-min10krank_"
for c in 0 1 2 3 4 5
do
#	join -1 1 -2 1  $anomlist$c"_anoms.txt" data/amazon-meta-catlist-rankmin-nodes.txt > $anomlist$c"_names.txt"
	innerdir=$anomlist"/"$catliststr"cluster"$c"_version"$v"_10_5_10_1_1"
	for c2 in 0 1 2 3
	do
		cfile=$innerdir"/cluster_"$c2".txt"
		names=$innerdir"/cluster_"$c2"_names.txt"
		sorted=$innerdir"/cluster_"$c2"_sorted.txt"
		awk 'NR==FNR {h[$1]=1; a[$1]=$0; next} {if (h[$1]) print a[$1]" " $0}' $cfile  "data/amazon-meta-catlist-rankmin-nodes.txt" >  $names
	   sort -n -k2,2 $names  > $sorted

	done   
done

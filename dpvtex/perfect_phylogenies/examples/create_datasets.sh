for i in 1000; do
  for j in 200; do
    echo Generate dataset with $j trees on $i leaves.
    python make_datasets.py -t $j -l $i --n_phylos 1 --n_threads 4 --depth 3 --split_data False
    python make_datasets.py -t $j -l $i --n_phylos 1 --n_threads 4 --spr True --split_data False
    done
  done

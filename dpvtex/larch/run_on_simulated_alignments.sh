for file in configs/*; do
   echo $file
   snakemake -c1 --configfile $file --rerun-incomplete
done

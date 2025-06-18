for file in configs_paper/*; do
   echo $file
   snakemake -c10 --configfile $file --rerun-incomplete
done

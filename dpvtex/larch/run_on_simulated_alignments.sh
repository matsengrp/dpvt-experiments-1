for file in configs/*; do
   echo $file
   snakemake -c10 --configfile $file 
done

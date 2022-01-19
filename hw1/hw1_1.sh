wget https://www.dropbox.com/s/it1ht3wr59vxiu5/best_model_p1.mdl?dl=1 -O best_model_p1.mdl
python3 ./p1_src/inference.py $1 best_model_p1.mdl 50 $2 
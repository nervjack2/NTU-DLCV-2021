wget https://www.dropbox.com/s/huqiagmi2fnj41s/best_generator_p1.mdl?dl=1 -O best_generator_p1.mdl
python3 ./p1_src/inference.py $1 './best_generator_p1.mdl'

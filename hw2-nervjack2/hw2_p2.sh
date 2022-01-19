wget https://www.dropbox.com/s/q3hry6poy30f1qw/best_generator_p2.mdl?dl=1 -O best_generator_p2.mdl
python3 ./p2_src/inference.py $1 './best_generator_p2.mdl'

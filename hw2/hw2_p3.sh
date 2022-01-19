echo $2
if [ $2 == "mnistm" ]
then 
    wget https://www.dropbox.com/s/cnzojgs3w4jrb47/best_classifier_mnistm.mdl?dl=1 -O best_classifier_mnistm.mdl
    python3 ./p3_src/p3_2_src/inference.py $1 $2 $3 './best_classifier_mnistm.mdl'
elif [ "$2" == "usps" ]
then
    wget https://www.dropbox.com/s/ayavbgn64rvu907/best_classifier_usps.mdl?dl=1 -O best_classifier_usps.mdl
    python3 ./p3_src/p3_2_src/inference.py $1 $2 $3 './best_classifier_usps.mdl'
elif [ "$2" == "svhn" ]
then
    wget https://www.dropbox.com/s/h111taaw7dtc2ow/best_classifier_svhn.mdl?dl=1 -O best_classifier_svhn.mdl
    python3 ./p3_src/p3_2_src/inference.py $1 $2 $3 './best_classifier_svhn.mdl'
else 
    echo "Target domain can't be rocognized."
fi 


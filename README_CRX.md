# How to use this project 

## Download matlab models
Make a directory 'models', download model files to this folder. 
### for gray model
wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17.net.mat -P models/

### for color+gray model 
wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17_gray025.net.mat -P models/

## Convert matlab models to pytorch models
run the following command:

python bin/convert_pretrained_model.py 

Afterward, a pytroch model 'siamfc_pretrained.pth' is generated in the folder 'models'.


# pySuperResolution
Implementations Of Super-Resolution Methods especially in IR image domain

This repository will be use pytorch 1.4. 

### Architecture
|   Folders   	|                         Functionalities                        	|                                    Details                                    	|
|:-----------:	|:--------------------------------------------------------------:	|:-----------------------------------------------------------------------------:	|
|    utils    	| It includes basic utilities like calculating PSNR, SSIM        	| Currently Only visualization exist.(imshow & metrics)                         	|
| dataloaders 	| It includes dataset class that takes paths that include images 	| Dataset can add noise, blur, random transforms, random crop and desired scale 	|
|      loss     | It includes custom loss functions and some basic loss functions   |                                                                                	|
|    model      | It includes network models                                      	|                                                                               	|
|   method    	| It defines how the model trained & tested                      	|                                                                               	|

- **main.py** is an example of how the architecture can be used to train or test desired model.
- **options.py** includes input parameters that can be easily set or can be loaded from desired config file.
- **demo.py** It will be written to show samples of trained models.

### Models

|   Model Name  |   Training Status     |  Testing Status    |
|:-----------:	|:-------------------:	|:-----------------: |
|    EDSR    	| Done       	        | Done               |
|    MDSR    	| Waiting       	    | Waiting            |
|    DDBPN    	| Waiting       	    | Waiting            |
|    RCAN    	| Waiting       	    | Waiting            |
|    RDN    	| Waiting       	    | Waiting            |
|    VDSR    	| Waiting       	    | Waiting            |
|    PFF    	| Done       	        | Done              |


## How to add a new SR method?

TODOs:
- [ ] A figure that explain architecture.
- [ ] add explanation of usage of basic blocks 
- [ ] a picture can be used to find SR
- [ ] a folder that contains test images is taken and their SR versions are saved a desired folder
# pySuperResolution
Implementations Of Super-Resolution Methods especially in IR image domain

This repository will be use pytorch 1.4. 

### Architecture
|   Folders   	|                         Functionalities                        	|                                    Details                                    	|
|:-----------:	|:--------------------------------------------------------------:	|:-----------------------------------------------------------------------------:	|
|    utils    	| It includes basic utilities like calculating PSNR, SSIM        	| Currently Only visualization exist.(imshow & metrics)                         	|
| dataloaders 	| It includes dataset class that takes paths that include images 	| Dataset can add noise, blur, random transforms, random crop and desired scale 	|
|             	|                                                                	|                                                                               	|

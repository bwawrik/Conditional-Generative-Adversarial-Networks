# Conditional-Generative-Adversarial-Networks
So I was bored and wanted to see if I could use my desktop to make fake pictures of clouds. The way to go seems to be a conditional generative adversarial neural network (cGANN).

Some Notes

- I have an I9-10900 CPU with an RTX 2060 Super; 32GB RAM
- The largest images I could process were 100x100 pixels; anything larger made my machine croak
- my training set are images of clouds scraped from google using the key word 'cloud'. I don't include the code how to do this here
- I deleted all images that contained text
- a manually modified a few images to exclude trees or the ground
- runtime for 75 epochs was at ~8 hours

![clouds](https://github.com/bwawrik/Conditional-Generative-Adversarial-Networks/blob/main/clouds_epoch100.png)

This script is loosely based on the following tutorial - Kudos to the author:

https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

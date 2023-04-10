# yates-beyond-fixation
Matlab and python code to support the methods in Yates et al., 2023

The focus of the manuscript is a set of hardware and software solutions to study of neural responses during natural oculomotor. Specifically, offline gaze-contingent analyses are used to build a reconstruction of the visual input after correcting for eye movements. 

The general idea can be seen in Figure 1b. Stimuli are reconstructed within a gaze-contingent window and that pixel-resolution movie is used to do model and study visual responses.
<img width="691" alt="image" src="https://user-images.githubusercontent.com/1760049/230817274-c15e221b-c945-4e03-bf05-227fd10eedc8.png">

In this repository, the matlab script `hires_demo.m` will demonstrate finding the receptive field locations coarsely, creating a region of interest (ROI), and reconstructing a pixel-resolution movie of the stimulus withing the gaze-contingent ROI. 

After exporting a stimulus movie, the shifter model can be fit. The shifter model fits a neural network model to the spiking data. The components of the model capture the stimulus processing and a correction grid for calibrating the eye tracking.
<img width="974" alt="image" src="https://user-images.githubusercontent.com/1760049/230821427-f2f9d6ad-9949-4cef-8a0f-87211d443b2b.png">

A python notebook [shifter_example](https://github.com/VisNeuroLab/yates-beyond-fixation/blob/main/shifter_example.ipynb) demonstrates the steps for fitting a shifter model.

## getting the data
An example dataset can be downloaded [here](https://doi.org/10.6084/m9.figshare.22580566!). Because we used matlab objects to generate the stimuli and reconstruct from seed, for the dataset to load properly, you will need the stimulus code from [here](https://github.com/jcbyts/MarmoV5) in your path. The supporting code that was used to preprocess the raw files can be found [here](https://github.com/jcbyts/MarmoPipe).

## hires_demo.m
In the manuscript, we used PsychToolbox to draw the stimulus. `hires_demo.m` turns that flag off and uses the equations to generate gabors or clips directly into natural images. This will produce tiny differences in pixel values that likely don't matter, but we did not explore this thoroughly.

After editing the paths in `addFreeViewingPaths`, load the data and then run each cell in succession.

The first step is to find the ROI. This can be done over the entire screen, but it's much faster to restrict to a retinotopic region. Here, we use a 3 d.v.a. window with 0.25 d.v.a. bin sizes. 
```matlab

ROI = [-1 -1 1 1]*3;
binSize = .25;
Frate = 120;
eyeposexclusion = 20;
win = [-1 20];

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', ROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
    'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
    'eyePos', Exp.vpx.smo(:,2:3), 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 2);
        
```
After using forward correlation to find the firing rate map and averaging across all neurons, we get an ROI here:
<img width="513" alt="image" src="https://user-images.githubusercontent.com/1760049/230820451-06e158b4-0b7d-4977-8628-4583b1f7d186.png">


The next step reconstructs each frame of the Gabor stimulus and Natural Images with a gaze-contingent ROI. This is SLOW. After running this, there will be an hdf5 file with the stimulus and spikes saved in it.

The rest of `hires_demo.m` will show some of the logic of shifting. The first step is we calculate the spike-triggered average energy (STE) of the stimulus. This entails squaring each pixel before calculating the STA. Squaring the stimulus produces a good amount of phase invariance and gives a good estimate of the RFs that is somewhat robust to errors in eye tracking. Using this analysis, we can grid up gaze positions on the central 10 degrees of the screen and calculate the STE in a sliding window across space by indexing into timesteps when the monkey was looking at each gaze position.

This analysis shows that the RF centers move around as a function of the gaze vector center:
<img width="836" alt="image" src="https://user-images.githubusercontent.com/1760049/230822186-125b4ef4-b7c5-4786-a179-1cd61650abac.png">

These positions look a lot like the output of the shifter network. This is a simple demonstration of the principles that the big machine learning model is learning: when the calibration is off, the RFs will show up in a shifted location. If you have enough data, you can measure and correct for this.
<img width="552" alt="image" src="https://user-images.githubusercontent.com/1760049/230822245-bdac5a3f-3f15-4c82-99cb-233db6f1a69d.png">


## shifter_example.ipynb
This python notebook shows the steps for fitting the shifter model. We do a few tricks to reduce the computational cost. The first is to crop the stimulus even smaller:
<img width="498" alt="image" src="https://user-images.githubusercontent.com/1760049/230822674-77f804b5-561a-48c2-bcf7-627d85f806eb.png">

After fitting the model, we can look at the performance on the validation set.
<img width="298" alt="image" src="https://user-images.githubusercontent.com/1760049/230822880-a275ff5c-4248-40c2-b96c-471fb302a5c0.png">

And then evaluate the effect of shifting on the STAs.

STAs without shift correction:

<img width="561" alt="image" src="https://user-images.githubusercontent.com/1760049/230823025-27a296a5-d154-4380-9f31-9ea47b1d1ab8.png">

STAs with shift correction:

<img width="555" alt="image" src="https://user-images.githubusercontent.com/1760049/230823069-714d25f4-9958-488c-9c6a-9a44fb2fbc12.png">

If you don't want to use matlab and want to skip the export, email yates@berkeley.edu for the exported stimulus file (it is ~4GB). 








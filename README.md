# FiberLyse
Fiber photometry GUI

In your terminal write; 
 - pip install numpy
 - pip install pandas
 - pip install matplotlib

copy the python code in the FiberLys code and paste in visual studio code, and run it. 

# How to use
1. Upload an CSV file from a neurophotometrics machine
2. Type in the Frequency used data recording
3. Run analysis
4. When fitting the isosbestic signal on the excitatory signal the user can use an interactiv window to change what data is used for the fitting, it deafults to 0s-6500s

# Optional settings 
the "mad" or "sd" is chosing which method is used for artifact removing, its recommended to use mad with a value of 11.90 when doing histamine reading. 
the pad tab, describes how many points adjecent is removed when deleting artifiacts, its recommended to set it to 1. 
The "align" tab described the method at which the isosbestic signal is fitted on the excitatroy signal

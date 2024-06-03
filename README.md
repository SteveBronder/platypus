IPOL Cradle Detection and Removal code package

This program implements an algorithmic solution for detecting and correcting 
the visually displeasing effects of cradling in X-ray images of panel paintings. 
The code is supplied in addition to the IPOL documentation material, available 
online at http://www.ipol.im.

== Authors ==

The program is written by Gabor Fodor <fogggab@yahoo.com>, based heavily on
the Matlab code of Rujie Yin <ry27@duke.edu> and with guidance from Bruno Cornelis
<bcorneli@etro.vub.ac.be> and Ingrid Daubechies <ingrid@math.duke.edu>. 

== License ==

Distributed under the terms of the BSD license. See the file license.txt 
for details.

== Contents ==

img            sample images folder
src			   source code folder
license.txt    license file (GPL v3+)

== Requirements ==

This code needs OpenCV 4+ in order to compile.
Additionally, compiling is recommended with gcc 4.7.2+, older compiler versions 
were not tested.

To obtain OpenCV, please refer to http://opencv.org/
OpenCV is released under a BSD license and hence it’s free for both academic and 
commercial use.

== Compilation ==

Make the executables with the following

```bash
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE
cd build
make -j2 all
```

The executables `Demo`, `CradleRemove`, and `TextureRemoval` will be available in the `build/bin` directory.


== Usage ==

Three executable are built, mainCradleRemoval and mainTextureRemoval, demonstrating the 
two functionalities of the code in separate executable, and mainDemo that performs full
scale cradle and wood-grain separation across the entire X-ray.

# mainCradleRemoval

This piece of demo code demonstrates cradle removal for an image specified by the user.
Given an X-ray image of a painting containing the cradle artefact, the code attempts to remove
it from the image. It first detects the approximate position of the cradle pieces and then
fits a multiplicative correction model that attempts to rectify the pixel intensity of
cradled pieces.
NOTE: we do not check if the image specified does not exist, and the code will result in 
an error.

Parameters:
*   <infilename>				path+filename of input image to be processed
*   <outfilename>				path+filename of output images (optional; defaults to "out" if not specified)
*   <nrh> and <nrv>				number of horizontal and vertical pieces in the cradled image (optional)

Output:
*  <outfilename>_original.png			original, unprocessed image file saved as grayscale png
*  <outfilename>_nointensity.png		x-ray with cradle removed, saved as grayscale png
*  <outfilename>_cradle.png			cradle component of the separation, saved as grayscale png
*  <outfilename>_mask.png			cradle mask file, marking location of cradle pieces
*  <outfilename>.msf				an auxiliary file containing processing information, usable for texture removal

The code handles RGB input (by converting it to grayscale), but all output will be grayscale.


# mainTextureRemoval

This piece of demo code demonstrates a refining step in the cradle removal process where
the wood-grain component of the cradle is targeted. The intensity adjustment still
leaves texture parts of the cradle in the processed image, namely, wood-grain structures
that are not part of the painting or the supporting wooden panel. Using a Bayesian supervised
learning technique, a statistical model is built that attempts to separate the wood-grain
component from the X-ray image.

NOTE: this code requires an already processed X-ray image file, as the one provided by mainCradleRemoval 
and the *.msf file that contains all processing information. This demo code was set up to run
in combination with mainCradleRemoval, but upon tweaking the code one can separate the two 
functionalities from one another.

Parameters:
*   <infilename>		path+filename of input image to be processed
*   <centerX>			X coordinate of the 464x464 block where wood-grain separation should be executed
*   <centerY>			Y coordinate of the 464x464 block where wood-grain separation should be executed

Output:
*  "inc.png"		a 464×464 crop of the original
*  "out.png"		result after cradle removal, without wood-grain separation
*  "out2.png"		result after cradle removal and wood-grain separation
*  "cradle.png"		cradle component, without wood-grain separation
*  "cradle2.png"	cradle component, after wood-grain separation

# mainDemo

This piece of demo code demonstrates the combined cradle removal and wood-grain separation 
for an image specified by the user.
If the image specified does not exist, the code results in an error.

Parameters:
*   arg[1]				path+filename of input image to be processed

Output:
*  "solution.png"				x-ray with cradle and wood-grain removed, saved as grayscale png
*  "difference.png"				cradle component, before wood-grain removal
*  "difference_textrem.png"		cradle component, after wood-grain removal

== Use examples ==

For a full-scale cradle and wood-grain separation, run mainDemo with the input image as argument
Example:

./mainCradleRemoval /img/Ghissi.png

Note that this is could take very long, so for a fast execution of the demo code run files
mainCradleRemoval and mainTextureRemoval one after the other.
Example:

#with specifying number of cradle pieces
./mainCradleRemoval /img/Ghissi.png outfile 1 1

#texture separation in a 464x464 block centered at position (200,200)
./mainTextureRemoval outfile 200 200

OR

#without specifying number of cradle pieces
./mainCradleRemoval /img/stjerome_detail1.png outfile2
#texture separation in a 464x464 block centered at position (200,200)
./mainTextureRemoval outfile2 400 500
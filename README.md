# cs585_project

The initial code here contains algorithms to create seams in an image as described in the [Avidan Paper](http://www.cs.bu.edu/faculty/betke/cs585/restricted/papers/Avidan-SeamCarving-2007.pdf).

The structure is currently set up to run on OSX with OpenCV installed with Homebrew. You should be able to edit `code/CMakeList.txt` and update `OpenCV_LIBS_DIR` and `OpenCV_INCLUDE_DIRS` with your OpenCV install directory, then compile and run with:

    $ cmake .
    $ make
    $ ./main

It should be fairly straightforward to copy this code over to run with VisualStudio.

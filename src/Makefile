# swap the following two lines to change between debug and release compilers
CXX=g++ -std=c++11 -g  # debug
CXX=g++ -std=c++11 -O3 # release

# path to the opencv installation (particular to a specific setting)
OPENCV=/opt/homebrew/Cellar/opencv/4.9.0_8/lib/pkgconfig
OPENCVPC=$(OPENCV)/opencv4.pc

# compiler options (should be fairly general)
CXXFLAGS=$(shell pkg-config $(OPENCVPC) --cflags)
LDFLAGS=$(shell pkg-config $(OPENCVPC) --libs) -Wl#,-rpath=$(OPENCV)/lib/

# no need to change anything below this line
OBJ=CradleFunctions.o DWT.o FDCT.o FFST.o HaarDWT.o MCA.o Shearlet.o TextureRemoval.o mainCradleRemoval.o
OBJ2=CradleFunctions.o DWT.o FDCT.o FFST.o HaarDWT.o MCA.o Shearlet.o TextureRemoval.o mainTextureRemoval.o
OBJ3=CradleFunctions.o DWT.o FDCT.o FFST.o HaarDWT.o MCA.o Shearlet.o TextureRemoval.o mainDemo.o

all: mainCradleRemoval mainTextureRemoval mainDemo

mainCradleRemoval: $(OBJ)
	$(CXX) $(OBJ) -o mainCradleRemoval $(LDFLAGS)
	
mainTextureRemoval: $(OBJ2)
	$(CXX) $(OBJ2) -o mainTextureRemoval  $(LDFLAGS)
	
mainDemo: $(OBJ3)
	$(CXX) $(OBJ3) -o mainDemo  $(LDFLAGS)
	
clean:
	rm -f $(OBJ) mainCradleRemoval
	rm -f $(OBJ2) mainTextureRemoval
	rm -f $(OBJ3) mainDemo
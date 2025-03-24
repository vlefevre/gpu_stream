ifndef it
	it=10
endif

ifndef wu
	wu=1
endif

ifndef sm
	sm=80
endif

all:
	nvcc main.cu -o gpu_stream -DWARMUPS=$(wu) -DITERS=$(it) -arch=sm_$(sm)

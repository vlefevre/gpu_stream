ifndef it
	it=10
endif

ifndef wu
	wu=1
endif

all:
	nvcc main.cu -o gpu_stream -DWARMUPS=$(wu) -DITERS=$(it)

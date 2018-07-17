####################################

all: dot.exec \
     pi.exec \
     norm.exec
####################################


## Compile dot_prod
dot.exec: dot_prod.cpp
	g++ dot_prod.cpp -fopenmp -o dot.exec

## compile serial_pi
pi.exec: serial_pi.cpp
	g++ serial_pi.cpp -fopenmp -o pi.exec

## compile serial_pi
norm.exec: normalize_vec.cpp
	g++ normalize_vec.cpp -fopenmp -o norm.exec

clean:
	rm -rf *.exec
